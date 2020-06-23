# This code is originally from https://colab.research.google.com/github/espnet/notebook/blob/master/tts_realtime_demo.ipynb#scrollTo=9gGRzrjyudWF

import pickle
import numpy as np
import wave
import yaml

import nltk
nltk.download('punkt')

# add path
import sys
sys.path.append("../../../espnet")

# define E2E-TTS model
from argparse import Namespace
from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import torch_load
from espnet.utils.dynamic_import import dynamic_import

# define neural vocoder
import parallel_wavegan.models

from tacotron_cleaner.cleaners import custom_english_cleaners
from g2p_en import G2p

# define device
import torch
#device = torch.device("cuda")
device = torch.device("cpu")

# set path
trans_type = "phn"
dict_path = "../artifacts/tacotron2/train_clean_460_units.txt"
model_path = "../artifacts/tacotron2/model.best"


# set path
vocoder_path = "../artifacts/parallelwavegan/model.pkl"
vocoder_conf = "../artifacts/parallelwavegan/config.yml"

xvectors_path = "../artifacts/xvectors.pkl"

xvectors = pickle.load(open(xvectors_path, mode='rb'))
k = list(xvectors.keys())
rndX = lambda : xvectors[np.random.choice(list(k))]

idim, odim, train_args = get_model_conf(model_path)
model_class = dynamic_import(train_args.model_module)
model = model_class(idim, odim, train_args)
torch_load(model_path, model)
model = model.eval().to(device)
inference_args = Namespace(**{
    "threshold": 0.5,"minlenratio": 0.0, "maxlenratio": 10.0,
    # Only for Tacotron 2
    "use_attention_constraint": True, "backward_window": 1,"forward_window":3,
    # Only for fastspeech (lower than 1.0 is faster speech, higher than 1.0 is slower speech)
    "fastspeech_alpha": 1.0,
    })


with open(vocoder_conf) as f:
    config = yaml.load(f, Loader=yaml.Loader)

vocoder_class = config.get("generator_type", "ParallelWaveGANGenerator")
vocoder = getattr(parallel_wavegan.models, vocoder_class)(**config["generator_params"])
vocoder.load_state_dict(torch.load(vocoder_path, map_location="cpu")["model"]["generator"])
vocoder.remove_weight_norm()
vocoder = vocoder.eval().to(device)

if config["generator_params"]["out_channels"] > 1:
    from parallel_wavegan.layers import PQMF
    pqmf = PQMF(config["generator_params"]["out_channels"]).to(device)

# define text frontend
with open(dict_path) as f:
    lines = f.readlines()
lines = [line.replace("\n", "").split(" ") for line in lines]
char_to_id = {c: int(i) for c, i in lines}
g2p = G2p()

def frontend(text):
    """Clean text and then convert to id sequence."""
    text = custom_english_cleaners(text)
    
    if trans_type == "phn":
        text = filter(lambda s: s != " ", g2p(text))
        text = " ".join(text)
        print(f"Cleaned text: {text}")
        charseq = text.split(" ")
    else:
        print(f"Cleaned text: {text}")
        charseq = list(text)
    idseq = []
    for c in charseq:
        if c.isspace():
            idseq += [char_to_id["<space>"]]
        elif c not in char_to_id.keys():
            idseq += [char_to_id["<unk>"]]
        else:
            idseq += [char_to_id[c]]
    idseq += [idim - 1]  # <eos>
    return torch.LongTensor(idseq).view(-1).to(device)

print("Now ready to synthesize!")


sps = [rndX() for _ in range(10)]
sentences = [f'This is speaker voice %d. Can you hear a difference?' % i for i in range(1, len(sps)+1)]

import time

pad_fn = torch.nn.ReplicationPad1d(
    config["generator_params"].get("aux_context_window", 0))
vocoder_class = config.get("generator_type", "ParallelWaveGANGenerator")
use_noise_input = vocoder_class == "ParallelWaveGANGenerator"

ys = []
for spemb, input_text in zip(sps, sentences):
  with torch.no_grad():
      start = time.time()
      x = frontend(input_text)
      c, _, _ = model.inference(x, inference_args, spemb=spemb.to(device))
      c = pad_fn(c.unsqueeze(0).transpose(2, 1)).to(device)
      xx = (c,)
      if use_noise_input:
          z_size = (1, 1, (c.size(2) - sum(pad_fn.padding)) * config["hop_size"])
          z = torch.randn(z_size).to(device)
          xx = (z,) + xx
      if config["generator_params"]["out_channels"] == 1:
          y = vocoder(*xx).view(-1)
      else:
          y = pqmf.synthesis(vocoder(*xx)).view(-1)    
  rtf = (time.time() - start) / (len(y) / config["sampling_rate"])
  print(f"RTF = {rtf:5f}")
  ys.append(y)

#from IPython.display import display, Audio
#for y in ys:
#  display(Audio(y.view(-1).cpu().numpy(), rate=config["sampling_rate"]))

with open('utterance.pkl', 'wb') as fo:
    pickle.dump(ys, fo)

#foutname = 'speaker'
#for i, y in enumerate(ys):
    #fo = f'{foutname}{i:02d}.wav'
    #print(f'Writing {fo} ...')
    #with wave.open(fo, mode='wb') as wout:
    #    # Set wave: Mono, 16Bit, 22khz, nFrames, Uncompressed
    #    wout.setparams((1, 2, config['sampling_rate'], y.shape[0], 'NONE', 'NONE'))
    #    wout.writeframes(y.view(-1).cpu().numpy())
