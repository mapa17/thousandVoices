# This code is originally from https://colab.research.google.com/github/espnet/notebook/blob/master/tts_realtime_demo.ipynb#scrollTo=9gGRzrjyudWF

import pickle
import numpy as np
import wave
import yaml
from pathlib import Path
from typing import Sequence
import itertools

import time
import nltk
nltk.download('punkt')

# add path
import sys
sys.path.append("../../../espnet")

#from TTS.basemodel.Tacotron2_ParallelWaveGan import trans_type

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
import torch.nn as nn
#device = torch.device("cuda")

class ThousandVoices(nn.Module):
    def __init__(self,
            model_path: Path,
            dict_path: Path,
            vocoder_path: Path,
            vocoder_conf: Path,
            xvectors_path: Path,
            device: torch.device, trans_type: str = "phn"):
        super().__init__()

        # Load TTS model
        idim, odim, train_args = get_model_conf(model_path)
        model_class = dynamic_import(train_args.model_module)
        model = model_class(idim, odim, train_args)
        torch_load(model_path, model)
        model = model.eval().to(device)

        # Load vocoder 
        with open(vocoder_conf) as f:
            config = yaml.load(f, Loader=yaml.Loader)

        vocoder_class = config.get("generator_type", "ParallelWaveGANGenerator")
        vocoder = getattr(parallel_wavegan.models, vocoder_class)(**config["generator_params"])
        vocoder.load_state_dict(torch.load(vocoder_path, map_location="cpu")["model"]["generator"])
        vocoder.remove_weight_norm()
        vocoder = vocoder.eval().to(device)

        # Pepare text preprocessing (frontend)
        if config["generator_params"]["out_channels"] > 1:
            from parallel_wavegan.layers import PQMF
            self.pqmf = PQMF(config["generator_params"]["out_channels"]).to(device)

        # define text frontend
        with open(dict_path) as f:
            lines = f.readlines()
        lines = [line.replace("\n", "").split(" ") for line in lines]
        char_to_id = {c: int(i) for c, i in lines}
        g2p = G2p()

        # Load speaker embeddings
        xvectors = pickle.load(open(xvectors_path, mode='rb'))

        inference_args = Namespace(**{
            "threshold": 0.5,"minlenratio": 0.0, "maxlenratio": 10.0,
            # Only for Tacotron 2
            "use_attention_constraint": True, "backward_window": 1,"forward_window":3,
            # Only for fastspeech (lower than 1.0 is faster speech, higher than 1.0 is slower speech)
            "fastspeech_alpha": 1.0,
            })

        self.tts_model = model
        self.vocoder_model = vocoder
        self.xvectors = xvectors
        self.config = config
        self.inference_args = inference_args
        self.trans_type = trans_type
        self.device = device
        self.idim = idim
        self.odim = odim
        self.char_to_id = char_to_id
        self.g2p = g2p


    def _speaker(self, spk: str = None):
        if spk is None:
            spk = np.random.choice(list(self.xvectors.keys())) 
        print(f'spk = {spk}')
        return self.xvectors[spk]


    def _frontend(self, text: str) -> torch.Tensor:
        """Clean text and then convert to id sequence."""
        text = custom_english_cleaners(text)
        
        if self.trans_type == "phn":
            text = filter(lambda s: s != " ", self.g2p(text))
            text = " ".join(text)
            print(f"Cleaned text (phn): {text}")
            charseq = text.split(" ")
        else:
            print(f"Cleaned text: {text}")
            charseq = list(text)
        idseq = []
        for c in charseq:
            if c.isspace():
                idseq += [self.char_to_id["<space>"]]
            elif c not in self.char_to_id.keys():
                idseq += [self.char_to_id["<unk>"]]
            else:
                idseq += [self.char_to_id[c]]
        idseq += [self.idim - 1]  # <eos>
        return torch.LongTensor(idseq).view(-1).to(self.device)


    def forward(self, input: Sequence[str], speaker: str = None) -> Sequence[torch.Tensor]:
        #sps = itertools.cycle(list(self._speaker(speaker)))
        sps = [self._speaker(speaker), ]

        pad_fn = torch.nn.ReplicationPad1d(
            self.config["generator_params"].get("aux_context_window", 0))
        vocoder_class = self.config.get("generator_type", "ParallelWaveGANGenerator")
        use_noise_input = vocoder_class == "ParallelWaveGANGenerator"

        ys = []
        for spemb, input_text in zip(sps, input):
            print(f'TTS Sentence: {input_text}')
            print(f'x-vector {spemb.shape}')
            with torch.no_grad():
                start = time.time()
                x = self._frontend(input_text)
                c, _, _ = self.tts_model.inference(x, self.inference_args, spemb=spemb.to(self.device))
                c = pad_fn(c.unsqueeze(0).transpose(2, 1)).to(self.device)
                xx = (c,)
                if use_noise_input:
                    z_size = (1, 1, (c.size(2) - sum(pad_fn.padding)) * self.config["hop_size"])
                    z = torch.randn(z_size).to(self.device)
                    xx = (z,) + xx
                if self.config["generator_params"]["out_channels"] == 1:
                    y = self.vocoder_model(*xx).view(-1)
                else:
                    y = self.pqmf.synthesis(self.vocoder_model(*xx)).view(-1)    
            rtf = (time.time() - start) / (len(y) / self.config["sampling_rate"])
            print(f"RTF = {rtf:5f}")
            ys.append(y)
        
        return ys

#from IPython.display import display, Audio
#for y in ys:
#  display(Audio(y.view(-1).cpu().numpy(), rate=config["sampling_rate"]))

#with open('utterance.pkl', 'wb') as fo:
#    pickle.dump(ys, fo)

#foutname = 'speaker'
#for i, y in enumerate(ys):
    #fo = f'{foutname}{i:02d}.wav'
    #print(f'Writing {fo} ...')
    #with wave.open(fo, mode='wb') as wout:
    #    # Set wave: Mono, 16Bit, 22khz, nFrames, Uncompressed
    #    wout.setparams((1, 2, config['sampling_rate'], y.shape[0], 'NONE', 'NONE'))
    #    wout.writeframes(y.view(-1).cpu().numpy())
