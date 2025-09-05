from speechbrain.pretrained import EncoderClassifier
# import torch.multiprocessing as mp 
import os
import torch
from tqdm.auto import tqdm
import sys
import numpy as np 
import json
import datetime
import glob 
import librosa

if __name__ == "__main__" : 
    inp_dir = sys.argv[1]
    out_file = sys.argv[2]
    device = sys.argv[3]
    ext = sys.argv[4]
    sr = 16000

    out_dir = os.path.dirname(out_file) 
    os.makedirs(out_dir, exist_ok=True)

    classifier = EncoderClassifier.from_hparams(source="Jzuluaga/accent-id-commonaccent_ecapa", savedir="pretrained_models/accent-id-commonaccent_ecapa")
    classifier = classifier.eval().to(torch.device(f'cuda:{device}'))
    classifier.device=f'cuda:{device}'

    out = {}
    with torch.no_grad() :
        files = glob.glob(os.path.join(inp_dir, f'*{ext}'))
        for idx in tqdm(range(0, len(files))):
            wavs = []
            ff = files[idx]
            _, fname = os.path.split(ff)
            fname = fname[:-len(ext)]
            wavs,sr  = librosa.load(ff, sr=16000)
                
            x = classifier.encode_batch(torch.tensor(wavs).unsqueeze(0).to(torch.device(f'cuda:{device}')))
            out[fname] = {'feats' : (x.cpu().flatten().numpy().tolist())}
        
    with open(out_file, "w" ) as f : 
        json.dump(out,f)
    


