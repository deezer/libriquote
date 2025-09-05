from funasr import AutoModel
# import torch.multiprocessing as mp 
import os

from tqdm.auto import tqdm
import sys
import numpy as np 
import json
import datetime
import glob 

if __name__ == "__main__" : 
    inp_dir = sys.argv[1]
    out_file = sys.argv[2]
    device = sys.argv[3]
    ext = sys.argv[4]

    os.environ["CUDA_VISIBLE_DEVICES"] = device

    out_dir = os.path.dirname(out_file) 
    os.makedirs(out_dir, exist_ok=True)
    
    now = datetime.datetime.now().strftime("%d%m%H%M%s") 
    o =os.path.splitext(out_file)[0]
    out_f = os.path.split(o)[-1]
    tmp_file = f"/tmp/{out_f}-{now}.scp"
    with open(tmp_file, "w") as f : 
        for ff in tqdm(glob.glob(f'{inp_dir}/*{ext}') ): 
            _, fname = os.path.split(ff)
            fname = fname[:-len(ext)]
            f.write(fname + "\t" + ff + "\n")
    # world_size = 6
    model = AutoModel(model="iic/emotion2vec_plus_base")
    out = {}
    res = model.generate(tmp_file, granularity="utterance", extract_embedding=True)
    
    for r in res: 
        pred = np.argmax(r["scores"])
        if pred != 8 :
            pred_l = r["labels"][pred].split("/")[1]
        else : 
            pred_l = "<unk>"
        fname  = r["key"]
        out[fname] = {"feats" : r["feats"].tolist(), "scores": r["scores"], "labels" : r["labels"], "pred" : pred_l}

    with open(out_file, "w" ) as f : 
        json.dump(out,f)
    



