from f0_corr import extract_fpc
from mcd import extract_mcd
from typing import List, Dict
import numpy as np
from multiprocessing import Pool
import sys 
from tqdm.auto import tqdm
import os

def parallelizable_one(args, logger=None) : 
    # audio_ref = args[0]
    # audio_deg = args[1]
    # if logger is not None : 
    #     print("Extracting FPC")
    audio_ref, audio_deg = args
    try : 
        fpc = extract_fpc(audio_ref, audio_deg, fs=16000)
    except Exception as e :
        print(e)
        fpc = -100
    # if logger is not None : 
    #     print("Extracting MCD")
    try : 
        mcd = extract_mcd(audio_ref, audio_deg, kwargs={"fs" : 16000})
    except ExceptionError as e :
        print(e)
        mcd = -100
        
    return audio_ref, audio_deg, fpc, mcd


if __name__ == "__main__" : 
    input_file = sys.argv[1]
    out_file = sys.argv[2]

    out_path, _ = os.path.split(out_file)
    if len(out_path) > 0 : 
        os.makedirs(out_path, exist_ok=True)
    
    metalst = open(input_file).readlines()

    args = []
    for line in metalst:
        line = line.strip()
        if len(line.split('|')) == 3: # for edit
            gen_path, gt_path, _ = line.split('|')
        else:
            raise NotImplementedError

        if not os.path.exists(gt_path):
            print("not found")
            continue    
        args.append((gt_path, gt_path))
    
    p = Pool(32) 

    fpcs = []
    mcds=[]
    with open(out_file, "w") as f : 
        for res in tqdm(p.imap_unordered(parallelizable_one, args), total = len(args)) : 
            audio_ref, audio_deg, fpc, mcd = res
            if fpc != -100 : 
                fpcs.append(fpc)
            if mcd != -100 :
                mcds.append(mcd)
            f.write(
                audio_ref + "\t" + audio_deg + "\t" + str(fpc) + "\t" + str(mcd) + "\n"
            )
    p.close()
    print("MCD: ", round(sum(mcds) / len(mcds), 2))
    print("FPC: ", round(sum(fpcs) / len(fpcs), 2))

            
            