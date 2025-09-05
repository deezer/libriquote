import sys, os
from tqdm import tqdm
import multiprocessing
from jiwer import process_words
from zhon.hanzi import punctuation
import string
import numpy as np
import soundfile as sf
import scipy
import torch.multiprocessing as mp 
import re 
from faster_whisper import WhisperModel
import glob


punctuation_all = punctuation + string.punctuation
wav_res_text_path = sys.argv[1]
res_path = sys.argv[2]
lang = sys.argv[3] # zh or en
# device = "cuda:0"

def load_en_model():
    model = WhisperModel("large-v3", device="cuda", compute_type="float32")
    return processor, model

def load_zh_model():
    model = AutoModel(model="paraformer-zh")
    return model

def merge(path="results", prefix="sparktts") :
    files = glob.glob(os.path.join(path, prefix + "_*.txt"))
    with open(os.path.join(path, prefix + ".txt"), "w") as f: 
        for file in files :
            data = open(file).readlines()
            f.writelines(data)


def process_one(orig_hypo, orig_truth):
    raw_truth = orig_truth
    raw_hypo = orig_hypo
    truth = orig_truth
    hypo = orig_hypo
    
    for x in punctuation_all:
        truth = truth.replace(x, '')
        hypo = hypo.replace(x, '')

    truth = truth.replace('  ', ' ')
    hypo = hypo.replace('  ', ' ')


    truth = truth.lower()
    hypo = hypo.lower()

    measures = process_words(truth, hypo)
    ref_list = truth.split(" ")
    wer = measures.wer
    subs = measures.substitutions / len(ref_list)
    dele = measures.deletions / len(ref_list)
    inse = measures.insertions / len(ref_list)
    return (raw_truth, raw_hypo, wer, subs, dele, inse)

def run_asr(rank, data, res_path):

    model = WhisperModel("large-v3", device=f"cuda", compute_type="float16")
    
    params = data[rank]
    fout = open(res_path + f"_{rank}.txt", "w")

    n_higher_than_50 = 0
    wers_below_50 = []
    for wav_res_path, text_ref in tqdm(params, desc = f"[WORKER {rank}]", total = len(params)):

        segments, _ = model.transcribe(wav_res_path, beam_size=5, language="en")
        hypo = ""
        for segment in segments:
            hypo = hypo + " " + segment.text

        raw_truth, raw_hypo, wer, subs, dele, inse = process_one(hypo, text_ref)
        fout.write(f"{wav_res_path}\t{wer}\t{raw_truth}\t{raw_hypo}\t{inse}\t{dele}\t{subs}\n")
        fout.flush()



RANKS = {0:"0", 1: "0", 2:"0", 3 : "0", 4: "1", 5:"1", 6 : '1', 7: '1'}

if __name__ == "__main__" : 
    wav_res_text_path = sys.argv[1]
    res_path = sys.argv[2]
    world_size = int(sys.argv[3])
    os.makedirs(os.path.dirname(res_path), exist_ok = True)
    
    params = []
    for line in open(wav_res_text_path).readlines():
        line = line.strip()
        if len(line.split('|')) == 2:
            wav_res_path, text_ref = line.split('|')
        elif len(line.split('|')) == 3:
            wav_res_path, wav_ref_path, text_ref = line.split('|')
        elif len(line.split('|')) == 4: # for edit
            wav_res_path, _, text_ref, wav_ref_path = line.split('|')
        else:
            raise NotImplementedError

        if not os.path.exists(wav_res_path):
            print("not found")
            continue
        params.append((wav_res_path, text_ref))

    batch_size = len(params) // world_size + 1 
    batches = []
    for idx in range(0, len(params) +1, batch_size) : 
        batches.append(params[idx:idx+batch_size])

    print(f"Processing {len(params)} files over {world_size} processes" )
    mp.spawn( run_asr, args=(batches, res_path), nprocs=world_size)

    dirname, file = os.path.split(res_path)
    merge(dirname, file)
