import subprocess
import sys 
import torch.multiprocessing as mp 
import os 
import datetime


sys.path.append("thirdparty/UniSpeech/downstreams/speaker_verification/")


RANKS = {0: 0, 1:1, 2:2, 3: 3}

def main(rank, files, ckpt_path, dest_files) : 
    subprocess.run([
        "python",
        "thirdparty/UniSpeech/downstreams/speaker_verification/verification_pair_list_v2.py",
        files[rank],
        "--model_name", "wavlm_large",
        "--checkpoint", ckpt_path,
        "--scores", dest_files[rank],
        "--wav1_start_sr", "0",
        "--wav2_start_sr", "0",
        "--wav1_end_sr", "-1",
        "--wav2_end_sr", "-1",
        "--device", f"cuda:{RANKS[rank]}"
])
    
if __name__ == "__main__" : 
    wav_res_ref_txt = sys.argv[1]
    tgt_file = sys.argv[2]
    ckpt_path = sys.argv[3]
    world_size = int(sys.argv[4])

    path, file = os.path.split(tgt_file)
    if path != "" : 
        os.makedirs(path, exist_ok=True)

    now = datetime.datetime.now().strftime("%d-%m-%H-%M")
    tmp_file = "/tmp/full-" + now
    
    
    num_per_thread = len(open(wav_res_ref_txt).readlines()) // world_size + 1
    subprocess.run([
        "split" , "-l", str(num_per_thread), "-d", "--additional-suffix=.lst", wav_res_ref_txt, "/tmp/thread-"+now+"-"
    ])
    
    files = [f"/tmp/thread-{now}-0{i}.lst" for i in range(world_size)]
    dest_files = [f"/tmp/out-{now}-0{i}.out" for i in range(world_size)]

    mp.spawn(main , args=(files, ckpt_path, dest_files), nprocs=world_size)
    fout = open(tgt_file, "w")
    scores = []
    for f in dest_files : 
        with open(f) as f : 
            for line in f.readlines()[:-1] : 
                fout.write(line)
                scores.append(float(line.split("\t")[-1]))
    out = round(sum(scores) / len(scores), 3)
    print(f"avg score: {out}")
    fout.write(f"avg score: {out}")
    fout.flush()

    
    
