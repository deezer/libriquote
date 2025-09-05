import sys, os

metalst = sys.argv[2]
path_to_libriquote = sys.argv[1]
wav_dir = sys.argv[3]
wav_res_ref_text = sys.argv[4]
ext = sys.argv[5]

f = open(metalst)
lines = f.readlines()
f.close()

if "|" in lines[0] : 
    sep = "|"
else : 
    sep = "\t" 

print(sep)
path, file = os.path.split(wav_res_ref_text)
if path != "" :
    os.makedirs(path, exist_ok=True)

f_w = open(wav_res_ref_text, 'w')
for line in lines:
    utt, infer_text = line.strip().split(sep)

    if not os.path.exists(os.path.join(wav_dir, utt + ext)):
        print(f'Could not find file {os.path.join(wav_dir, utt + ext)}')
        continue

    gt_wav = os.path.join(path_to_libriquote, 'test_audios/wavs/', utt+'.flac')
    if not os.path.isfile(gt_wav) : 
        print(f'Could not find file {gt_wav}')

    out_line = '|'.join([os.path.join(wav_dir, utt + ext), gt_wav, infer_text])
    f_w.write(out_line + '\n')
f_w.close()
