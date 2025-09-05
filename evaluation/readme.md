# Benchmarking

We provide scripts to automatically compute objective metrics (WER, SIM-O, FPC, MCD, A-Sim, E-Sim) between generated audio samples and ground-truth LibriQuote-test samples.

## Dependancies & Installation
First, install the required python dependancies. You can do so with the following:

```
python3 -m venv /tmp/libriquote_eval/
source /tmp/libriquote_eval/bin/activate
pip install -r requirements.txt
```

Make sure you download the speaker verification model `wavlm_large_finetune.pth`. You can find it [here](https://huggingface.co/datasets/libriquote/libriquote_submission/resolve/main/models/wavlm_large_finetune.pth?download=true).

## Scripts
We provide  a `libriquote.lst` file that can be used to for further evaluation.

Test audios (targets and prompts) are available on the [Huggingface repository](https://huggingface.co/datasets/gasmichel/LibriQuote/tree/main/test_audios).


```sh
# build the file mapper
python get_wav_res_ref_text.py \
PATH_TO_LIBRIQUOTE \ # root folder of LibriQuote
libriquote.lst \
PATH_TO_MODEL_GENERATIONS \ # folder where generations are saved
PATH_TO_RESULTS/input.txt \ # path where results will be stored
.wav # file format of your generations

# FPC & MCD
python run_fpc_mcd.py \
PATH_TO_RESULTS/input.txt \
PATH_TO_RESULTS/fpc_mcd.txt  \

# WER
python run_wer.py \
PATH_TO_RES/input.txt \
PATH_TO_RES/wer/res \ # path to results
1 # number of parallel GPU processes

python average_wer.py \
PATH_TO_RES/wer/res.txt \
PATH_TO_RES/wer/res.wer

# SIM-O
python run_sim.py \
PATH_TO_RES/input.txt \
PATH_TO_RES/sparktts.sim.out \ # results will be stored here
PATH_TO_WAVLM\ # path to the `wavlm_large_fine.pth` file
1\ # number of parallel GPU processes

# A-Sim
python main_accents.py \
PATH_TO_LIBRIQUOTE/test_audios/wavs/ \ # path to target wavs
accent_outputs/targets.json \ # will save the embeddings here
0 \ # cuda device to use
.flac # file extension 

python main_accents.py \
PATH_TO_MODEL_GENERATIONS \ 
accent_outputs/MODEL.json \
0 \ # cuda device to use
.flac 

python agg.py \
accent_outputs/ \ # path to folder containing the saved embeddings
targets.json \ # saved embeddings to compare with
accent_sim.json # output file

# E-Sim
python main_emo.py \
PATH_TO_LIBRIQUOTE/test_audios/wavs/ \
emotion_outputs/targets.json \
0 \ # cuda device to use
.flac

python main_emo.py \
PATH_TO_MODEL_GENERATIONS \
emotion_outputs/MODEL.json \
0 \
.wav

python agg.py \
emotion_outputs/ \ # path to folder containing the saved embeddings
targets.json \ # saved embeddings to compare with
emotion_sim.json # output file
```