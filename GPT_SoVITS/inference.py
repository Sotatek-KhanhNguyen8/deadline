import random
import os, re, logging, sys
import pdb
import torch
from pydub import AudioSegment
import numpy as np
from TTS_infer_pack.TTS import TTSA, TTSA_Config
from TTS_infer_pack.text_segmentation_method import get_method
from tools.i18n.i18n import I18nAuto
import soundfile as sf
import time

is_half = False
gpt_path = './pretrained_models/GPT_weights/ellen_best_gpt.ckpt'
sovits_path = './pretrained_models/SoVITS_weights/ellen_best_sovits.pth'
bert_path = './pretrained_models/chinese-roberta-wwm-ext-large'
cnhubert_base_path = './pretrained_models/chinese-hubert-base'

i18n = I18nAuto()
device = "cpu"

tts_config = TTSA_Config("./GPT_SoVITS/configs/tts_infer.yaml")
tts_config.device = device
tts_config.is_half = is_half
if gpt_path is not None:
    tts_config.t2s_weights_path = gpt_path
if sovits_path is not None:
    tts_config.vits_weights_path = sovits_path
if cnhubert_base_path is not None:
    tts_config.cnhuhbert_base_path = cnhubert_base_path
if bert_path is not None:
    tts_config.bert_base_path = bert_path

tts_pipline = TTSA(tts_config)

def inference(text, text_lang,
              ref_audio_path, prompt_text,
              prompt_lang, top_k,
              top_p, temperature,
              text_split_method, batch_size,
              speed_factor, ref_text_free,
              split_bucket,fragment_interval,
              seed,
              ):
    actual_seed = seed if seed not in [-1, "", None] else random.randrange(1 << 32)
    inputs={
        "text": text,
        "text_lang": text_lang,
        "ref_audio_path": ref_audio_path,
        "prompt_text": prompt_text if not ref_text_free else "",
        "prompt_lang": prompt_lang,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "text_split_method": text_split_method,
        "batch_size":int(batch_size),
        "speed_factor":float(speed_factor),
        "split_bucket":split_bucket,
        "return_fragment":False,
        "fragment_interval":fragment_interval,
        "seed":actual_seed,
    }
    print(inputs)
    for item in tts_pipline.run(inputs):
        yield item, actual_seed

text = "Hello what the fuck. what is your name"    #input text
text_language = "en"           #select "en","all_zh","all_ja"
inp_ref = "../audio/ellen_reference.wav"   #path of reference speaker
prompt_text = "and say, no, no, it's okay, he only smokes when he drinks. During the monologue, what I'll do is I'll just talk to an audience member."               #text of reference speech
prompt_language = "en"         #reference speech language
batch_size = 100              #inference batch size
speed_factor = 1.0             #control speed of output audio
top_k = 5                      #gpt
top_p = 1
temperature = 1
how_to_cut = "cut4"            #"cut0": not cut   "cut1": 4 sentences a cut   "cut2": 50 words a cut   "cut3": cut at chinese '。'  "cut4": cut at english '.'   "cut5": auto cut
ref_text_free = False
split_bucket = True            #suggest on
fragment_interval = 0.07     #interval between every sentence
seed = 233333               #seed

infer_time = time.time()
[output] = inference(text,text_language, inp_ref,
                prompt_text, prompt_language,
                top_k, top_p, temperature,
                how_to_cut, batch_size,
                speed_factor, ref_text_free,
                split_bucket,fragment_interval,
                seed)
print(f"Inference time: {time.time() - infer_time:.2f} seconds")

sf.write('./output21121212121.wav', output[0][1], samplerate=output[0][0], subtype='PCM_16')
