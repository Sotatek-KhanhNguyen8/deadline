import random
import time
import datetime
import soundfile as sf
import os
from pathlib import Path
from .TTS import TTSA, TTSA_Config

from pydub import AudioSegment
from pydub.playback import play

current_dir = Path(__file__).parent
parent_dir = current_dir.parent

_t2s_weights_path = parent_dir / "pretrained_models" / "GPT_weights" / "ellen_best_gpt.ckpt"
_vits_weights_path = parent_dir / "pretrained_models" / "SoVITS_weights" / "ellen_best_sovits.pth"
_cnhuhbert_base_path = parent_dir / "pretrained_models" / "chinese-hubert-base"
_bert_base_path = parent_dir / "pretrained_models" / "chinese-roberta-wwm-ext-large"

_infer_audio_ref = parent_dir / "a_infer_guide_audio" / "ellen_reference.wav"

class TTSModule:
    def __init__(self, config_path, device="cpu"):
        self.tts_config = TTSA_Config(config_path)
        self.tts_config.device = device
        self.tts_config.is_half = False
        self.tts_config.t2s_weights_path = str(_t2s_weights_path)
        self.tts_config.vits_weights_path = str(_vits_weights_path)
        self.tts_config.cnhuhbert_base_path = str(_cnhuhbert_base_path)
        self.tts_config.bert_base_path = str(_bert_base_path)
        self.tts_pipeline = TTSA(self.tts_config)

    def inference(self, text, text_lang, ref_audio_path, prompt_text, prompt_lang, top_k,
                  top_p, temperature, text_split_method, batch_size, speed_factor,
                  ref_text_free, split_bucket, fragment_interval, seed):
        actual_seed = seed if seed not in [-1, "", None] else random.randrange(1 << 32)
        inputs = {
            "text": text,
            "text_lang": text_lang,
            "ref_audio_path": ref_audio_path,
            "prompt_text": prompt_text if not ref_text_free else "",
            "prompt_lang": prompt_lang,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "text_split_method": text_split_method,
            "batch_size": int(batch_size),
            "speed_factor": float(speed_factor),
            "split_bucket": split_bucket,
            "return_fragment": False,
            "fragment_interval": fragment_interval,
            "seed": actual_seed,
        }
        print(inputs)
        for item in self.tts_pipeline.run(inputs):
            yield item, actual_seed

    def generate_audio(self, text, count, warmup=False):
        # 预设参数
        text_language = "en"
        inp_ref = str(_infer_audio_ref)
        prompt_text = "and say, no, no, it's okay, he only smokes when he drinks. During the monologue, what I'll do is I'll just talk to an audience member."
        prompt_language = "en"
        batch_size = 100
        speed_factor = 1.0
        top_k = 5
        top_p = 1
        temperature = 1
        how_to_cut = "cut4"
        ref_text_free = False
        split_bucket = True
        fragment_interval = 0.07
        seed = 233333

        [output] = self.inference(text, text_language, inp_ref,
                        prompt_text, prompt_language,
                        top_k, top_p, temperature,
                        how_to_cut, batch_size,
                        speed_factor, ref_text_free,
                        split_bucket,fragment_interval,
                        seed)

        # ad = AudioSegment(channels=1, sample_width=2, frame_rate=output[0][0], data=output[0][1].tobytes())
        # play(ad)
        # return 'done'

        if warmup:
            return 'warmup'

        print(f"Audio generated path : '{parent_dir}'")
        output_folder = "./output"  # 指定输出目录
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)  # 如果目录不存在，则创建它
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")  # 获取当前时间的时间戳
        file_name = os.path.join(output_folder, f'output_{now}_{count}.wav')
        sf.write(file_name, output[0][1], samplerate=output[0][0], subtype='PCM_16')

        print(f"Audio file saved to: {file_name}")
        return file_name




# 使用示例
# tts_module = TTSModule("./GPT_SoVITS/configs/tts_infer.yaml")
# text = "Well, you know what the say, Families are like fudge, mostly sweet, but sometimes nuts. My family is doing great, thanks for asking! My son is growing up to be a smart and handsome young man, just like his mom. He's currently working on his own talker show, which I'm sure will be even more hilarious than mine."
# tts_module.generate_audio(text)

