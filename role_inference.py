import sys

sys.path.insert(0, 'GPT_SoVITS')

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

from GPT_SoVITS.TTS_infer_pack.TTS import TTSA, TTSA_Config


class VoiceClone:
    speaker_id = ''

    def __init__(self, is_half: bool = True):
        self._base_dir = Path(__file__).resolve().parent

        tts_config_path = self._base_dir / 'GPT_SoVITS' / 'configs' / f'{self.speaker_id}_infer.yaml'

        tts_config = TTSA_Config(str(tts_config_path.absolute()))
        tts_config.device = self.device
        tts_config.is_half = is_half

        gpt_path = self._base_dir / 'GPT_weights' / 'G:/download/Nu_tho_ngoc/tts-demo4/tts-demo/GPT_SoVITS/pretrained_models/GPT_weights/nagisa-e15.ckpt'
        tts_config.t2s_weights_path = str(gpt_path.absolute())

        sovits_path = self._base_dir / 'SoVITS_weights' / 'G:/download/Nu_tho_ngoc/tts-demo4/tts-demo/GPT_SoVITS/pretrained_models/SoVITS_weights/nagisa_e10_s310.pth'
        tts_config.vits_weights_path = str(sovits_path.absolute())

        cnhubert_base_path = self._base_dir / 'GPT_SoVITS' / 'pretrained_models' / 'chinese-hubert-base'
        tts_config.cnhuhbert_base_path = str(cnhubert_base_path.absolute())

        bert_path = self._base_dir / 'GPT_SoVITS' / 'pretrained_models' / 'chinese-roberta-wwm-ext-large'
        tts_config.bert_base_path = str(bert_path.absolute())

        self.voice_generate_pipeline = TTSA(tts_config)
        self.speaker_voice_reference = None

    @property
    def device(self):
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'

    @property
    def voice_parameters(self):
        return {
            "text_lang": 'en',
            # text of reference speech
            "prompt_text": "",
            "prompt_lang": 'en',
            "top_k": 5,
            "top_p": 1,
            "temperature": 1,
            "text_split_method": 'cut4',
            "batch_size": 100,
            "speed_factor": 1.0,
            "split_bucket": True,
            "return_fragment": False,
            "fragment_interval": 0.3,
            "seed": 233333
        }

    def inference(self, speech_text: str):
        sample_rate = 233333
        input_parameters = self.voice_parameters
        input_parameters.update(text=speech_text, seed=sample_rate, ref_audio_path=str(self.speaker_voice_reference))
        if self.speaker_id == 'ellen':
            silence_duration = np.random.uniform(0.4, 0.6)
        else:
            silence_duration = np.random.uniform(0.2, 0.5)

        for item in self.voice_generate_pipeline.run(input_parameters):
            sample_rate, audio_data, *_ = item
            # return audio_data, sample_rate
            silence = np.zeros(int(silence_duration * sample_rate), dtype=np.int16)
            return np.concatenate([audio_data, silence]), sample_rate

    def say(self, text, output_path=None):
        audio_data, sample_rate = self.inference(text)

        if audio_data is not None and output_path:
            # audio_data = np.array(audio_data)
            time_length = librosa.get_duration(y=audio_data, sr=sample_rate)
            sf.write(output_path, audio_data, samplerate=sample_rate, subtype='PCM_16')
            return time_length
        return 0


class EllenVoiceGenerator(VoiceClone):
    speaker_id = 'ellen'

    def __init__(self, is_half: bool = True):
        super().__init__(is_half)

        self.speaker_voice_reference = self._base_dir / 'dataset' / 'test_audio' / 'elon.mp3_0002882240_0003086400.wav'

    @property
    def voice_parameters(self):
        return {
            "text_lang": 'en',
            "prompt_text": "and say, no, no, it's okay, he only smokes when he drinks. During the monologue, what I'll do is I'll just talk to an audience member.",
            "prompt_lang": 'en',
            "top_k": 5,
            "top_p": 1,
            "temperature": 1,
            "text_split_method": 'cut4',
            "batch_size": 100,
            "speed_factor": 1.0,
            "split_bucket": True,
            "return_fragment": False,
            "fragment_interval": 0.07,
            "seed": 5646565
        }


class TrumpVoiceGenerator(VoiceClone):
    speaker_id = 'trump'

    def __init__(self, is_half: bool = True):
        super().__init__(is_half)

        self.speaker_voice_reference = 'G:/download/Nu_tho_ngoc/tts-demo4/tts-demo/audio/ellen_reference.wav'

    @property
    def voice_parameters(self):
        return {
            "text_lang": 'en',
            # text of reference speech
            "prompt_text": "And this week, we took historic action to continue delivering on that promise. We did so in one of the many proud industrial towns",
            "prompt_lang": 'en',
            "top_k": 79,
            "top_p": 1,
            "temperature": 1,
            "text_split_method": 'cut4',
            "batch_size": 100,
            "speed_factor": 1.03,
            "split_bucket": True,
            "return_fragment": False,
            "fragment_interval": 0.07,
            "seed": 233333
        }


class LxVoiceGenerator(VoiceClone):
    speaker_id = 'lx'

    def __init__(self, is_half: bool = True):
        super().__init__(is_half)

        self.speaker_voice_reference = self._base_dir / 'dataset' / 'test_audio' / 'lx_reference.wav'

    @property
    def voice_parameters(self):
        return {
            "text_lang": 'all_zh',
            # text of reference speech
            "prompt_text": "啊，你为什么谴责不正义，并不是怕做不正义的事，而是怕啥呢，而是怕吃不正义的亏。",
            "prompt_lang": 'zh',
            "top_k": 60,
            "top_p": 1,
            "temperature": 1,
            "text_split_method": 'cut5',
            "batch_size": 100,
            "speed_factor": 1.1,
            "split_bucket": True,
            "return_fragment": False,
            "fragment_interval": 0.1,
            "seed": 233333
        }


if __name__ == '__main__':
    # voice_generator = EllenVoiceGenerator()
    voice_generator = TrumpVoiceGenerator()
    speech_text = "Look, if I had to pick a weakness, it's that I'm too honest."

    texts = [
        "All right, plain and simple, I see what you mean.",
        "All right, Your point is well taken and I understand it.",
        "Okay, you've got something on your mind, and that's why we're here, isn't it?",
        "Well, you've brought something to the table, and that's what dialogue is all about, Let's take a look."
    ]

    for index, text in enumerate(texts):
        duration = voice_generator.say(text, f"./trump_{index}.wav")
        print('duration : ', duration)
