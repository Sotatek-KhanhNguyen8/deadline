import math
import os
import random
from transformers import HubertModel, HubertConfig
from transformers.onnx import export
from pathlib import Path
from transformers.onnx import export
from pathlib import Path

from transformers import (
    Wav2Vec2FeatureExtractor,
    HubertModel,
)
model = HubertModel.from_pretrained('G:/download/Nu_tho_ngoc/tts-demo4/tts-demo/GPT_SoVITS/pretrained_models/chinese-hubert-base')
config = HubertConfig.from_pretrained('G:/download/Nu_tho_ngoc/tts-demo4/tts-demo/GPT_SoVITS/pretrained_models/chinese-hubert-base')
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('G:/download/Nu_tho_ngoc/tts-demo4/tts-demo/GPT_SoVITS/pretrained_models/chinese-hubert-base')
onnx_path = Path("bert_model.onnx")
export(model=model, output=onnx_path, opset=13, config = config, preprocessor=feature_extractor)
