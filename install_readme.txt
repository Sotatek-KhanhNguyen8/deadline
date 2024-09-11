conda create -n GPTSoVits python=3.9
conda activate GPTSoVits

pip install -r requirements.txt
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia

generate wav files:  python GPT_SoVITS/inference.py
