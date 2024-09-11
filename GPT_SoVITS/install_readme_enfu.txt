conda create -n GPTSoVits python=3.9
conda activate GPTSoVits

pip install -r requirements.txt
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 -c pytorch

生成音频 python GPT_SoVITS/inference.py(若直接到GPT_SoVITS路径下跑inference可能需要解决一些路径问题）
新文件tts_add.py和inference3.py。
跑llama+tts命令：生成音频 python GPT_SoVITS/inference3.py(若直接到GPT_SoVITS路径下跑inference可能需要解决一些路径问题）


-------------------------------------------------------
conda create -n tts python=3.9

conda activate tts

pip3 install torch torchvision torchaudio

pip install -r requirements.txt

pip install langchain langchain-community

CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python

pip install -U numba



