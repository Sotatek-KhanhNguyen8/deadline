import re
import sys
from pathlib import Path
from typing import List
import filelock
import streamlit as st
import streamlit.web.cli as streamlit_cli
from langchain_core.output_parsers.base import T
from langchain_core.outputs import Generation

import random
import time
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from GPT_SoVITS.conf import settings

from GPT_SoVITS.inference_enfu import add_text_to_audio_queue, init_tts_pipeline
from GPT_SoVITS.file_utils import HParams


class UtilsModule:
    HParams = HParams


sys.modules['utils'] = UtilsModule


print('>>>>>>>> page start......')
st.set_page_config(
    page_title="Chat",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
    }
)


def convert_uppercase_words_to_lowercase(text):
    # 使用正则表达式找到全大写单词
    uppercase_words = re.findall(r'\b[A-Z]+\b', text)

    # 将全大写单词转换为全小写
    for word in uppercase_words:
        text = text.replace(word, word.lower())

    return text


def convert_comma_separated_numbers(text):
    # 使用正则表达式找到包含逗号的数字
    comma_separated_numbers = re.findall(r'\b\d{1,3}(,\d{3})+\b', text)

    # 移除数字中的逗号
    for number in comma_separated_numbers:
        text = text.replace(number, number.replace(',', ''))

    return text


class CopyGenOutputParser(StrOutputParser):
    answer = ""
    end_of_sentence_markers = ('!', '?', '.')
    first_sentence_markers = ('!', '?', '.', ',')
    first_token_time = 0

    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U0001F900-\U0001F9FF"  # supplemental symbols and pictographs
        "]+", re.UNICODE
    )

    stars_pattern = re.compile(r'\*[\w\s]+\*', re.UNICODE)
    bracket_pattern = re.compile(r'\(*[\w\s]+\)', re.UNICODE)

    def remove_emojis(self, data):
        text = re.sub(self.stars_pattern, '', data)
        text = re.sub(self.bracket_pattern, '', text)
        text = re.sub(self.emoji_pattern, '', text).strip()
        return text.strip()

    def parse_answer(self, answer: str) -> str:
        ans = self.remove_emojis(answer)
        ans = convert_comma_separated_numbers(ans)
        an = convert_uppercase_words_to_lowercase(ans)
        if len(an) == 0:
            return an
        end_char = an[-1]
        an = an[:-1]
        res = an.replace('!', ',').replace('?', ',').replace('.', ',')
        return res + end_char

    def parse_result(self, result: List[Generation], *, partial: bool = False) -> T:
        if self.first_token_time == 0:
            self.first_token_time = time.time()
            print(f">>>> Time to first token: {self.first_token_time:.2f} seconds.")

        processed_data = super().parse_result(result, partial=partial)

        self.answer += processed_data
        self.answer = self.parse_answer(self.answer)
        if len(processed_data) == 0:
            print(f'>>>>>>>>> processed_data is None or empty...[{processed_data}]')
            print(f'********** answer ***********: {self.answer}')
            if len(self.answer) > 0:
                add_text_to_audio_queue(self.answer, time.time(), self.first_token_time)
                self.answer = ''
        else:
            if len(self.answer) > 0 and self.answer[-1] in self.first_sentence_markers and len(self.answer.split()) > 4:
                print(f'********** answer ***********: {self.answer}')
                add_text_to_audio_queue(self.answer, time.time(), self.first_token_time)
                self.answer = ''
        return processed_data


if "llamacpp_fd" not in st.session_state:
    st.session_state.llamacpp_fd = None
    st.session_state.llm_chain = None
    st.session_state.current_role = 'AI Assistant'
    st.session_state.au = None

if st.session_state.llamacpp_fd is None:
    with st.spinner("Initializing ..."):
        init_tts_pipeline()
    print(">>>>>>> Initializing LlamaCpp...")
    gguf_path = 'models/Funny-Meow/Llama-2-7b-chat-hf-Q4_K_M-GGUF/llama-2-7b-chat-hf-q4_k_m.gguf'
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    st.session_state.llamacpp_fd = LlamaCpp(
        model_path=str(gguf_path),
        streaming=True,
        n_gpu_layers=-1,
        n_batch=512,
        n_ctx=2048,
        f16_kv=True,
        callback_manager=callback_manager,
        verbose=False
    )
    print('-' * 60)

    st.session_state.using_voice = False
    st.session_state.prompt = ChatPromptTemplate.from_template(settings.default_llm_prompt)

    st.session_state.text_chain = st.session_state.prompt | st.session_state.llamacpp_fd | StrOutputParser()
    st.session_state.voice_chain = st.session_state.prompt | st.session_state.llamacpp_fd | CopyGenOutputParser()

    st.session_state.llm_chain = st.session_state.text_chain
    print("Initializing chat pipeline...")
    print(f'Prompt: {st.session_state.prompt}')
    for txt in st.session_state.llm_chain.stream({"topic": "hello"}):
        pass
    print('-' * 60)


st.sidebar.title("Options")
st.sidebar.divider()
# selected_role = st.sidebar.selectbox('Select a Role', ['Ellen', 'Trump'])
using_voice = st.sidebar.toggle("Using Voice", False)
st.sidebar.success("reading the answer automatically")
if st.session_state.using_voice != using_voice:
    st.session_state.using_voice = using_voice
    st.session_state.messages = []
    if st.session_state.using_voice:
        print('>>>>>> using voice output parser...')
        st.session_state.llm_chain = st.session_state.voice_chain
    else:
        print('>>>>>> using non voice output parser...')
        st.session_state.llm_chain = st.session_state.text_chain
    st.rerun()


selected_role = 'AI Assistant'
if st.session_state.current_role != selected_role:
    st.session_state.current_role = selected_role
    prompt = ChatPromptTemplate.from_template(settings.default_llm_prompt)
    st.session_state.llm_chain = prompt | st.session_state.llamacpp_fd | CopyGenOutputParser()
    st.session_state.messages = []

st.markdown("# Chat to TTS")
st.divider()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(st.session_state.llm_chain.stream({"topic": prompt}))
        # st.audio('./output/output_20240717_014356.wav', format='audio/wav', autoplay=False)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})


# listen_for_audio(audio_queue)

print('>>>>>>>> page end......')
if __name__ == '__main__':
    st_dir = Path.home() / '.streamlit' / 'credentials.toml'
    st_dir.parent.mkdir(parents=True, exist_ok=True)
    if not st_dir.exists():
        with st_dir.open('a+') as f:
            content = '''[general]
email=""
                '''
            f.write(content)

    # init_tts_pipeline()

    locker = filelock.FileLock(settings.SINGLE_INSTANCE_LOCKFILE.absolute())
    if not locker.is_locked:
        print('>>>>>> locker not locked...')
        with locker:
            sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
            sys.argv = [
                'streamlit', 'run', sys.argv[0],
                '--global.developmentMode=false',
                '--server.enableCORS=true',
                '--server.enableXsrfProtection=true',
            ]
            sys.exit(streamlit_cli.main())
