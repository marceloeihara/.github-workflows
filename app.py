import os
import time
import tempfile
import streamlit as st
from streamlit import audio
from streamlit_chat import message
from chat_ollama import ChatPDF

from gtts import gTTS
from pydub import AudioSegment
from IPython.display import Audio, display

st.set_page_config(page_title="IABula - Sua nova forma de se medicar")

def display_messages():
    st.subheader("Chat")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        if is_user:
            message(msg, is_user=is_user, key=str(i), seed=100)
        else:
            message(msg, is_user=is_user, key=str(i), seed=11)
            print(msg)
            tts = gTTS(text=msg, lang='pt')
            tts.save("output.mp3")
            #audio = AudioSegment.from_file("output.mp3")
            #speed = 1.2
            #audio_with_speed = audio.speedup(playback_speed=speed)
            #audio_with_speed.export("output_speed.mp3", format="mp3")
            #display(Audio("output_speed.mp3", autoplay=True))
            audio("output.mp3", format="audio/mpeg", start_time=0, sample_rate=None, end_time=None, loop=False, autoplay=False)
            #message(display(Audio("output.mp3", autoplay=True)), is_user=is_user, key=str(i), seed=11)
    st.session_state["thinking_spinner"] = st.empty()

def process_input():
    if (
        st.session_state["user_input"]
        and len(st.session_state["user_input"].strip()) > 0
    ):
        user_text = st.session_state["user_input"].strip()
        with st.session_state["thinking_spinner"], st.spinner("Pensando..."):
            agent_text = st.session_state["assistant"].ask_com_db(user_text)

        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))

def read_and_save_file():
    st.session_state["assistant"].clear()
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""

    for file in st.session_state["file_uploader"]:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        with st.session_state["ingestion_spinner"], st.spinner(
            f"Carregando arquivo {file.name}"
        ):
            t0 = time.time()
            st.session_state["assistant"].ingest(file_path)
            t1 = time.time()

        st.session_state["messages"].append(
            (
                f"Arquivo {file.name} carregado em {t1 - t0:.2f} segundos",
                False,
            )
        )
        os.remove(file_path)


def page():
    if len(st.session_state) == 0:
        st.session_state["messages"] = []
        st.session_state["assistant"] = ChatPDF()

    st.header("IABula - Sua nova forma de se medicar")
    st.subheader("Informe o nome do remédio, peso, altura, idade, sexo (masc/femi) para consultar")

    st.subheader("Upload de documento")
    st.file_uploader(
        "Upload de documento",
        type=["pdf"],
        key="file_uploader",
        on_change=read_and_save_file,
        label_visibility="collapsed",
        accept_multiple_files=True,
    )

    st.session_state["ingestion_spinner"] = st.empty()

    display_messages()
    st.text_input("Mensagem", key="user_input", on_change=process_input)


if __name__ == "__main__":
    page()