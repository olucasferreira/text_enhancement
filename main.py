import langchain_helper as lch
import streamlit as st

st.title("DEMO - Text Enhancement Feature")

raw_text = st.text_area("Enter your text here:", max_chars=1600)

if raw_text:
    response = lch.improve_text(raw_text=raw_text)
    st.write(response["raw_text"])
