import langchain_helper as lch
import streamlit as st

st.title("COMMUNICAITOR - Text Enhancement Playground")

raw_text = st.text_area("Enter your text here:", max_chars=3000)

if raw_text:
    response = lch.improve_text(raw_text=raw_text)
    st.write(response["raw_text"])
