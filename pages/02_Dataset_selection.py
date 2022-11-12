import streamlit as st
import pandas as pd

st.set_page_config(page_title="Data Selection", page_icon="📐")

st.markdown("# Dataset Selection")
st.write(
    """Lorem ipsum!"""
)

df = pd.read_csv("data/datasets.csv")

st.write(df)