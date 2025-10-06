import streamlit as st

st.set_page_config(
    page_title="CNT SmartDocs",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown('<h1 style="text-align: center;"><span style="color: #002345;">CNT</span> <span style="color: #00ade4;">SmartDocs</span></h1>', unsafe_allow_html=True)
st.markdown('<h3 style="text-align: center;">Welcome to CNT SmartDocs</h3>', unsafe_allow_html=True)

# Add description
st.markdown("""
This is the main application page. You can access the document analysis tool from the pages menu in the sidebar.
""")

with st.sidebar:
    st.write("Powered by CNT") 