import streamlit as st

st.set_page_config(
    page_title="CNT SmartDocs",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)
from src.keyword_code.utils.helpers import get_base64_encoded_image

# Display SmartDocs logo
try:
    logo_base64 = get_base64_encoded_image("src/keyword_code/assets/smartdocslogo.png")
    if logo_base64:
        st.markdown(
            f"""
            <div style="text-align: center; margin-bottom: 20px;">
                <img src="data:image/png;base64,{logo_base64}" alt="CNT SmartDocs" style="max-width: 500px; width: 100%; height: auto;">
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Fallback to text if image fails to load
        st.markdown('<h1 style="text-align: center;"><span style="color: #002345;">CNT</span> <span style="color: #00ade4;">SmartDocs</span></h1>', unsafe_allow_html=True)
except Exception:
    # Fallback to text if image fails to load
    st.markdown('<h1 style="text-align: center;"><span style="color: #002345;">CNT</span> <span style="color: #00ade4;">SmartDocs</span></h1>', unsafe_allow_html=True)

st.markdown('<h3 style="text-align: center;">Welcome to CNT SmartDocs</h3>', unsafe_allow_html=True)

# Add description
st.markdown("""
This is the main application page. You can access the document analysis tool from the pages menu in the sidebar.
""")

with st.sidebar:
    st.write("Powered by CNT") 