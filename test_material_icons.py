"""
Test script to verify Material Icons are loading correctly from local files.
Run this with: streamlit run test_material_icons.py
"""
import streamlit as st
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.keyword_code.utils.ui_helpers import apply_ui_styling

# Apply the styling with local Material Icons
apply_ui_styling()

st.title("Material Icons Test")
st.write("If the icons below display correctly (not as text), the local font loading is working!")

st.write("---")

# Test expanders (should show arrow icons, not text)
with st.expander("🔽 Test Expander 1 - Click to expand"):
    st.write("This expander should show a proper arrow icon, not text like 'keyboard_double_arrow_right'")

with st.expander("📁 Test Expander 2 - Another test", expanded=True):
    st.write("When you collapse this, you should see an arrow icon")

st.write("---")

# Test with columns
col1, col2 = st.columns(2)
with col1:
    with st.expander("Left Expander"):
        st.write("Test content left")

with col2:
    with st.expander("Right Expander"):
        st.write("Test content right")

st.write("---")
st.success("✅ If all expander arrows display correctly (not as text), Material Icons are working!")
st.info("ℹ️ If you see text like 'keyboard_double_arrow_right' instead of arrows, the fix needs adjustment")

# Display font file status
st.write("---")
st.subheader("Font File Status")

font_dir = os.path.join(os.path.dirname(__file__), "src", "keyword_code", "assets", "fonts")
regular_font = os.path.join(font_dir, "MaterialIcons-Regular.woff2")
outlined_font = os.path.join(font_dir, "MaterialIconsOutlined-Regular.woff2")

if os.path.exists(regular_font):
    size = os.path.getsize(regular_font)
    st.success(f"✅ MaterialIcons-Regular.woff2 found ({size:,} bytes)")
else:
    st.error(f"❌ MaterialIcons-Regular.woff2 NOT FOUND at {regular_font}")

if os.path.exists(outlined_font):
    size = os.path.getsize(outlined_font)
    st.success(f"✅ MaterialIconsOutlined-Regular.woff2 found ({size:,} bytes)")
else:
    st.error(f"❌ MaterialIconsOutlined-Regular.woff2 NOT FOUND at {outlined_font}")
