import streamlit as st
import pandas as pd
from src.keyword_code.processors.pdf_processor import PDFProcessor
from src.keyword_code.utils.ui_helpers import apply_ui_styling

st.set_page_config(page_title="PDF Inspector", page_icon="🔍", layout="wide")
apply_ui_styling()

st.title("🔍 PDF Structure Inspector")

uploaded_file = st.file_uploader("Upload a PDF to inspect", type=["pdf"])

if uploaded_file:
    st.info("Processing PDF...")
    file_bytes = uploaded_file.read()
    
    # Process the PDF
    processor = PDFProcessor(file_bytes)
    chunks, full_text = processor.extract_structured_text_and_chunks()
    
    st.success(f"Extracted {len(chunks)} chunks.")
    
    # Convert chunks to DataFrame for easier viewing
    data = []
    for i, chunk in enumerate(chunks):
        meta = chunk.get("metadata", {})
        data.append({
            "Chunk ID": i,
            "Page": meta.get("page_number"),
            "Article": f"{meta.get('article_type', '')} {meta.get('article_number', '')}",
            "Article Title": meta.get("article_title"),
            "Section": meta.get("section_number"),
            "Section Title": meta.get("section_title"),
            "Text Preview": chunk.get("text", "")[:100] + "...",
            "Full Text": chunk.get("text", "")
        })
    
    df = pd.DataFrame(data)
    
    # Display summary stats
    st.subheader("Structure Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Chunks", len(chunks))
    with col2:
        st.metric("Unique Articles", df["Article"].nunique())
    with col3:
        st.metric("Unique Sections", df["Section"].nunique())
        

    # Full Inspection
    st.subheader("Full Chunk Inspection")
    
    # Search box
    search_term = st.text_input("Search in chunks text:")
    if search_term:
        filtered_df = df[df["Full Text"].str.contains(search_term, case=False)]
        st.dataframe(filtered_df)
    else:
        st.dataframe(df)

    # Detailed view with expanders
    st.subheader("Detailed View")
    num_to_show = st.slider("Number of chunks to show", 1, len(chunks), 10)
    
    for i in range(num_to_show):
        chunk = chunks[i]
        meta = chunk["metadata"]
        title = f"Chunk {i} | Page {meta.get('page_number')} | {meta.get('article_type')} {meta.get('article_number')} | {meta.get('section_number')}"
        with st.expander(title):
            st.text(chunk["text"])
            st.json(meta)

