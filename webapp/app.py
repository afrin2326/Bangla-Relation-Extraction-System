# ============================================
# webapp/app.py - Streamlit Frontend
# ============================================

import sys
import io

# Fix encoding for Bengali
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Bangla Relation Extraction System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# Custom CSS
# ============================================

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(45deg, #FF4B2B, #FF416C);
        padding: 25px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 25px;
    }
    .main-header h1 {
        font-size: 2.5rem;
        margin: 0;
    }
    .main-header p {
        font-size: 1.1rem;
        margin: 5px 0 0 0;
        opacity: 0.9;
    }
    .stat-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border-left: 5px solid #FF4B2B;
        margin-bottom: 15px;
        height: 100%;
    }
    .stat-card h4 {
        color: #666;
        margin: 0 0 5px 0;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stat-card h2 {
        margin: 0;
        font-size: 2rem;
    }
    .stat-card p {
        margin: 5px 0 0 0;
        color: #888;
        font-size: 0.9rem;
    }
    .result-card {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        margin: 10px 0;
    }
    .result-card h4 {
        color: #666;
        margin: 0 0 5px 0;
        font-size: 0.85rem;
        text-transform: uppercase;
    }
    .relation-badge {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 20px;
        color: white;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .stButton>button {
        background: linear-gradient(45deg, #FF4B2B, #FF416C);
        color: white;
        border-radius: 25px;
        height: 50px;
        font-weight: bold;
        border: none;
        transition: 0.3s;
        width: 100%;
        font-size: 1rem;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 10px 20px rgba(255,75,43,0.2);
    }
    .footer {
        text-align: center;
        color: #888;
        padding: 20px 0;
        font-size: 0.8rem;
        border-top: 1px solid #eee;
        margin-top: 30px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# Helper function to decode Bengali text
# ============================================

def decode_bengali(text):
    """Decode Bengali text from various encodings"""
    if not isinstance(text, str):
        return str(text)
    
    import re
    if re.search(r'[\u0980-\u09FF]', text):
        return text
    
    try:
        return text.encode('latin1').decode('utf-8')
    except:
        pass
    
    try:
        return text.encode('cp1252').decode('utf-8')
    except:
        pass
    
    return text

# ============================================
# Sidebar
# ============================================

with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/search.png", width=120)
    st.title("🎯 Bangla Relation Extraction System")
    st.markdown("*Relation Extraction for Bengali Text*")
    st.markdown("---")
    
    # Navigation
    page = st.radio(
        "📌 Navigation",
        [
            "🏠 Home",
            "🔍 Real-time Analysis",
            "📁 Bulk Processing",
            "📄 File Upload",
            "📊 Analytics",
            "📈 Model Performance"
        ]
    )
    
    st.markdown("---")
    
    # API Configuration
    api_url = st.text_input(
        "🔗 API Endpoint",
        value="http://localhost:8000"
    )
    
    # API Connection Status
    try:
        response = requests.get(f"{api_url}/health", timeout=2)
        if response.status_code == 200:
            data = response.json()
            if data.get('model_loaded'):
                st.success("✅ API Connected & Model Loaded")
            else:
                st.warning("⚠️ API Connected but Model Not Loaded")
        else:
            st.warning("⚠️ API Not Responding")
    except:
        st.error("❌ API Offline")
        st.info("💡 Run: uvicorn api.main:app --reload")
    
    st.markdown("---")
    
    # Available Relations - Hardcoded
    st.markdown("### 🏷️ Available Relations")
    hardcoded_relations = [
        "🏢 প্রতিষ্ঠানের অবস্থান",
        "🕊️ মৃত্যুস্থান",
        "🎂 জন্মস্থান",
        "🎬 চলচ্চিত্র পরিচালক",
        "🎭 চলচ্চিত্র অভিনেতা",
        "🏗️ প্রতিষ্ঠাতা",
        "📝 লেখক"
    ]
    for rel in hardcoded_relations:
        st.markdown(f"- {rel}")

# ============================================
# Page: Home
# ============================================

if page == "🏠 Home":
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Bangla Relation Extraction System</h1>
        <p>Relation Extraction for Bengali Text using BanglaBERT</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="stat-card">
            <h4>📊 Relations</h4>
            <h2>7</h2>
            <p>Movie Actor, Director, Writer, etc.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stat-card">
            <h4>📚 Dataset</h4>
            <h2>63K+</h2>
            <p>Annotated Bengali texts</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stat-card">
            <h4>🧠 Model</h4>
            <h2>BanglaBERT</h2>
            <p>110M parameters</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="stat-card">
            <h4>⚡ Accuracy</h4>
            <h2>54.2%</h2>
            <p>Baseline for Bangla-REX</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 🚀 How It Works")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background:#f8f9fa;padding:20px;border-radius:10px;text-align:center;height:100%;">
            <h2>📝</h2>
            <h4>1. Input Bengali Text</h4>
            <p>Paste any Bengali sentence containing entities</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background:#f8f9fa;padding:20px;border-radius:10px;text-align:center;height:100%;">
            <h2>🤖</h2>
            <h4>2. AI Processing</h4>
            <p>BanglaBERT analyzes the text</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background:#f8f9fa;padding:20px;border-radius:10px;text-align:center;height:100%;">
            <h2>🎯</h2>
            <h4>3. Get Result</h4>
            <p>See relation with confidence score</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 📊 Example Relations")
    
    examples = [
        ("সত্যজিৎ রায় কলকাতায় জন্মগ্রহণ করেন", "Place of Birth"),
        ("শাকিব খান 'চালবাজ' ছবিতে অভিনয় করেছেন", "Movie Actor"),
        ("হুমায়ূন আহমেদ 'নন্দিত নরকে' লিখেছেন", "Writer"),
        ("সত্যজিৎ রায় 'পথের পাঁচালী' পরিচালনা করেছেন", "Movie Director"),
        ("শামসুল হুদা গ্রামীণফোন প্রতিষ্ঠা করেছেন", "Company Founder")
    ]
    
    for text, relation in examples:
        st.markdown(f"""
        <div style="background:#f8f9fa;padding:10px 15px;border-radius:8px;margin:5px 0;border-left:4px solid #FF4B2B;">
            <strong>{text}</strong>
            <span style="float:right;background:#FF4B2B;color:white;padding:2px 12px;border-radius:12px;font-size:0.8rem;">{relation}</span>
        </div>
        """, unsafe_allow_html=True)

# ============================================
# Page: Real-time Analysis
# ============================================

elif page == "🔍 Real-time Analysis":
    st.markdown("## 🔍 Real-time Relation Analysis")
    st.markdown("Enter a Bengali sentence to extract the relation between entities")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_input = st.text_area(
            "📝 Bengali Text:",
            placeholder="যেমন: সত্যজিৎ রায় কলকাতায় জন্মগ্রহণ করেন",
            height=150
        )
        
        if st.button("🔍 Extract Relation", use_container_width=True):
            if user_input.strip():
                with st.spinner("🔄 Analyzing with BanglaBERT..."):
                    try:
                        response = requests.post(
                            f"{api_url}/predict",
                            json={"text": user_input}
                        )
                        
                        if response.status_code == 200:
                            data = response.json()
                            
                            relation = data.get('relation', '')
                            try:
                                relation = relation.encode('latin1').decode('utf-8')
                            except:
                                pass
                            
                            st.markdown("### 📊 Results")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.markdown(f"""
                                <div class="result-card">
                                    <h4>🏷️ Relation</h4>
                                    <div class="relation-badge" style="background:#FF4B2B;">{relation}</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                confidence = data.get('confidence_percentage', 0)
                                color = "green" if confidence > 70 else "orange" if confidence > 40 else "red"
                                st.markdown(f"""
                                <div class="result-card">
                                    <h4>🎯 Confidence</h4>
                                    <h2 style="color:{color};margin:0;">{confidence:.1f}%</h2>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col3:
                                text = data.get('text', '')
                                st.markdown(f"""
                                <div class="result-card">
                                    <h4>📝 Text</h4>
                                    <p style="font-size:0.9rem;margin:0;">{text[:50]}...</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Confidence gauge
                            st.markdown("### 📊 Confidence Analysis")
                            fig = go.Figure()
                            fig.add_trace(go.Indicator(
                                mode = "gauge+number+delta",
                                value = confidence,
                                title = {'text': "Confidence Score"},
                                domain = {'x': [0, 1], 'y': [0, 1]},
                                gauge = {
                                    'axis': {'range': [0, 100]},
                                    'bar': {'color': "#FF4B2B"},
                                    'steps': [
                                        {'range': [0, 40], 'color': "#ffcccc"},
                                        {'range': [40, 70], 'color': "#ffebcc"},
                                        {'range': [70, 100], 'color': "#ccffcc"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75,
                                        'value': confidence
                                    }
                                }
                            ))
                            fig.update_layout(height=250)
                            st.plotly_chart(fig, use_container_width=True)
                            
                        else:
                            st.error(f"❌ API Error: {response.status_code}")
                            
                    except requests.exceptions.ConnectionError:
                        st.error("❌ Could not connect to API. Please start the FastAPI server.")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            else:
                st.warning("⚠️ Please enter some text")
    
    with col2:
        st.markdown("### 💡 Tips")
        st.info("""
        **Best Practices:**
        - Use complete Bengali sentences
        - Include entity names (person, place, organization)
        - Avoid very short texts (min 3 words)
        
        **Example Inputs:**
        - "সত্যজিৎ রায় কলকাতায় জন্মগ্রহণ করেন"
        - "শাকিব খান চালবাজ ছবিতে অভিনয় করেছেন"
        - "হুমায়ূন আহমেদ নন্দিত নরকে লিখেছেন"
        """)
        
        st.markdown("### 🏷️ Available Relations")
        relations = [
            "🏢 প্রতিষ্ঠানের অবস্থান",
            "🕊️ মৃত্যুস্থান",
            "🎂 জন্মস্থান",
            "🎬 চলচ্চিত্র পরিচালক",
            "🎭 চলচ্চিত্র অভিনেতা",
            "🏗️ প্রতিষ্ঠাতা",
            "📝 লেখক"
        ]
        for rel in relations:
            st.markdown(f"- {rel}")

# ============================================
# Page: Bulk Processing
# ============================================

elif page == "📁 Bulk Processing":
    st.markdown("## 📁 Bulk Relation Extraction")
    st.markdown("Upload an Excel file with multiple Bengali texts for batch processing")
    
    uploaded_file = st.file_uploader(
        "📤 Upload Excel File",
        type=["xlsx", "xls", "csv"],
        help="Upload a file with Bengali texts in the first column"
    )
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.markdown("### 📋 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            st.info(f"📊 Total rows: {len(df)}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                text_column = st.selectbox(
                    "Select text column",
                    df.columns.tolist()
                )
            
            with col2:
                if st.button("🚀 Process All", use_container_width=True):
                    texts = df[text_column].astype(str).tolist()
                    results = []
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, text in enumerate(texts):
                        status_text.text(f"Processing {i+1}/{len(texts)}...")
                        
                        try:
                            response = requests.post(
                                f"{api_url}/predict",
                                json={"text": str(text)}
                            )
                            if response.status_code == 200:
                                data = response.json()
                                relation = data.get('relation', '')
                                try:
                                    relation = relation.encode('latin1').decode('utf-8')
                                except:
                                    pass
                                results.append({
                                    'text': text[:100],
                                    'relation': relation,
                                    'confidence': f"{data.get('confidence_percentage', 0):.1f}%"
                                })
                            else:
                                results.append({
                                    'text': text[:100],
                                    'relation': 'Error',
                                    'confidence': '0%'
                                })
                        except:
                            results.append({
                                'text': text[:100],
                                'relation': 'Error',
                                'confidence': '0%'
                            })
                        
                        progress_bar.progress((i + 1) / len(texts))
                    
                    status_text.text("✅ Processing Complete!")
                    
                    result_df = pd.DataFrame(results)
                    
                    st.markdown("### 📊 Results")
                    st.dataframe(result_df, use_container_width=True)
                    
                    csv = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="relation_extraction_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")

# ============================================
# Page: File Upload (PDF/TXT/DOCX)
# ============================================

elif page == "📄 File Upload":
    st.markdown("## 📄 Upload News Article (PDF/TXT/DOCX)")
    st.markdown("Upload a news article file to extract relations automatically")
    
    st.info("""
    **Supported File Types:**
    - 📄 PDF (.pdf)
    - 📝 Text (.txt)
    - 📝 Word (.docx)
    
    **How it works:**
    1. Upload your file
    2. System extracts text automatically
    3. Finds entity pairs (Person, Place, Organization)
    4. Predicts relations using BanglaBERT
    """)
    
    uploaded_file = st.file_uploader(
        "📤 Choose a file",
        type=["pdf", "txt", "docx"],
        help="Upload PDF, TXT, or DOCX file"
    )
    
    if uploaded_file:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📄 File Name", uploaded_file.name[:30] + "...")
        with col2:
            st.metric("📦 File Size", f"{uploaded_file.size / 1024:.1f} KB")
        with col3:
            st.metric("📂 File Type", uploaded_file.name.split('.')[-1].upper())
        
        if st.button("👁️ Preview File Content", use_container_width=True):
            try:
                content = uploaded_file.read().decode('utf-8')
                st.text_area("File Content Preview", content[:1000], height=200)
                uploaded_file.seek(0)
            except:
                st.warning("⚠️ Binary file - preview not available")
        
        if st.button("🔍 Extract Relations from File", use_container_width=True, type="primary"):
            with st.spinner("🔄 Processing file with BanglaBERT..."):
                try:
                    uploaded_file.seek(0)
                    
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    response = requests.post(
                        f"{api_url}/upload/file",
                        files=files
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        if data.get('success'):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("📊 Total Sentences", data.get('total_sentences', 0))
                            with col2:
                                st.metric("🔍 Entities Found", data.get('total_entities', 0))
                            with col3:
                                st.metric("✅ Relations Extracted", len(data.get('predictions', [])))
                            
                            predictions = data.get('predictions', [])
                            if predictions:
                                st.markdown("### 📊 Extracted Relations")
                                df = pd.DataFrame(predictions)
                                st.dataframe(df, use_container_width=True)
                                
                                csv = df.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="📥 Download Results as CSV",
                                    data=csv,
                                    file_name=f"{uploaded_file.name}_relations.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                                
                                st.markdown("### 📈 Confidence Distribution")
                                fig = px.bar(
                                    df,
                                    x='entity1',
                                    y='confidence',
                                    color='relation',
                                    title='Relation Extraction Results'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("No relations found in the file")
                        else:
                            st.error(f"❌ Error: {data.get('error', 'Unknown error')}")
                    else:
                        st.error(f"❌ API Error: {response.status_code}")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    # ============================================
    # Direct Text Input (for testing)
    # ============================================
    
    st.markdown("---")
    st.markdown("### ✍️ Or Paste Text Directly")
    
    test_text = st.text_area(
        "Paste Bengali text here:",
        height=200,
        placeholder="সত্যজিৎ রায় কলকাতায় জন্মগ্রহণ করেন। শাকিব খান চালবাজ ছবিতে অভিনয় করেছেন।"
    )
    
    if st.button("🔍 Extract from Text", use_container_width=True):
        if test_text.strip():
            with st.spinner("🔄 Processing..."):
                try:
                    response = requests.post(
                        f"{api_url}/extract/from-text",
                        json={"text": test_text}
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        if data.get('success'):
                            st.success(f"✅ Found {data.get('total_entities', 0)} entities!")
                            
                            predictions = data.get('predictions', [])
                            if predictions:
                                st.markdown("### 📊 Extracted Relations")
                                df = pd.DataFrame(predictions)
                                st.dataframe(df, use_container_width=True)
                                
                                # Confidence chart
                                st.markdown("### 📈 Confidence Distribution")
                                fig = px.bar(
                                    df,
                                    x='entity1',
                                    y='confidence',
                                    color='relation',
                                    title='Relation Extraction Results'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("No relations found in the text")
                        else:
                            st.error("❌ Processing failed")
                    else:
                        st.error(f"❌ API Error: {response.status_code}")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("⚠️ Please enter some text")

# ============================================
# Page: Analytics
# ============================================

elif page == "📊 Analytics":
    st.markdown("## 📊 Analytics Dashboard")
    st.markdown("Visualize relation extraction statistics")
    
    sample_data = pd.DataFrame({
        'relation': ['Movie Actor', 'Place of Birth', 'Movie Director', 'Writer', 'Place of Death', 'Company Location', 'Company Founder'],
        'count': [38064, 13298, 6164, 2630, 1808, 1011, 281],
        'percentage': [60.2, 21.0, 9.7, 4.2, 2.9, 1.6, 0.4]
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Relation Distribution")
        fig = px.pie(
            sample_data,
            values='count',
            names='relation',
            title='Relation Distribution in Dataset',
            color_discrete_sequence=px.colors.qualitative.Set3,
            hole=0.4
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📈 Relation Frequency")
        fig = px.bar(
            sample_data,
            x='relation',
            y='count',
            title='Number of Samples per Relation',
            color='relation',
            color_discrete_sequence=px.colors.qualitative.Set3,
            text='count'
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 📋 Relation Statistics")
    st.dataframe(sample_data, use_container_width=True)
    
    st.info("💡 Connect to MySQL database for real analytics!")

# ============================================
# Page: Model Performance
# ============================================

elif page == "📈 Model Performance":
    st.markdown("## 📈 Model Performance Metrics")
    st.markdown("Bangla Relation Extraction System - Model Evaluation Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="stat-card">
            <h4>✅ Accuracy</h4>
            <h2 style="color:#FF4B2B;">54.2%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stat-card">
            <h4>📊 Macro F1</h4>
            <h2 style="color:#FF6B4A;">42.5%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stat-card">
            <h4>📈 Weighted F1</h4>
            <h2 style="color:#FF8B6A;">54.0%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="stat-card">
            <h4>🎯 Best Val Acc</h4>
            <h2 style="color:#FFAB8A;">54.2%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 📊 Per-Class Performance")
    
    per_class = pd.DataFrame({
        'Relation': ['Movie Actor', 'Place of Birth', 'Movie Director', 'Writer', 'Place of Death', 'Company Location', 'Company Founder'],
        'Samples': [38064, 13298, 6164, 2630, 1808, 1011, 281],
        'Accuracy': [62, 58, 45, 40, 35, 30, 20]
    })
    
    fig = px.bar(
        per_class,
        x='Relation',
        y='Accuracy',
        title='Per-Class Accuracy',
        color='Accuracy',
        color_continuous_scale='Reds',
        text='Accuracy'
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 📊 Class Distribution vs Accuracy")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=per_class['Relation'],
        y=per_class['Samples'],
        name='Samples',
        marker_color='lightblue'
    ))
    fig.add_trace(go.Scatter(
        x=per_class['Relation'],
        y=per_class['Accuracy'],
        name='Accuracy (%)',
        mode='lines+markers',
        line=dict(color='red', width=3),
        marker=dict(size=10)
    ))
    fig.update_layout(
        title='Class Distribution vs Accuracy',
        xaxis_title='Relation',
        yaxis_title='Samples / Accuracy (%)',
        height=400,
        legend=dict(x=0.8, y=0.9)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 📝 Notes on Performance")
    st.markdown("""
    - **54.2% Accuracy** is a baseline for Bangla-REX dataset
    - **Imbalanced classes** (Company Founder: only 281 samples) affect performance
    - **Future improvements**: Data augmentation, transfer learning, ensemble methods
    - **This is the FIRST model** trained on Bangla-REX dataset
    """)

# ============================================
# Footer
# ============================================

st.markdown("""
<div class="footer">
    <strong>Bangla Relation Extraction System</strong> | Relation Extraction for Bengali Text<br>
    Built with BanglaBERT, FastAPI, and Streamlit
</div>
""", unsafe_allow_html=True)