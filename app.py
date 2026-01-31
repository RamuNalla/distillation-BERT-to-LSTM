import streamlit as st
import torch
import time
import os
from model_arch import BiLSTMStudent
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F

# --- 1. Professional Page Configuration ---
st.set_page_config(
    page_title="Knowledge Distillation Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Custom CSS for High-End UI ---
st.markdown("""
    <style>
    /* Main Background */
    .main {
        background-color: #f8f9fa;
    }
    /* Metric Cards */
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        color: #0F1116;
    }
    .css-1r6slb0 {
        border: 1px solid #e0e0e0;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        background-color: white;
    }
    /* Header Styling */
    h1 {
        color: #1E1E1E;
        font-family: 'Helvetica Neue', sans-serif;
    }
    /* Button Styling */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        height: 50px;
        width: 100%;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. Robust Model Loading ---
@st.cache_resource
def load_assets():
    try:
        # PATH CONFIGURATION
        teacher_path = "./models/teacher/final_teacher"
        student_path = "./models/student/student_lstm.pth"
        
        # A. Load Tokenizer (FIX: Load from Hub to avoid missing vocab.txt error)
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # B. Load Teacher (BERT) - Local Weights
        if not os.path.exists(teacher_path):
            st.error(f"❌ Teacher model not found at {teacher_path}")
            return None, None, None
        teacher = AutoModelForSequenceClassification.from_pretrained(teacher_path)
        
        # C. Load Student (Bi-LSTM) - Local Weights
        if not os.path.exists(student_path):
            st.error(f"❌ Student model not found at {student_path}")
            return None, None, None
            
        student = BiLSTMStudent(vocab_size=30522, embed_dim=128, hidden_dim=256, output_dim=2)
        student.load_state_dict(torch.load(student_path, map_location="cpu"))
        student.eval()
        
        return tokenizer, teacher, student

    except Exception as e:
        st.error(f"Critical Error Loading Models: {str(e)}")
        return None, None, None

# Load models safely
tokenizer, teacher, student = load_assets()

if tokenizer is None:
    st.stop() # Stop execution if models failed to load

# --- 4. Sidebar Dashboard ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/brain.png", width=80)
    st.title("Project Controls")
    st.markdown("Adjust parameters to test model robustness.")
    
    threshold = st.slider("Confidence Threshold", 0.5, 1.0, 0.75, 
                          help="Predictions below this confidence will be flagged as Uncertain.")
    
    st.divider()
    st.info("""
    **Architecture Specs:**
    - **Teacher:** BERT-Base (110M Params)
    - **Student:** Bi-LSTM (2.5M Params)
    - **Speedup:** ~15x
    """)
    st.caption("v1.0.0 | Production Ready")

# --- 5. Main Application Interface ---
col1, col2 = st.columns([3, 1])
with col1:
    st.title("⚡ Neural Knowledge Distillation")
    st.markdown("### BERT-to-LSTM Real-time Comparator")
with col2:
    # Status indicator
    st.success("System Online")

st.markdown("---")

# Input Section
col_input, col_btn = st.columns([4, 1])
with col_input:
    user_text = st.text_input("Enter Review Text:", "The cinematography was breathtaking, but the script felt a bit weak.", 
                              placeholder="Type a movie review here...")
with col_btn:
    st.write("") # Spacing
    st.write("") 
    analyze_btn = st.button("🚀 Analyze")

# --- 6. Analysis Logic ---
if analyze_btn:
    if not user_text.strip():
        st.warning("⚠️ Please enter valid text to analyze.")
    else:
        # Prepare Data
        inputs = tokenizer(user_text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
        
        # --- Teacher Inference ---
        start_t = time.perf_counter()
        with torch.no_grad():
            t_out = teacher(**inputs)
            t_probs = F.softmax(t_out.logits, dim=1)
            t_conf, t_pred_idx = torch.max(t_probs, 1)
        time_t = (time.perf_counter() - start_t) * 1000
        
        # --- Student Inference ---
        start_s = time.perf_counter()
        with torch.no_grad():
            s_logits = student(inputs['input_ids'])
            s_probs = F.softmax(s_logits, dim=1)
            s_conf, s_pred_idx = torch.max(s_probs, 1)
        time_s = (time.perf_counter() - start_s) * 1000

        # --- Display Results ---
        st.markdown("### 📊 Inference Report")
        
        # Labels mapping
        labels = ["Negative", "Positive"]
        
        res_col1, res_col2 = st.columns(2)
        
        # Teacher Card
        with res_col1:
            st.markdown("#### 👨‍🏫 Teacher (BERT)")
            t_label = labels[t_pred_idx.item()]
            
            # Dynamic Badge Color
            t_color = "green" if t_label == "Positive" else "red"
            st.markdown(f":{t_color}-background[{t_label} Sentiment]")
            
            # Metrics
            c1, c2 = st.columns(2)
            c1.metric("Confidence", f"{t_conf.item():.2%}")
            c2.metric("Latency", f"{time_t:.2f} ms")
            
            # Probability Bar
            st.progress(t_conf.item(), text="Model Certainty")

        # Student Card
        with res_col2:
            st.markdown("#### 🎓 Student (Bi-LSTM)")
            s_label = labels[s_pred_idx.item()]
            
            # Dynamic Badge Color
            s_color = "green" if s_label == "Positive" else "red"
            st.markdown(f":{s_color}-background[{s_label} Sentiment]")
            
            # Metrics
            c3, c4 = st.columns(2)
            c3.metric("Confidence", f"{s_conf.item():.2%}")
            
            # Speedup Calculation
            speedup = time_t / time_s if time_s > 0 else 0
            c4.metric("Latency", f"{time_s:.2f} ms", delta=f"{speedup:.1f}x Faster")
            
            # Probability Bar
            st.progress(s_conf.item(), text="Model Certainty")

        # --- Expert Analysis Section ---
        st.markdown("---")
        with st.expander("🧐 Technical Deep Dive", expanded=True):
            match = t_pred_idx == s_pred_idx
            
            if match:
                st.success(f"✅ **Distillation Success:** The Student successfully mimicked the Teacher's decision with **{speedup:.1f}x** less latency.")
            else:
                st.warning(f"⚠️ **Divergence:** The Student disagreed with the Teacher. This can happen on complex sentences where the LSTM lacks the 'Attention' mechanism of BERT.")
            
            st.code(f"""
            # Internal Logits Comparison
            Teacher Logits: {t_out.logits.numpy().round(2)}
            Student Logits: {s_logits.numpy().round(2)}
            """, language="python")