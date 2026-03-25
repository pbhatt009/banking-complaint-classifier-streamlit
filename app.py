import streamlit as st
import torch
import torch.nn.functional as F
import joblib
import pandas as pd
import plotly.express as px
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import dotenv
dotenv.load_dotenv()
hf_api_key = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

# --- Page Config ---
st.set_page_config(
    page_title="Banking Complaint Classifier",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = "Priyanshub009/banking_complaint_classfier_new"
LE_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")

# --- Loaders ---
@st.cache_resource(show_spinner="Loading NLP Model & Tokenizer. This might take a few moments...")
def load_resources():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR,token=hf_api_key)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, token=hf_api_key )
    le = joblib.load(LE_PATH)
    return tokenizer, model, le

# --- Main UI Header ---
st.title("🏦 Banking Complaint Classifier")
st.markdown("Classify customer banking complaints into their appropriate handling categories using a locally compiled, fine-tuned Hugging Face transformer pipeline.")

# File structure validation check
if not os.path.exists(LE_PATH):
    st.error(f"⚠️ Could not find label encoder file. Please verify this path exists:\n- `{LE_PATH}`")
    st.stop()

try:
    tokenizer, model, le = load_resources()
except Exception as e:
    st.error(f"An error occurred loading the pipeline models: {e}")
    st.stop()

classes = le.classes_
num_classes = len(classes)

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Model Info")
st.sidebar.info("Hugging Face Sequence Classification Model powered backend.")
st.sidebar.metric("Categories (Classes)", num_classes)

st.sidebar.markdown("---")
st.sidebar.header("📊 Visualization Settings")
show_top_k = st.sidebar.slider("Show Top-K Classes in Chart", min_value=1, max_value=num_classes, value=min(5, num_classes))
show_all_probs = st.sidebar.checkbox("Show Detailed Probabilities Table", value=True)


# --- Example Injection & Main Form ---
st.markdown("### ✍️ Input Complaint text")

EXAMPLES = [
    "Select an example...",
    "I was charged twice on my credit card.",
    "My bank account transfer is pending for 5 days.",
    "Debt collectors keep calling me for a loan I never took.",
    "My loan application was rejected without explanation.",
    "Money transfer failed but amount was deducted."
]

selected_example = st.selectbox("Choose from example complaints:", EXAMPLES)

default_text = ""
if selected_example != "Select an example...":
    default_text = selected_example

user_input = st.text_area("Or type your own raw complaint here:", value=default_text, height=150, placeholder="E.g., I'm writing to dispute an unauthorized transaction reflecting in my deposit account...")

if st.button("🚀 Classify Complaint", type="primary", use_container_width=True):
    if not user_input.strip():
        st.warning("Please enter a text complaint to classify.")
    else:
        with st.spinner("Analyzing complaint intent..."):
            # Tokenize & Execute Inference
            inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Extract Logits and transform to Probability Vectors
            logits = outputs.logits
            probs = F.softmax(logits, dim=1).squeeze().numpy()
            
            # Map predictions to Labels
            prob_dict = {classes[i]: float(probs[i]) for i in range(num_classes)}
            sorted_probs = dict(sorted(prob_dict.items(), key=lambda item: item[1], reverse=True))
            
            best_class = list(sorted_probs.keys())[0]
            best_prob = sorted_probs[best_class]
        
        # --- Notification & Analytics Visuals ---
        st.markdown("---")
        st.markdown("### 🎯 Classification Results")
        
        # Highlight metrics
        c1, c2 = st.columns(2)
        c1.metric(label="Predicted Category", value=best_class)
        c2.metric(label="Confidence Score", value=f"{best_prob * 100:.2f}%")

        st.markdown("### 📊 Probability Distribution")
        
        # Prepare Dataframe for visualization
        df_probs = pd.DataFrame(list(sorted_probs.items()), columns=["Category", "Probability"])
        df_probs["Percentage (%)"] = df_probs["Probability"] * 100
        
        # Filter for top K config and reverse rank to draw appropriately Horizontal Chart from Largest->Smallest Top-to-Bottom
        top_k_df = df_probs.head(show_top_k).copy()
        top_k_df = top_k_df.iloc[::-1]  
        
        # Construct Plotly Bar Chart
        fig = px.bar(
            top_k_df,
            x="Percentage (%)",
            y="Category",
            orientation="h",
            text=top_k_df["Percentage (%)"].apply(lambda x: f"{x:.1f}%"),
            color="Percentage (%)",
            color_continuous_scale="Blues_r" # Strong contrast color setup
        )
        
        fig.update_layout(
            xaxis_title="Confidence Percentage (%)",
            yaxis_title="Category",
            showlegend=False,
            height=max(250, 45 * show_top_k), # Keep bars looking natural
            margin=dict(l=0, r=0, t=10, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

        if show_all_probs:
            with st.expander("Show All Detailed Probabilities Matrix", expanded=True):
                df_display = df_probs[["Category", "Percentage (%)"]].copy()
                df_display["Percentage (%)"] = df_display["Percentage (%)"].apply(lambda x: f"{x:.3f}%")
                st.dataframe(df_display, use_container_width=True)
