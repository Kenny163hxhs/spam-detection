# app.py - Improved Email Spam Detection System
# Run with: streamlit run app.py

import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from io import BytesIO

# Download NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Page configuration
st.set_page_config(
    page_title="Email Spam Detection System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Initialize session state variables ---
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame({
        'No.': [1, 2, 3, 4, 5],
        'Message': [
            "Urgent! Please call 09062703810",
            "Hey, are we still meeting for lunch tomorrow?",
            "Congratulations! You've won $1000...",
            "Thanks for the birthday wishes!",
            "Free entry to win iPhone! Text WIN..."
        ],
        'Status': ['Pending']*5
    })

# --- Sidebar ---
st.sidebar.title("📁 Controls")
st.sidebar.markdown("Manage your dataset, settings, and spam keywords here.")

uploaded_file = st.sidebar.file_uploader("📄 Upload CSV (Optional)", type=["csv"])
if uploaded_file:
    df_upload = pd.read_csv(uploaded_file)
    if 'Status' not in df_upload.columns:
        df_upload['Status'] = 'Pending'
    st.session_state.df = df_upload

st.sidebar.subheader("⚡ Custom Spam Keywords")
custom_keywords = st.sidebar.text_area(
    "Add keywords separated by commas",
    placeholder="e.g., lottery, winner, free, click here"
)
custom_keywords_list = [kw.strip().lower() for kw in custom_keywords.split(",") if kw.strip()]

show_dataset = st.sidebar.checkbox("📊 Show Dataset / Inbox", value=True)

# --- Work with session_state.df ---
df = st.session_state.df

# --- Spam Detection ---
def detect_spam(message):
    message_lower = message.lower()
    strong_spam = ['urgent', 'call', 'won', 'winner', 'cash', 'prize',
                   'congratulations', 'click here', 'claim now', 'text to', 'free entry']
    medium_spam = ['free', 'win', 'award', 'nokia', 'mobile', 'entry',
                   'competition', 'gift', 'reward', 'urgent', 'claim']
    for kw in custom_keywords_list:
        if kw not in strong_spam:
            medium_spam.append(kw)
    score = sum(3 for w in strong_spam if w in message_lower)
    score += sum(1 for w in medium_spam if w in message_lower)
    if any(c.isdigit() for c in message) and ('call' in message_lower or 'text' in message_lower):
        score += 2
    is_spam = score >= 2
    confidence = min(60 + (score*8), 99) if is_spam else min(95 + (5 if len(message)<20 else 0), 99)
    return "SPAM" if is_spam else "HAM", confidence, score

# --- Batch Detection ---
if st.sidebar.button("🚀 Run Detection on Dataset"):
    st.session_state.df['Status'] = st.session_state.df['Message'].apply(lambda msg: detect_spam(msg)[0])
    st.success("Detection complete! Dataset updated.")
    df = st.session_state.df
    if show_dataset:
        st.subheader("Inbox / Exported Dataset")
        st.dataframe(df, use_container_width=True)

# --- Dataset Display (default view) ---
elif show_dataset:
    st.subheader("Inbox / Exported Dataset")
    st.dataframe(df, use_container_width=True)

# --- Examples ---
tab1, tab2 = st.tabs(["📌 Spam Example", "📌 Ham Example"])

with tab1:
    st.subheader("Spam Message")
    spam_example = "Urgent! Please call 09062703810"
    st.text_area("Message:", spam_example, height=80, key="spam_input", disabled=True)
    if st.button("🔍 Detect Spam", key="btn_spam"):
        result, confidence, score = detect_spam(spam_example)
        st.progress(int(confidence))
        st.markdown(
            f'<div style="padding:1rem; border-left:5px solid #b71c1c; background:#ffcccc; border-radius:10px;">'
            f'<h3 style="color:#b71c1c;">🚨 DETECTED AS: {result}</h3>'
            f'<p><b>Confidence:</b> {confidence:.1f}%</p>'
            f'<p><b>Spam Indicators Found:</b> {score}</p>'
            f'</div>',
            unsafe_allow_html=True
        )

with tab2:
    st.subheader("Ham Message")
    ham_example = "Thanks for the birthday wishes!"
    st.text_area("Message:", ham_example, height=80, key="ham_input", disabled=True)
    if st.button("✅ Detect Ham", key="btn_ham"):
        result, confidence, score = detect_spam(ham_example)
        st.progress(int(confidence))
        st.markdown(
            f'<div style="padding:1rem; border-left:5px solid #1b5e20; background:#c8e6c9; border-radius:10px;">'
            f'<h3 style="color:#1b5e20;">✅ DETECTED AS: {result}</h3>'
            f'<p><b>Confidence:</b> {confidence:.1f}%</p>'
            f'<p><b>Spam Indicators Found:</b> {score}</p>'
            f'</div>',
            unsafe_allow_html=True
        )

# --- Custom Message ---
st.markdown("---")
st.subheader("🧪 Test Your Own Message")
custom_msg = st.text_area("Enter your message:", height=100, placeholder="Type or paste message...")
if st.button("🔍 Detect Message", type="primary"):
    if custom_msg:
        result, confidence, score = detect_spam(custom_msg)
        color = "#1b5e20" if result=="HAM" else "#b71c1c"
        bg = "#c8e6c9" if result=="HAM" else "#ffcccc"
        st.progress(int(confidence))
        st.markdown(
            f'<div style="padding:1rem; border-left:5px solid {color}; background:{bg}; border-radius:10px;">'
            f'<h3 style="color:{color};">Result: {result}</h3>'
            f'<p><b>Confidence:</b> {confidence:.1f}%</p>'
            f'<p><b>Spam Indicators Found:</b> {score}</p>'
            f'</div>',
            unsafe_allow_html=True
        )
        st.session_state.messages.append({
            'message': custom_msg[:50]+"..." if len(custom_msg)>50 else custom_msg,
            'result': result,
            'confidence': f"{confidence:.1f}%"
        })
        new_row = {'No.': len(st.session_state.df)+1, 'Message': custom_msg, 'Status': result}
        st.session_state.df = pd.concat([st.session_state.df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        st.warning("Please enter a message!")

# --- History ---
if st.session_state.messages:
    st.markdown("---")
    st.subheader("📋 Detection History")
    df_history = pd.DataFrame(st.session_state.messages)
    st.dataframe(df_history, use_container_width=True)
    buffer = BytesIO()
    df_history.to_csv(buffer, index=False)
    st.download_button("💾 Download History CSV", data=buffer.getvalue(), file_name="detection_history.csv")

st.caption("Improved Email Spam Detection System | Powered by Streamlit & NLTK")