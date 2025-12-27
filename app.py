%%writefile app.py
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# إعداد الصفحة
st.set_page_config(page_title="كاشف الأخبار الزائفة", page_icon="🕵️‍♂️", layout="centered")

# تحميل الموديل والتوكينايزر
@st.cache_resource
def load_prediction_resources():
    try:
        model = load_model('fake_news_model.h5')
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        return model, tokenizer
    except:
        return None, None

model, tokenizer = load_prediction_resources()

# الواجهة
st.title("🕵️‍♂️ Fake News Detection System")

if model is None or tokenizer is None:
    st.error("⚠️ الملفات (fake_news_model.h5) أو (tokenizer.pickle) غير موجودة في الملفات على اليسار!")
    st.stop()

text_input = st.text_area("أدخل نص الخبر باللغة الإنجليزية:", height=200)

if st.button("تحقق من الخبر 🔍"):
    if not text_input.strip():
        st.warning("اكتب نص عشان أقدر أحلله!")
    else:
        # التجهيز والتوقع
        sequences = tokenizer.texts_to_sequences([text_input])
        padded = pad_sequences(sequences, maxlen=300, padding='post', truncating='post')
        prediction = model.predict(padded)[0][0]
        
        st.markdown("---")
        if prediction > 0.5:
            st.success(f"✅ **خبر حقيقي (Real News)** - ثقة: {prediction*100:.1f}%")
        else:
            st.error(f"🚨 **خبر زائف (Fake News)** - شك: {(1-prediction)*100:.1f}%")
