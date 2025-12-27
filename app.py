import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# 1. إعداد الصفحة وتصميمها
st.set_page_config(page_title="كاشف الأخبار الزائفة", page_icon="🕵️‍♂️", layout="centered")

# CSS عشان نخلي الشكل عربي ومظبوط
st.markdown("""
<style>
    .stTextArea textarea {
        font-size: 16px !important;
        direction: ltr; /* النص الإنجليزي يفضل من الشمال */
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# 2. دالة تحميل الموديل والتوكينايزر (Cashed عشان السرعة)
@st.cache_resource
def load_prediction_resources():
    try:
        # تحميل الموديل
        model = load_model('fake_news_model.h5')
        
        # تحميل التوكينايزر
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
            
        return model, tokenizer
    except Exception as e:
        return None, None

# تحميل الموارد
model, tokenizer = load_prediction_resources()

# 3. واجهة التطبيق
st.title("🕵️‍♂️ Fake News Detection System")
st.caption("Developed by AI Team - Minufiya University")
st.markdown("---")

# التأكد من تحميل الملفات بنجاح
if model is None or tokenizer is None:
    st.error("⚠️ ملفات النظام ناقصة! تأكد من وجود 'fake_news_model.h5' و 'tokenizer.pickle' بجوار ملف الكود.")
    st.stop()

st.header("أدخل نص الخبر باللغة الإنجليزية:")
text_input = st.text_area("", height=200, placeholder="Paste the article content here...")

# 4. زر التحقق والمنطق
if st.button("تحقق من الخبر 🔍"):
    if not text_input.strip():
        st.warning("الرجاء إدخال نص للتحقق منه.")
    else:
        with st.spinner('جاري تحليل الخبر...'):
            # تجهيز النص (نفس خطوات التدريب)
            sequences = tokenizer.texts_to_sequences([text_input])
            padded = pad_sequences(sequences, maxlen=300, padding='post', truncating='post')
            
            # التوقع
            prediction = model.predict(padded)[0][0]
            
            # عرض النتيجة
            st.markdown("---")
            # بما إننا دربنا إن 1 = حقيقي و 0 = مزيف
            if prediction > 0.5:
                confidence = prediction * 100
                st.success(f"✅ **خبر حقيقي (Real News)**")
                st.progress(int(confidence))
                st.write(f"نسبة الثقة: {confidence:.2f}%")
            else:
                confidence = (1 - prediction) * 100
                st.error(f"🚨 **خبر زائف (Fake News)**")
                st.progress(int(confidence))
                st.write(f"نسبة الشك: {confidence:.2f}%")