import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os

# إعداد الصفحة
st.set_page_config(page_title="كاشف الأخبار الزائفة", page_icon="🕵️‍♂️")

# العنوان
st.title("🕵️‍♂️ Fake News Detection System")
st.write("نظام ذكاء اصطناعي لكشف الأخبار المضللة - مشروع مادة الـ AI")

# تحميل الموديل والملفات
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('fake_news_model.h5')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

# التأكد من وجود الملفات قبل التحميل
if not os.path.exists('fake_news_model.h5') or not os.path.exists('tokenizer.pickle'):
    st.error("⚠️ ملفات الموديل غير موجودة! تأكد أنك وضعت fake_news_model.h5 و tokenizer.pickle بجانب ملف app.py")
else:
    try:
        model, tokenizer = load_model()
        
        # واجهة المستخدم
        st.subheader("أدخل نص الخبر باللغة الإنجليزية:")
        text_input = st.text_area("News Text", height=200, placeholder="Paste the article content here...")

        if st.button("تحقق من الخبر 🔍"):
            if text_input.strip():
                # تجهيز النص للموديل
                sequences = tokenizer.texts_to_sequences([text_input])
                padded = pad_sequences(sequences, maxlen=300, padding='post', truncating='post')
                
                # التوقع
                prediction = model.predict(padded)[0][0]
                
                # عرض النتيجة
                st.markdown("---")
                if prediction > 0.5:
                    st.success(f"✅ **خبر حقيقي (Real News)**\n\nنسبة الثقة: {prediction*100:.1f}%")
                else:
                    st.error(f"🚨 **خبر زائف (Fake News)**\n\nنسبة الشك: {(1-prediction)*100:.1f}%")
            else:
                st.warning("من فضلك اكتب نص الخبر أولاً.")
                
    except Exception as e:
        st.error(f"حدث خطأ أثناء تحميل الموديل: {e}")

# تذييل الصفحة
st.markdown("---")
st.caption("Developed by AI Team - Minufiya University")