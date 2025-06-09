import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import joblib
import matplotlib.pyplot as plt
from collections import Counter
from processing import crop_banana_yolo, ekstrak_fitur_lengkap, yolo_model

# Load model dan tools
model_rf = joblib.load("model PMD/model_rf.pkl")
model_svm = joblib.load("model PMD/model_svm.pkl")
scaler = joblib.load("model PMD/scaler.pkl")
encoder = joblib.load("model PMD/label_encoder.pkl")

# Konfigurasi halaman
st.set_page_config(page_title="🍌 Banana Ripeness Classifier", layout="wide")
st.markdown("""
    <style>
        .title {
            font-size:48px !important;
            font-weight:700;
            text-align:center;
            margin-bottom:10px;
        }
        .subtitle {
            font-size:20px !important;
            text-align:center;
            color:gray;
        }
        .block-container {
            padding-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# Gambar header di atas panel
st.sidebar.image("gambar header.jpg", use_column_width=True)

# Judul utama
st.markdown('<div class="title">🍌 Klasifikasi Tingkat Kematangan Pisang</div>', unsafe_allow_html=True)

# Sidebar pengaturan
st.sidebar.header("⚙️ Pengaturan")
model_option = st.sidebar.radio("Pilih Model Klasifikasi", ["Random Forest", "SVM"])

uploaded_file = st.file_uploader("📤 Upload Gambar Pisang", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang Diupload", use_column_width=True)

    with st.spinner("🔍 Mendeteksi pisang dan memproses gambar..."):
        hasil_deteksi = yolo_model.predict(source=image, conf=0.4, verbose=False)
        crops = crop_banana_yolo(image)

    st.success(f"✅ {len(crops)} buah pisang terdeteksi dalam gambar.")

    if crops:
        st.subheader("📊 Hasil Klasifikasi Tiap Pisang")
        col_kotak = st.columns(min(4, len(crops)))
        pred_labels = []

        for i, crop in enumerate(crops):
            fitur = ekstrak_fitur_lengkap(crop).reshape(1, -1)
            fitur_scaled = scaler.transform(fitur)

            if model_option == "Random Forest":
                pred = model_rf.predict(fitur_scaled)
            else:
                pred = model_svm.predict(fitur_scaled)

            label = encoder.inverse_transform(pred)[0]
            pred_labels.append(label)

            with col_kotak[i % len(col_kotak)]:
                st.image(crop, caption=f"Pisang #{i+1}\n**Prediksi: {label}**", use_column_width=True)

        # Statistik prediksi
        st.subheader("📈 Statistik Hasil Klasifikasi")
        counter = Counter(pred_labels)
        stat_labels, stat_counts = zip(*counter.items())
        fig1, ax1 = plt.subplots()
        ax1.bar(stat_labels, stat_counts, color='gold')
        ax1.set_ylabel("Jumlah Pisang")
        ax1.set_title("Distribusi Kelas Kematangan")
        st.pyplot(fig1)

        # Gambar dengan bounding box asli
        st.subheader("🖼️ Visualisasi Bounding Box")
        draw_image = image.copy()
        draw = ImageDraw.Draw(draw_image)
        for hasil in hasil_deteksi:
            for box in hasil.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        st.image(draw_image, caption="Bounding Box Deteksi Pisang", use_column_width=True)
    else:
        st.warning("⚠️ Tidak ada pisang yang terdeteksi. Pastikan gambar cukup jelas dan pencahayaan baik.")

st.markdown("---")
st.markdown("""
<div style='text-align: center;'>
    Dibuat dengan ❤️ oleh <b>Kelompok 1</b> - Klasifikasi Kematangan Pisang
</div>
""", unsafe_allow_html=True)
