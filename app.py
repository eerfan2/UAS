from flask import Flask, render_template, request
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import gdown

app = Flask(__name__)

# URL Google Drive (Ganti FILE_ID dengan ID file model di Google Drive)
GDRIVE_URL = "https://drive.google.com/uc?id=FILE_ID"

# Path penyimpanan model
MODEL_PATH = os.path.join(os.getcwd(), "model_tomat_cnn.h5")

# Jika model belum ada, unduh dari Google Drive
if not os.path.exists(MODEL_PATH):
    print("Model not found, downloading from Google Drive...")
    gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)

# Pastikan model tersedia
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Failed to download model from {GDRIVE_URL}")

# Load model setelah dipastikan ada
model = load_model(MODEL_PATH)

# Label kelas sesuai model
dic = {0: "tomat_segar", 1: "tomat_hama"}

# Load model
MODEL_PATH = os.path.join(os.getcwd(), "model_tomat_cnn.h5")  # Pastikan path benar
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = load_model(MODEL_PATH)

# Fungsi prediksi dengan error handling
def predict_label(img_path):
    try:
        i = image.load_img(img_path, target_size=(224, 224))  # Ubah ukuran gambar sesuai model
        i = image.img_to_array(i) / 255.0  # Normalisasi gambar
        i = np.expand_dims(i, axis=0)  # Sesuaikan bentuk untuk model

        p = np.argmax(model.predict(i), axis=1)  # Prediksi kelas gambar
        return dic[p[0]]  # Ambil label berdasarkan indeks prediksi
    except Exception as e:
        print(f"Error in prediction: {e}")
        return "Error in prediction"

# Route utama
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("classification.html")

# Route untuk upload dan klasifikasi gambar
@app.route("/submit", methods=['POST'])
def get_output():
    if 'my_image' not in request.files:
        return "No file uploaded", 400

    img = request.files['my_image']
    if img.filename == '':
        return "No selected file", 400

    # Buat folder static jika belum ada
    if not os.path.exists("static"):
        os.makedirs("static")

    # Simpan gambar dengan path yang benar
    img_path = os.path.join("static", img.filename)
    img.save(img_path)

    # Debug: memastikan gambar tersimpan
    print(f"Image saved at: {img_path}")

    # Prediksi
    p = predict_label(img_path)

    return render_template("classification.html", prediction=p, img_path=img_path)

# Jalankan Flask hanya jika file ini dieksekusi langsung
if __name__ == '__main__':
    app.run(debug=True)
