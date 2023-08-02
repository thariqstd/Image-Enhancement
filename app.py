import os
import numpy as np
import cv2
from PIL import Image
from flask import Flask, flash, request, render_template, send_from_directory, send_file
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import uuid

# Inisialisasi aplikasi Flask
app = Flask(__name__, static_url_path='')
app.secret_key = os.urandom(24)

# Konfigurasi folder untuk penyimpanan gambar hasil proses
app.config['ENHANCEMENT_FOLDER'] = 'images'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['HISTOGRAM_FOLDER'] = 'histograms'

# Route handler untuk mengirimkan gambar dari folder 'uploads' ke frontend
@app.route('/uploads/<filename>')
def upload_img(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Route handler untuk mengirimkan gambar hasil proses dari folder 'images' ke frontend
@app.route('/images/<filename>')
def enhancement_img(filename):
    return send_from_directory(app.config['ENHANCEMENT_FOLDER'], filename)

# Fungsi untuk melakukan median filtering pada data menggunakan ukuran filter tertentu.
def median_filter(data, filter_size):
    # Inisialisasi list sementara untuk menyimpan nilai-nilai dalam filter
    temp = []
    # Hitung indeks tengah filter
    indexer = filter_size // 2
    # Inisialisasi array hasil dengan ukuran yang sama seperti data dan diisi dengan nol
    data_final = np.zeros_like(data)

    # Looping untuk setiap baris pada data
    for i in range(len(data)):
        # Looping untuk setiap kolom pada data
        for j in range(len(data[0])):
            # Looping untuk setiap elemen dalam filter
            for z in range(filter_size):
                # Jika indeks di luar batas atas atau bawah data, tambahkan nol ke dalam temp
                if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
                    for c in range(filter_size):
                        temp.append(0)
                else:
                    # Jika indeks di luar batas kiri atau kanan data, tambahkan nol ke dalam temp
                    if j + z - indexer < 0 or j + indexer > len(data[0]) - 1:
                        temp.append(0)
                    else:
                        # Tambahkan nilai data ke dalam temp sesuai dengan posisi filter
                        for k in range(filter_size):
                            temp.append(data[i + z - indexer][j + k - indexer])

            # Urutkan nilai dalam temp dari yang terkecil hingga terbesar
            temp.sort()
            # Pilih nilai median dari temp dan simpan pada posisi yang sesuai dalam data_final
            data_final[i][j] = temp[len(temp) // 2]
            # Bersihkan temp untuk digunakan kembali pada iterasi selanjutnya
            temp = []
    return data_final

# Fungsi untuk melakukan median filtering pada citra berwarna (RGB) menggunakan ukuran filter tertentu.
def median_filter_rgb(img, filter_size):
    channels = cv2.split(img) # Memisahkan kanal warna (R, G, B) dari citra
    filtered_channels = []
    for channel in channels:
        filtered_channel = median_filter(channel, filter_size) # Memanggil fungsi median_filter untuk setiap kanal warna
        filtered_channels.append(filtered_channel)
    return cv2.merge(filtered_channels) # Menggabungkan kembali kanal warna yang telah di-filter menjadi citra hasil

#Fungsi untuk melakukan histogram equalization pada citra menggunakan model warna YUV.
def histogram_equalization(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img_output

#Fungsi untuk menampilkan histogram citra dan menyimpannya dalam format gambar PNG.
def show_image_histogram(image_path, save_path):
    # Membaca citra dari path yang diberikan
    image = cv2.imread(image_path)

    # Memeriksa apakah citra berhasil dibaca atau tidak
    if image is None:
        print("Gambar tidak dapat dibaca. Pastikan path gambar benar dan gambar ada.")
        return
    
    # Membuat objek figure untuk menampilkan plot histogram dengan ukuran 7x5 inch
    plt.figure(figsize=(7, 5))

    # Memeriksa apakah citra merupakan citra warna (BGR) atau grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Jika citra warna, hitung histogram untuk masing-masing kanal warna (RGB)
        hist_channels = []
        for i in range(image.shape[2]):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            hist_channels.append(hist)

        # Plot histogram untuk masing-masing kanal warna
        colors = ['red', 'green', 'blue']
        for i, color in enumerate(colors):
            plt.plot(hist_channels[i], color=color)
        plt.title('Histogram')
        plt.xlabel('Nilai Pixel')
        plt.ylabel('Frekuensi')

    else:
        # Jika citra grayscale, hitung histogram untuk citra tersebut
        hist_gray = cv2.calcHist([image], [0], None, [256], [0, 256])

        # Plot histogram untuk citra grayscale
        plt.plot(hist_gray, color='black')
        plt.title('Histogram')
        plt.xlabel('Nilai Pixel')
        plt.ylabel('Frekuensi')

    # Menyesuaikan tata letak plot histogram agar sesuai
    plt.tight_layout()

    # Simpan histogram sebagai file gambar dalam format PNG ke lokasi yang diberikan
    plt.savefig(save_path)

    # Buat objek BytesIO untuk menyimpan gambar histogram dalam format BytesIO
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Convert objek BytesIO ke base64-encoded string untuk kemudian ditampilkan dalam halaman web atau aplikasi
    histogram_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()

    return histogram_base64

# Route handler untuk mengembalikan hasil render dari template index.html
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Memeriksa apakah permintaan yang diterima adalah POST
    if request.method == 'POST':
        # Mengambil file gambar dari permintaan POST
        f = request.files.get('file')

        # Mengambil pilihan algoritma dari form
        algorithm = request.form.get('Algorithm')

         # Memeriksa apakah ada file gambar yang diunggah
        if not f:
            flash('Please select an image')
            return render_template('index.html')
        
        # Mengambil path dasar direktori
        basepath = os.path.dirname(__file__)
        # Memisahkan nama dan ekstensi file
        filename, file_extension = os.path.splitext(secure_filename(f.filename))
        # Membuat nama unik dengan menggunakan UUID untuk menghindari nama file yang sama
        unique_filename = f"{filename}_{str(uuid.uuid4())}{file_extension}"
        # Menggabungkan path untuk menyimpan file gambar yang diunggah
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        # Menyimpan file gambar yang diunggah ke path yang ditentukan
        f.save(file_path)
        # Mendapatkan nama file saja tanpa path
        file_name = os.path.basename(file_path)

        # Membaca gambar yang diunggah menggunakan OpenCV
        img = cv2.imread(file_path)

        # Menghasilkan histogram sebelum proses gambar dan menyimpannya dalam bentuk file gambar PNG
        histogram_path_original = os.path.join(app.config['HISTOGRAM_FOLDER'], f'{file_name}_before.png')
        histogram_base64_original = show_image_histogram(file_path, histogram_path_original)

        # Memeriksa algoritma yang dipilih dan memproses gambar sesuai dengan algoritma yang dipilih
        # Apbalia algoritma yang dipilih adalah median filter
        if algorithm == "Med_Filter":
            # Menghasilkan nama file untuk hasil enhancement dengan sufiks "medfil"
            enhance_fname = f"{filename}_medfil{file_extension}"
            # Memproses gambar dengan algoritma median_filter_rgb dengan ukuran filter 3x3
            enhancement = median_filter_rgb(img, filter_size=3)
            # Menggabungkan path direktori, nama file enhancement, dan membuat path lengkap
            enhancement_path = os.path.join(basepath, 'images', secure_filename(enhance_fname))
            # Mendapatkan nama file dari path enhancement
            fname = os.path.basename(enhancement_path)
            # Menyimpan hasil enhancement ke path yang telah ditentukan
            cv2.imwrite(enhancement_path, enhancement)

            # Menggabungkan path direktori basepath, folder 'images', dan nama file enhancement
            enhancement_path = os.path.join(basepath, 'images', secure_filename(enhance_fname))
            # Menggabungkan path direktori app.config['HISTOGRAM_FOLDER'] dan nama file histogram hasil enhancement
            histogram_path_enhanced = os.path.join(app.config['HISTOGRAM_FOLDER'], f'{enhance_fname}_after.png')
            # Memperoleh gambar histogram setelah proses enhancement dan menyimpannya di path yang telah ditentukan
            histogram_base64_enhanced = show_image_histogram(enhancement_path, histogram_path_enhanced)

            # Render halaman predict.html dengan menyediakan informasi hasil prediksi
            return render_template('predict.html', file_name=file_name,
                                   enhancement_file=fname,
                                   before_histogram_base64=histogram_base64_original,
                                   after_histogram_base64=histogram_base64_enhanced)
        
        #Apabila algoritma yang dipilih adalah histogram equalization
        elif algorithm == "Hist_Equalization":
            # Membuat nama file enhancement dengan sufiks "_hiseq" (Histogram Equalization)
            enhance_fname = f"{filename}_hiseq{file_extension}"
            # Melakukan histogram equalization pada gambar menggunakan fungsi histogram_equalization
            enhancement = histogram_equalization(img)
            # Menggabungkan path direktori basepath, folder 'images', dan nama file enhancement
            enhancement_path = os.path.join(basepath, 'images', secure_filename(enhance_fname))
            # Mengambil nama file dari enhancement_path
            fname = os.path.basename(enhancement_path)
            # Menyimpan gambar enhancement ke path yang telah ditentukan
            cv2.imwrite(enhancement_path, enhancement)

            # Menggabungkan path direktori basepath, folder 'images', dan nama file enhancement (setelah enhancement)
            enhancement_path = os.path.join(basepath, 'images', secure_filename(enhance_fname))
            # Menggabungkan path direktori app.config['HISTOGRAM_FOLDER'] dan nama file histogram hasil enhancement
            histogram_path_enhanced = os.path.join(app.config['HISTOGRAM_FOLDER'], f'{enhance_fname}_after.png')
            # Memperoleh gambar histogram setelah proses enhancement dan menyimpannya di path yang telah ditentukan
            histogram_base64_enhanced = show_image_histogram(enhancement_path, histogram_path_enhanced)

            # Render halaman predict.html dengan menyediakan informasi hasil prediksi
            return render_template('predict.html', file_name=file_name,
                                   enhancement_file=fname,
                                   before_histogram_base64=histogram_base64_original,
                                   after_histogram_base64=histogram_base64_enhanced)
        
        # Jika algoritma yang dipilih tidak sesuai dengan yang ada dalam pilihan
        # Munculkan pesan flash untuk meminta pengguna memilih algoritma yang sesuai
        else:
            flash('Please select algorithm')
            return render_template('index.html')

    return ""


if __name__ == '__main__':
    app.run(debug=True, host="localhost", port=8080)
