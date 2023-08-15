# Mengimpor modul os untuk interaksi dengan sistem operasi
import os

# Mengimpor modul numpy dengan alias np untuk pengolahan data numerik
import numpy as np

# Mengimpor modul cv2 (OpenCV) untuk operasi pengolahan gambar dan video
import cv2

# Mengimpor class Image dari modul PIL untuk manipulasi gambar
from PIL import Image

# Mengimpor kelas Flask dari modul flask untuk membuat aplikasi web
# Mengimpor fungsi flash, request, render_template, dan send_from_directory dari modul flask
from flask import Flask, flash, request, render_template, send_from_directory

# Mengimpor fungsi secure_filename dari modul werkzeug.utils
# Ini akan membantu dalam mengamankan nama file yang diunggah ke server
from werkzeug.utils import secure_filename

# Mengimpor modul matplotlib untuk membuat visualisasi, grafik, dan plot
import matplotlib

# Menggunakan backend 'Agg' agar matplotlib dapat bekerja dalam lingkungan tanpa GUI
matplotlib.use('Agg')

# Mengimpor modul pyplot dari matplotlib dengan alias plt untuk membuat plot
import matplotlib.pyplot as plt

# Mengimpor class BytesIO dari modul io untuk bekerja dengan data dalam bentuk byte
from io import BytesIO

# Mengimpor modul base64 untuk enkoding dan dekoding data dalam format base64
import base64


# Membuat instance aplikasi Flask dengan nama 'app'
# Parameter __name__ menunjukkan nama modul saat ini
# static_url_path='' mengatur alamat URL untuk konten statis menjadi akar (root)
app = Flask(__name__, static_url_path='')

# Menetapkan kunci rahasia (secret key) untuk aplikasi Flask
# Kunci ini digunakan untuk menjaga keamanan sesi dan data terenkripsi
# Menggunakan os.urandom(24) untuk menghasilkan urutan acak 24 byte sebagai kunci
app.secret_key = os.urandom(24)

# Mengkonfigurasi folder tempat penyimpanan gambar hasil proses
# ENHANCEMENT_FOLDER adalah folder tempat hasil pemrosesan gambar disimpan
# UPLOAD_FOLDER adalah folder tempat gambar yang diunggah akan disimpan
# HISTOGRAM_FOLDER adalah folder tempat gambar histogram disimpan
app.config['ENHANCEMENT_FOLDER'] = 'images'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['HISTOGRAM_FOLDER'] = 'histograms'

# Decorator @app.route digunakan untuk menghubungkan URL dengan fungsi yang sesuai
# URL '/uploads/<filename>' akan mengarahkan ke fungsi upload_img
@app.route('/uploads/<filename>')
def upload_img(filename):
    # Menggunakan send_from_directory untuk mengirimkan file dari folder UPLOAD_FOLDER
    # Parameter app.config['UPLOAD_FOLDER'] digunakan untuk menentukan folder sumber file
    # Parameter filename adalah nama file yang diminta dari URL
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# URL '/images/<filename>' akan mengarahkan ke fungsi enhancement_img
@app.route('/images/<filename>')
def enhancement_img(filename):
    # Menggunakan send_from_directory untuk mengirimkan file dari folder ENHANCEMENT_FOLDER
    # Parameter app.config['ENHANCEMENT_FOLDER'] digunakan untuk menentukan folder sumber file
    # Parameter filename adalah nama file yang diminta dari URL
    return send_from_directory(app.config['ENHANCEMENT_FOLDER'], filename)

# Fungsi untuk melakukan median filtering pada data menggunakan ukuran filter tertentu.
def median_filter(data, filter_size):
    # Membuat array kosong untuk menyimpan nilai sementara dari area yang akan di-filter
    temp = []
    
    # Menghitung indeks tengah dari ukuran filter
    indexer = filter_size // 2
    
    # Membuat array kosong yang memiliki bentuk yang sama dengan data
    # Ini akan menjadi array hasil setelah median filtering
    data_final = np.zeros_like(data)

    # Looping melalui baris data
    for i in range(len(data)):
        # Looping melalui kolom data
        for j in range(len(data[0])):
            # Looping untuk mengakses area sekitar titik (i, j) sesuai ukuran filter
            for z in range(filter_size):
                if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
                    # Jika indeks berada di luar batas gambar, tambahkan nol ke array sementara
                    for c in range(filter_size):
                        temp.append(0)
                else:
                    if j + z - indexer < 0 or j + indexer > len(data[0]) - 1:
                        # Jika indeks berada di luar batas gambar, tambahkan nol ke array sementara
                        temp.append(0)
                    else:
                        # Jika indeks berada dalam batas gambar, tambahkan nilai dari area filter ke array sementara
                        for k in range(filter_size):
                            temp.append(data[i + z - indexer][j + k - indexer])

            # Mengurutkan nilai dalam array sementara
            temp.sort()
            
            # Menentukan nilai median dan memasukkannya ke array hasil median filtering
            data_final[i][j] = temp[len(temp) // 2]
            
            # Mengosongkan array sementara untuk digunakan kembali dalam iterasi selanjutnya
            temp = []
    
    # Mengembalikan data hasil median filtering
    return data_final

# Fungsi untuk melakukan median filtering pada citra berwarna (RGB) menggunakan ukuran filter tertentu.
def median_filter_rgb(img, filter_size):
    # Memisahkan kanal warna (R, G, B) dari citra menggunakan cv2.split
    channels = cv2.split(img)
    
    # Membuat array kosong untuk menyimpan kanal-kanal yang telah difilter
    filtered_channels = []
    
    # Melakukan loop melalui setiap kanal warna
    for channel in channels:
        # Memanggil fungsi median_filter untuk setiap kanal warna
        filtered_channel = median_filter(channel, filter_size)
        
        # Menambahkan kanal hasil filter ke dalam array filtered_channels
        filtered_channels.append(filtered_channel)
    
    # Menggabungkan kembali kanal-kanal warna yang telah difilter untuk membentuk citra hasil
    return cv2.merge(filtered_channels)

# Fungsi untuk melakukan histogram equalization pada citra menggunakan model warna YUV.
def histogram_equalization(img):
    # Mengubah citra dari warna BGR ke warna YUV
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    
    # Melakukan equalisasi histogram pada kanal Y (luminance) dari citra
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    
    # Mengubah citra kembali ke warna BGR setelah equalisasi histogram
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    
    # Mengembalikan citra hasil setelah histogram equalization
    return img_output


# Fungsi untuk menampilkan histogram citra dan menyimpannya dalam format gambar PNG.
def show_image_histogram(image_path, save_path):
    # Membaca citra dari path yang diberikan
    image = cv2.imread(image_path)

    # Memeriksa apakah citra berhasil dibaca atau tidak
    if image is None:
        # Jika citra tidak bisa dibaca, cetak pesan kesalahan
        print("Gambar tidak dapat dibaca. Pastikan path gambar benar dan gambar ada.")
        return
    
    # Membuat objek plot untuk menampilkan histogram
    plt.figure(figsize=(7, 5))

    # Memeriksa apakah citra berwarna (RGB) atau grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Jika citra berwarna, hitung histogram untuk setiap kanal warna (R, G, B)
        hist_channels = []
        for i in range(image.shape[2]):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            hist_channels.append(hist)

        # Definisikan warna-warna yang akan digunakan untuk plot
        colors = ['red', 'green', 'blue']
        
        # Loop melalui setiap kanal warna dan plot histogramnya
        for i, color in enumerate(colors):
            plt.plot(hist_channels[i], color=color)
        
        # Menambahkan judul dan label sumbu pada plot
        plt.title('Histogram')
        plt.xlabel('Nilai Pixel')
        plt.ylabel('Frekuensi')
    else:
        # Jika citra grayscale, hitung histogram untuk citra tersebut
        hist_gray = cv2.calcHist([image], [0], None, [256], [0, 256])

        # Plot histogram citra grayscale dengan warna hitam
        plt.plot(hist_gray, color='black')
        
        # Menambahkan judul dan label sumbu pada plot
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

    # Convert objek BytesIO ke base64-encoded string untuk ditampilkan dalam halaman web atau aplikasi
    histogram_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()

    # Mengembalikan gambar histogram dalam format base64
    return histogram_base64


# Route handler untuk URL '/' dengan metode GET
@app.route('/', methods=['GET'])
def index():
    # Mengembalikan hasil render dari template 'index.html'
    return render_template('index.html')

# Route handler untuk URL '/predict' dengan metode GET dan POST
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Memeriksa jika metode permintaan adalah POST
    if request.method == 'POST':
        # Mendapatkan file yang diunggah (dalam bentuk objek) dan algoritma yang dipilih dari form
        f = request.files.get('file')
        algorithm = request.form.get('Algorithm')

        # Memeriksa jika tidak ada file gambar yang diunggah
        if not f:
            # Menampilkan pesan flash dan kembali ke halaman 'index.html'
            flash('Please select an image')
            return render_template('index.html')
        
        # Mendapatkan direktori induk dari script yang sedang berjalan
        basepath = os.path.dirname(__file__)
        
        # Mendapatkan nama file dan ekstensinya
        filename, file_extension = os.path.splitext(secure_filename(f.filename))
        
        # Menggabungkan direktori untuk menyimpan file gambar yang diunggah
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        
        # Menyimpan file gambar yang diunggah di path yang telah ditentukan
        f.save(file_path)

        # Mendapatkan nama file saja dari path yang telah ditentukan
        file_name = os.path.basename(file_path)
        
        # Membaca citra menggunakan OpenCV
        img = cv2.imread(file_path)

        # Menentukan path untuk menyimpan histogram citra sebelum proses
        histogram_path_original = os.path.join(app.config['HISTOGRAM_FOLDER'], f'{file_name}_before.png')
        
        # Menghasilkan gambar histogram sebelum proses dan menyimpannya sebagai gambar PNG
        histogram_base64_original = show_image_histogram(file_path, histogram_path_original)

        # Memeriksa algoritma yang dipilih dan memproses citra sesuai dengan algoritma yang dipilih
        if algorithm == "Med_Filter":
            # Menentukan nama file untuk citra hasil pemrosesan menggunakan median filter
            enhance_fname = f"{filename}_medfil{file_extension}"
            
            # Memproses citra menggunakan median filter
            enhancement = median_filter_rgb(img, filter_size=3)
            
            # Menentukan path untuk menyimpan citra hasil pemrosesan
            enhancement_path = os.path.join(basepath, 'images', secure_filename(enhance_fname))
            fname = os.path.basename(enhancement_path)
            
            # Menyimpan citra hasil pemrosesan
            cv2.imwrite(enhancement_path, enhancement)

            # Menentukan path untuk citra hasil pemrosesan
            enhancement_path = os.path.join(basepath, 'images', secure_filename(enhance_fname))
            
            # Menentukan path untuk menyimpan histogram hasil pemrosesan
            histogram_path_enhanced = os.path.join(app.config['HISTOGRAM_FOLDER'], f'{enhance_fname}_after.png')
            
            # Menghasilkan gambar histogram setelah pemrosesan dan menyimpannya sebagai gambar PNG
            histogram_base64_enhanced = show_image_histogram(enhancement_path, histogram_path_enhanced)

            # Mengembalikan hasil render dari template 'predict.html'
            return render_template('predict.html', file_name=file_name,
                                   enhancement_file=fname,
                                   before_histogram_base64=histogram_base64_original,
                                   after_histogram_base64=histogram_base64_enhanced)
        
        elif algorithm == "Hist_Equalization":
            # Menentukan nama file untuk citra hasil pemrosesan menggunakan histogram equalization
            enhance_fname = f"{filename}_hiseq{file_extension}"
            
            # Memproses citra menggunakan histogram equalization
            enhancement = histogram_equalization(img)
            
            # Menentukan path untuk menyimpan citra hasil pemrosesan
            enhancement_path = os.path.join(basepath, 'images', secure_filename(enhance_fname))
            fname = os.path.basename(enhancement_path)
            
            # Menyimpan citra hasil pemrosesan
            cv2.imwrite(enhancement_path, enhancement)

            # Menentukan path untuk citra hasil pemrosesan
            enhancement_path = os.path.join(basepath, 'images', secure_filename(enhance_fname))
            
            # Menentukan path untuk menyimpan histogram hasil pemrosesan
            histogram_path_enhanced = os.path.join(app.config['HISTOGRAM_FOLDER'], f'{enhance_fname}_after.png')
            
            # Menghasilkan gambar histogram setelah pemrosesan dan menyimpannya sebagai gambar PNG
            histogram_base64_enhanced = show_image_histogram(enhancement_path, histogram_path_enhanced)

            # Mengembalikan hasil render dari template 'predict.html'
            return render_template('predict.html', file_name=file_name,
                                   enhancement_file=fname,
                                   before_histogram_base64=histogram_base64_original,
                                   after_histogram_base64=histogram_base64_enhanced)
        
        else:
            # Jika algoritma yang dipilih tidak sesuai dengan pilihan yang ada
            # Menampilkan pesan flash dan kembali ke halaman 'index.html'
            flash('Please select algorithm')
            return render_template('index.html')

    # Mengembalikan string kosong jika metode permintaan bukan POST
    return ""
    

# Menjalankan aplikasi Flask jika skrip ini dijalankan sebagai program utama
if __name__ == '__main__':
    # Menjalankan aplikasi dalam mode debug pada host "localhost" dan port 8080
    app.run(debug=True, host="localhost", port=8080)
