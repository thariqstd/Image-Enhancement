import os
import numpy as np
import cv2
from PIL import Image
from flask import Flask, flash, request, render_template, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__, static_url_path='')
app.secret_key = os.urandom(24)

app.config['ENHANCEMENT_FOLDER'] = 'images'
app.config['UPLOAD_FOLDER'] = 'uploads'


@app.route('/uploads/<filename>')
def upload_img(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/images/<filename>')
def enhancement_img(filename):
    return send_from_directory(app.config['ENHANCEMENT_FOLDER'], filename)


def median_filter(data, filter_size):
    temp = []
    indexer = filter_size // 2
    data_final = np.zeros_like(data)
    for i in range(len(data)):
        for j in range(len(data[0])):
            for z in range(filter_size):
                if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
                    for c in range(filter_size):
                        temp.append(0)
                else:
                    if j + z - indexer < 0 or j + indexer > len(data[0]) - 1:
                        temp.append(0)
                    else:
                        for k in range(filter_size):
                            temp.append(data[i + z - indexer][j + k - indexer])

            temp.sort()
            data_final[i][j] = temp[len(temp) // 2]
            temp = []
    return data_final


def median_filter_rgb(img, filter_size):
    channels = cv2.split(img)
    filtered_channels = []
    for channel in channels:
        filtered_channel = median_filter(channel, filter_size)
        filtered_channels.append(filtered_channel)
    return cv2.merge(filtered_channels)


def histogram_equalization(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img_output


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        f = request.files['file']
        style = request.form.get('style')

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        file_name = os.path.basename(file_path)

        img = cv2.imread(file_path)

        if style == "Style1":
            enhance_fname = file_name + "_style1.jpg"
            enhancement = median_filter_rgb(img, filter_size=3)
            enhancement_path = os.path.join(basepath, 'images', secure_filename(enhance_fname))
            fname = os.path.basename(enhancement_path)
            cv2.imwrite(enhancement_path, enhancement)
            return render_template('predict.html', file_name=file_name, enhancement_file=fname)
        elif style == "Style2":
            enhance_fname = file_name + "_style2.jpg"
            enhancement = histogram_equalization(img)
            enhancement_path = os.path.join(basepath, 'images', secure_filename(enhance_fname))
            fname = os.path.basename(enhancement_path)
            cv2.imwrite(enhancement_path, enhancement)
            return render_template('predict.html', file_name=file_name, enhancement_file=fname)
        else:
            flash('Please select style')
            return render_template('index.html')

    return ""


if __name__ == '__main__':
    app.run(debug=True, host="localhost", port=8080)
