import os
from flask import Flask, request, render_template, send_from_directory
import cv2
import numpy as np
from src.roberts import Roberts
from src.sobel import Sobel
from src.canny import Canny
from src.utils import Helper

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create upload and output folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return render_template('index.html', error='No file selected')
        
        file = request.files['file']
        
        if file.filename == '':
            return render_template('index.html', error='No file selected')
        
        if file and allowed_file(file.filename):
            # Save uploaded file
            input_path = os.path.join(UPLOAD_FOLDER, 'input.png')
            file.save(input_path)
            
            # Read and process image
            img = Helper.read_image(input_path)
            if len(img.shape) == 3:  # If RGB image
                img = Helper.convert_greyscale(img)
            
            # Apply all edge detection algorithms
            # Roberts
            roberts_detector = Roberts(img)
            roberts_edges = roberts_detector.find_edges()
            cv2.imwrite(os.path.join(OUTPUT_FOLDER, 'roberts.png'), roberts_edges)
            
            # Sobel
            sobel_detector = Sobel(img)
            sobel_edges = sobel_detector.find_edges()
            cv2.imwrite(os.path.join(OUTPUT_FOLDER, 'sobel.png'), sobel_edges)
            
            # Canny
            canny_detector = Canny(img)
            canny_edges = canny_detector.canny()
            cv2.imwrite(os.path.join(OUTPUT_FOLDER, 'canny.png'), canny_edges)
            
            return render_template('index.html', 
                                 input_image='uploads/input.png',
                                 roberts_image='outputs/roberts.png',
                                 sobel_image='outputs/sobel.png',
                                 canny_image='outputs/canny.png',
                                 show_results=True)
    
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/outputs/<filename>')
def output_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
