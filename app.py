from flask import Flask, request, jsonify, render_template
import pytesseract
import cv2
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

OUTPUT_FOLDER = 'static/outputs'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']
    img = Image.open(file)

    # Convert the image to RGB if it is not in that format
    if img.mode != 'RGB':
        img = img.convert('RGB')

    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Use Tesseract to extract text and get bounding box data
    boxes = pytesseract.image_to_data(img_cv, output_type=pytesseract.Output.DICT)

    # Prepare text and bounding boxes for the response
    recognized_data = []
    for i in range(len(boxes['text'])):
        if int(boxes['conf'][i]) > 60:  # Only consider confident detections
            text = boxes['text'][i]
            (x, y, w, h) = (boxes['left'][i], boxes['top'][i], boxes['width'][i], boxes['height'][i])
            recognized_data.append({
                'text': text,
                'x': x,
                'y': y,
                'w': w,
                'h': h
            })

    # Save the output image
    output_path = os.path.join(OUTPUT_FOLDER, 'output_image.png')
    cv2.imwrite(output_path, img_cv)

    return jsonify({'data': recognized_data, 'output_image': output_path})

if __name__ == '__main__':
    app.run(debug=True)
