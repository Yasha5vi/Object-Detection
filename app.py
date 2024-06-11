from flask import Flask, render_template, Response, request, redirect, url_for
from objectdetection import videoStream, imageCapture
import os

app = Flask(__name__)
UPLOAD_FOLDER = './static/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/image')
def image_page():
    return render_template('image.html')  # Image page

@app.route('/stream')
def video_page():
    return render_template('stream.html')  # Video page

@app.route('/video_feed')
def video_feed():
    return Response(videoStream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)
    
    if file:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(image_path)
        imageCapture(image_path)
        return "Success"

# if __name__ == '__main__':
#     app.run(debug=True)
