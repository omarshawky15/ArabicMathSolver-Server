import os

import requests
from cv2 import imwrite
from flask import Flask, flash, request, redirect, render_template
from keras.models import load_model
from werkzeug.utils import secure_filename

from classification import classify, labels_to_symbols
from evaluation import calculate, polynomial, differentiate, integrate
from image_utility import crop_image
from parsing import *

app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
MODEL_PATH = r'./resources/model.h5'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

model = load_model(MODEL_PATH)


@app.route('/', methods=['GET', 'POST'])
@app.route('/<problem>', methods=['GET', 'POST'])
def index(problem='calculate'):
    if request.method == 'POST':
        returned_file_result = get_file()
        if returned_file_result is None:
            return redirect(request.url)
        else:
            file, filepath = get_file()
            file.save(filepath)
            expression, mapping = predict(filepath)

            if problem == 'polynomial':
                solution, error = polynomial(expression, mapping)
            elif problem == 'differentiate':
                solution, error = differentiate(expression, mapping)
            elif problem == 'integrate':
                solution, error = integrate(expression, mapping)
            else:
                solution, error = calculate(expression, mapping)

            prediction = {'expression': expression, 'mapping': str(mapping), 'solution': str(solution),
                          'error': str(error)}
            # sendImage('equation :' + eqn + '\nmapping : ' + str(mapping) + '\nsolution :' + str(solution),
            os.remove(filepath)
            return prediction
            # return render_template('index.html', prediction=prediction)
    return render_template('index.html')
    # return 'Nothing'


def get_file():
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return None
    file = request.files['file']
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        flash('No selected file')
        return None
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        return file, filepath
    return None


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def sendCropped(cropped_eqn_imgs, most_probable):
    for i in range(len(cropped_eqn_imgs)):
        cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], str(most_probable[i]) + '_image.png'),
                    cropped_eqn_imgs[i])
        sendImage(str([all_labels[w] for w in most_probable[i][::-1]]),
                  os.path.join(app.config['UPLOAD_FOLDER'], str(most_probable[i]) + '_image.png'))



def predict(img_path):
    cropped_eqn_imgs, rects = crop_image(img_path)
    most_probable = classify(model, cropped_eqn_imgs)
    # sendCropped(cropped_eqn_imgs, most_probable)
    pred_labbels = most_probable[:, -1]
    symbols = labels_to_symbols(rects, pred_labbels)
    # eqn_before_map = labels_to_eqn(pred_labbels)
    # print_labels(cropped_eqn_imgs,most_probable)
    edu_symbols = educated_parse(symbols)
    initial_mapping = {}
    expression, mapping = toExpr(edu_symbols, initial_mapping)
    return expression, mapping


def sendCropped(cropped_eqn_imgs, most_probable):
    for i in range(len(cropped_eqn_imgs)):
        imwrite(os.path.join(app.config['UPLOAD_FOLDER'], str(most_probable[i]) + '_image.png'),
                cropped_eqn_imgs[i])
        sendImage(str([all_labels[w] for w in most_probable[i][::-1]]),
                  os.path.join(app.config['UPLOAD_FOLDER'], str(most_probable[i]) + '_image.png'))


def sendImage(filename, file):
    token = 'TOKEN'
    headers = {'Authorization': token}
    files = {
        "file": (filename + '.png', open(file, 'rb'))  # The picture that we want to send in binary
    }
    requests.post(f'LINK',
                  headers=headers,
                  json={'content': filename+'.png'})
    requests.post(f'LINK',
                  headers=headers,
                  files=files)

if __name__ == '__main__':
    # Normal running of server (Don't forget to uncomment this after testing)
    app.run()

    # Run server locally with debug mode on
    # app.run(debug=True)

    # Test prediction with local images without running server
    # expression, mapping = predict('uploads/log.jpg')
    # solution, error = calculate(expression, mapping)
    # prediction = {'expression': expression, 'mapping': str(mapping), 'solution': str(solution),
    #               'error': str(error)}
    # print(prediction)
