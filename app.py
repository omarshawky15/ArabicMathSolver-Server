import os
import time

import requests
from cv2 import imwrite
from flask import Flask, flash, request, redirect, render_template
from keras.models import load_model
from werkzeug.utils import secure_filename

from classification import classify, labels_to_symbols
from evaluation import calculate, polynomial, differentiate, integrate
from image_utility import crop_image
from parsing import *
from translation import translate_to_arabic_html

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

            arabic_expr, arabic_sol = translate_to_arabic_html(expression, solution, mapping)
            prediction = {'expression': arabic_expr, 'solution': arabic_sol, 'error': str(error)}
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


def predict(img_path, crop_with_labels=False):
    if not os.path.exists(img_path):  # Image not found in path
        return '', {}

    cropped_eqn_imgs, rects = crop_image(img_path)
    most_probable = classify(model, cropped_eqn_imgs)
    if crop_with_labels:
        sendCropped(cropped_eqn_imgs, most_probable)
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
        if i % 4 == 0:  # Every 5 images wait 5 seconds because of limit of the number of messages
            time.sleep(5)


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


def runLocal(img_path='', img_link=''):
    if img_path == '' and img_link == '':
        return
    elif img_link != '':  # Download image by link
        img_path = UPLOAD_FOLDER + img_link.split('/')[-1]
        if not os.path.exists(img_path):
            open(img_path, 'wb').write(requests.get(img_link, allow_redirects=True).content)
    else:  # Continue with local image path
        pass

    expression, mapping = predict(img_path)
    solution, error = polynomial(expression, mapping)
    eng_prediction = {'expression': expression, 'mapping': str(mapping), 'solution': str(solution), 'error': str(error)}
    print(eng_prediction)

    arabic_expr, arabic_sol = translate_to_arabic_html(expression, solution, mapping)
    arabic_prediction = {'expression': arabic_expr, 'solution': arabic_sol, 'error': str(error)}
    print(arabic_prediction)


if __name__ == '__main__':
    # Normal running of server (Don't forget to uncomment this after testing)
    app.run()

    # Run server locally with debug mode on
    # app.run(debug=True)

    # Test prediction with local images without running server
    # runLocal(img_path='', img_link='')
