import os
from model_functions import *
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename


app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
MODEL_PATH = './resources/model.zip'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

model = load_model(MODEL_PATH)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            eqn, mapping, solution = predict(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            prediction = {'equation': eqn, 'mapping': mapping, 'solution': solution}
            return prediction
#            return render_template('index.html', prediction=prediction)
#    return render_template('index.html')
    return 'Nothing'


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict(img_path):
    cropped_eqn_imgs, rects = crop_image(img_path)
    most_probable = classify(model, cropped_eqn_imgs)
    pred_labbels = most_probable[:, -1]
    symbols = labels_to_symbols(rects, pred_labbels)
    # eqn_before_map = labels_to_eqn(pred_labbels)
    # print_labels(cropped_eqn_imgs,most_probable)
    edu_symbols = educated_parse(symbols)
    equation, mapping = toEqn(edu_symbols)
    equationSpilited = equation.split("=")

    try:
        solution = solve(equationSpilited[0], mapping)
        return equation, mapping, solution
    except:
        return equation, mapping, "No Solution"


if __name__ == '__main__':
    app.run(debug=True)
