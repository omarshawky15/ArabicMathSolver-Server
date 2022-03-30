import requests
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
from model_functions import *

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
            prediction = {'equation': eqn, 'mapping': str(mapping), 'solution': str(solution)}
            # sendImage('equation :' + eqn + '\nmapping : ' + str(mapping) + '\nsolution :' + str(solution),
            # os.path.join(app.config['UPLOAD_FOLDER'], filename))
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return prediction
            # return render_template('index.html', prediction=prediction)
    return render_template('index.html')
    # return 'Nothing'


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
    equation, mapping = toEqn(edu_symbols, initial_mapping)
    equationSpilited = equation.split("=")

    try:
        solution = solve(equationSpilited[0], mapping)
        return equation, mapping, solution
    except:
        return equation, mapping, "No Solution"


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
    # equation, mapping, solution = predict('IMG_PATH')
    # print('Eqn: ' + equation)
    # print('Mapping: ', end='')
    # print(mapping)
    # print('Solution: ', end='')
    # print(solution)
