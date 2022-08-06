from distutils.log import debug
from traceback import print_tb
from flask import Flask,request,flash, Response, send_file, jsonify,render_template,request, redirect, url_for
from werkzeug.utils import secure_filename

from PIL import Image
#this is comment

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import uuid
import os
import base64

from model import U2NET
from torch.autograd import Variable
from skimage import io, transform
from PIL import Image


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
UPLOAD_FOLDER = 'static/first_image'
GET_IMAGE = 'static/results'

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['GET_IMAGE'] = GET_IMAGE
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

currentDir = os.path.dirname(__file__)
global output_name
def save_output(image_name, output_name, pred, d_dir, type):
     
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    im = Image.fromarray(predict_np*255).convert('RGB')
    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]))
    pb_np = np.array(imo)
    if type == 'image':
        mask = pb_np[:, :, 0]
        mask = np.expand_dims(mask, axis=2)
        imo = np.concatenate((image, mask), axis=2)
        imo = Image.fromarray(imo, 'RGBA')
    
    imo.save(d_dir+output_name)
    return render_template('index.html', filename=output_name)
    
    



def removeBg(imagePath,unique_filename):
    inputs_dir = os.path.join(currentDir, 'static/inputs/')
    results_dir = os.path.join(currentDir, 'static/results/')
    masks_dir = os.path.join(currentDir, 'static/masks/')

    with open(imagePath, "rb") as image:
        f = image.read()
        img = bytearray(f)

    nparr = np.frombuffer(img, np.uint8)

    if len(nparr) == 0:
        return jsonify({'msg': 'Empty image'})

    try:
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except:
    
        return jsonify({'msg': 'Empty image'})

    #unique_filename = str(uuid.uuid4())
    cv2.imwrite(inputs_dir+unique_filename+'.jpg', img)

    image = transform.resize(img, (320, 320), mode='constant')

    tmpImg = np.zeros((image.shape[0], image.shape[1], 3))

    tmpImg[:, :, 0] = (image[:, :, 0]-0.485)/0.229
    tmpImg[:, :, 1] = (image[:, :, 1]-0.456)/0.224
    tmpImg[:, :, 2] = (image[:, :, 2]-0.406)/0.225

    tmpImg = tmpImg.transpose((2, 0, 1))
    tmpImg = np.expand_dims(tmpImg, 0)
    image = torch.from_numpy(tmpImg)

    image = image.type(torch.FloatTensor)
    image = Variable(image)

    d1, d2, d3, d4, d5, d6, d7 = net(image)
    pred = d1[:, 0, :, :]
    ma = torch.max(pred)
    mi = torch.min(pred)
    dn = (pred-mi)/(ma-mi)
    pred = dn

    #def save_output(image_name, output_name, pred, d_dir, type):
    save_output(inputs_dir+unique_filename+'.jpg', unique_filename +'.png', pred, results_dir, 'image')
    # print("image name ", inputs_dir )
    # print("pred ", pred )
    # print("result ", results_dir )
    save_output(inputs_dir+unique_filename+'.jpg', unique_filename +'.png', pred, masks_dir, 'mask')

model_name = 'u2net'
model_dir = os.path.join(currentDir, 'saved_models',
                         model_name, model_name + '.pth')
                        
net = U2NET(3, 1)
if torch.cuda.is_available():
    net.load_state_dict(torch.load(model_dir))
    net.cuda()
else:
    net.load_state_dict(torch.load(model_dir, map_location='cpu'))

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#############
@app.route('/')
def home():
    return render_template("index.html")
#################################    



@app.route('/', methods=['POST'])
def upload_image():
    if request.method == 'POST':
    
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            unique_filename = str(uuid.uuid4())
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            removeBg(os.path.join(app.config['UPLOAD_FOLDER'], filename),unique_filename)
            get_image = os.path.join(app.config['GET_IMAGE'],  unique_filename+'.png')
            encoded_string = base64.b64encode(open(get_image, 'rb').read())
            #show_index(get_image)

            # image = Image.open(get_image)
            # image.show()
            base64string = 'data:image/jpeg;base64,'+str(encoded_string).split('\'')[1]
    return render_template('index.html', user_image = get_image)
    


def show_index(full_filename):
    print("#################")
    #full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'shovon.jpg')
    return render_template("index.html", user_image = "static/results/output.png")
    
    #return jsonify({'msg': 'success','encode': 'base64string'})

@app.route('/display/<filename>')
def display_image(filename):
    print("#########################")
    print("display_image filename: " + filename)
    return redirect(url_for('static', filename='inputs/' + filename))



if __name__ == "__main__":
    app.run(debug=False)   