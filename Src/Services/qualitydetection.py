from flask import Flask, request
from flask_cors import CORS
from flask import Blueprint
import json
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model as lm
import cv2
from flask import jsonify ,make_response
from  Src.Services.Qualitychema import QualitySchema
import numpy as np
from Src.Model.Quality import db, Quality
import werkzeug
import os

# from pycaret.classification import *

# model = load_model('CapsicumDiseaseDetection/Naive_Bayes')

app = Flask(__name__)
CORS(app)
app = Flask(__name__, instance_relative_config=False)
app.config.from_object('config.Config')
db.init_app(app)
quality_schema = QualitySchema(many=True)
qualitydetection = Blueprint("qualitydetection", __name__)
part_01_model = lm('QuolityDetection/model.h5')


@qualitydetection.route('/QuolityDetection', methods=['GET', 'POST'])
def QuolityDetection():
    imagefile = request.files['image']
    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("\nReceived image File name : " + imagefile.filename)
    imagefile.save('upload/' + filename)

    img = cv2.imread('upload/' + filename)
    # img = cv2.imread('Test/' + name)
    img = np.asarray(img)
    img = cv2.resize(img, (32, 32))
    img = preProcessing(img)
    img = img.reshape(1, 32, 32, 1)
    predictions = part_01_model.predict(img)
    label = np.argmax(predictions, axis=1)[0]
    # probVal = np.amax(predictions)
    a = np.amax(predictions) * 100
    x = "%.2f" % round(a, 2)
    print(x)
    qualitydata = Quality(
        quality_is = str(label),
    )
    db.session.add(qualitydata)  # Adds new User record to database
    db.session.commit()

    return jsonify({
       'quality_is' : str(label)
    })



@qualitydetection.route('/AllQualities', methods=['GET'])

def getqualities():
    
    res =  Quality.query.all()
    result = quality_schema.dump(res)
    return jsonify(result)




def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img

