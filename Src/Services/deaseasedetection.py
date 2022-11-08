from flask import Flask, request
from flask_cors import CORS
from flask import Blueprint
import json
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model as lm
import cv2
# import pandas as pd
from flask import jsonify ,make_response
from  Src.Services.deseaseschema import DeseaseSchema , ClasificationSchema
import numpy as np
import werkzeug
from Src.Model.user import Deseases, db ,Clasification
import os

# from pycaret.classification import *

# model = load_model('CapsicumDiseaseDetection/Naive_Bayes')

app = Flask(__name__)
CORS(app)
app.config.from_object('config.Config')
db.init_app(app)
deseases_schema = DeseaseSchema(many=True)
clasification_schema = ClasificationSchema(many=True)
deaseasedetection = Blueprint("deaseasedetection", __name__)
part_01_model = lm('QuolityDetection/model.h5')
# predict_model = lm('CapsicumDiseaseDetection/model.h5')
data = np.load('CapsicumDiseaseDetection/Processed_Data/data.npy')
target = np.load('CapsicumDiseaseDetection/Processed_Data/target.npy')

category = {0: 'Phytophthora Blight', 1: 'Powdery Mildew', 2: 'Cercospora Leaf Spot'}

model = Sequential()

model.add(Conv2D(256, (3, 3), input_shape=data.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))

model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.load_weights('CapsicumDiseaseDetection/model.h5')

# part_01_model = lm('QuolityDetection/model.h5')


@deaseasedetection.route('/CapsicumDiseaseDetection', methods=['GET', 'POST'])
def CapsicumDiseaseDetection():
    imagefile = request.files['image']
    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("\nReceived image File name : " + imagefile.filename)
    imagefile.save('upload/' + filename)

    img = cv2.imread('upload/' + filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (50, 50))
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, 50, 50, 1))

    result = model.predict(reshaped)
    label = np.argmax(result, axis=1)[0]
    prob = np.max(result, axis=1)[0]
    prob = round(prob, 2) * 100

    print('Confident : ', np.max(result, axis=1)[0])
    deseasedata = Deseases(
        desease = str(category[label]),
    )
    db.session.add(deseasedata)  # Adds new User record to database
    db.session.commit()

    return jsonify({
       'desease' : str(category[label])
    })
    # return json.loads('{ "result" : "' + str(category[label]) + '"}')



@deaseasedetection.route('/Alldeseases', methods=['GET'])

def getdeseases():
    
    res =  Deseases.query.all()
    result = deseases_schema.dump(res)
    return jsonify(result)



@deaseasedetection.route('/Classification', methods=['GET', 'POST'])
def Classification():
    Temperature = request.form['Temperature']
    Humidity = request.form['Humidity']

    try:
        data = np.array([['Temperature', 'Humidity'], [Temperature, Humidity]])

        result = predict_model(model, data=pd.DataFrame(data=data[0:, 0:], index=data[0:, 0], columns=data[0, 0:])).iat[
                1, 2]
        print('Predicted result ' + str(result))
        clasifydata = Clasification(
                desease = str(result),
            )
        db.session.add(clasifydata)  # Adds new User record to database
        db.session.commit()

        return jsonify({
        'desease' : str(result)
        })    
        # return json.loads('{ "result" : "' + str(result) + '"}')
    except Exception as e:
        print(e)
        clasifydata = Clasification(
            desease = str('1'),
        )
        db.session.add(clasifydata)  # Adds new User record to database
        db.session.commit()

        return jsonify({
        'desease' : "1"
        })
        return json.loads('{ "result" : "1"}')


@deaseasedetection.route('/Allclasifications', methods=['GET'])

def getclasifications():
    
    res =  Clasification.query.all()
    result = clasification_schema.dump(res)
    return jsonify(result)

def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img
