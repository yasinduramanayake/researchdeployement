from flask import Blueprint
from flask import jsonify ,make_response
from flask import Flask
from datetime import datetime as dt
# from Src.Constants.status_codes import HTTP_200_OK, HTTP_201_CREATED, HTTP_400_BAD_REQUEST, HTTP_401_UNAUTHORIZED, HTTP_409_CONFLICT
from flask import request
import cv2
import os
from  Src.Services.detect_object import *
from  Src.Services.Harvestschema import ProductSchema
import numpy as np
from Src.Model.harvest import db, Harvest


# register auth file

UPLOAD_FOLDER = './upload'
products_schema = ProductSchema(many=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
Harvestdetection = Blueprint("Harvestdetection", __name__)

# register route


@Harvestdetection.post('/detect')
def detect():
   
 
    # Load Aruco detector
    parameters = cv2.aruco.DetectorParameters_create()
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)


# Load Object Detector
    detector = HomogeneousBgDetector()

# Load Image
    image =  request.files['image']
    path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
    image.save(path)
    img = cv2.imread(path)

# Get Aruco marker

    corners, _, _ = cv2.aruco.detectMarkers(
        img, aruco_dict, parameters=parameters)

# Draw polygon around the marker

    int_corners = np.int0(corners)
    cv2.polylines(img, int_corners, True, (0, 255, 0), 5)

# Aruco Perimeter
    aruco_perimeter = cv2.arcLength(corners[0], True)

# Pixel to cm ratio
    pixel_cm_ratio = aruco_perimeter / 20

    contours = detector.detect_objects(img)

# Draw objects boundaries

    width = []

    height = []

    for cnt in contours:
        # Get rect

        rect = cv2.minAreaRect(cnt)
        (x, y), (w, h), angle = rect

    # Get Width and Height of the Objects by applying the Ratio pixel to cm
        object_width = w / pixel_cm_ratio
        object_height = h / pixel_cm_ratio

    # Display rectangle
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
        cv2.polylines(img, [box], True, (255, 0, 0), 2)
        cv2.putText(img, "Width {} cm".format(round(object_width, 1)), (int(
            x - 100), int(y - 20)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
        cv2.putText(img, "Height {} cm".format(round(object_height, 1)), (int(
            x - 100), int(y + 15)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)


        if object_width > 1.0 and object_height > 1.0:
            width.append(object_width)
            height.append(object_height)

       

    # cv2.imshow("Image", img)
    # cv2.waitKey(0)
    harvestdata = Harvest(
            full_harvest=len(contours) - 1,
            can_be_utilized=len(width) - 1,
        )
    db.session.add(harvestdata)  # Adds new User record to database
    db.session.commit() 
    return jsonify({
        'widtht': width,
        'height': height,
        'object Count': len(contours) - 1
    })


@Harvestdetection.get('/getdetections')

def getdetections():
    
    res =  Harvest.query.all()
    result = products_schema.dump(res)
    return jsonify(result)
