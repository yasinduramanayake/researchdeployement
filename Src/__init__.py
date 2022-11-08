from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from Src.Services.auth  import auth
from Src.Services.Harvestdetection import Harvestdetection
from Src.Model.user import db
from Src.Services.Harvestschema import ProductSchema
from Src.Services.qualitydetection import qualitydetection
from Src.Services.deaseasedetection import deaseasedetection
from flask_cors import CORS




def init_app():
    app = Flask(__name__, instance_relative_config=False)
    app.config.from_object('config.Config')
    db.init_app(app)
    app.register_blueprint(auth)
    app.register_blueprint(Harvestdetection)
    app.register_blueprint(qualitydetection)
    app.register_blueprint(deaseasedetection)
    CORS(app)
    cors = CORS(app, resources={"/api/*": {"origins": "*"}})

    with app.app_context():
        db.create_all()  # Create sql tables for our data models
        return app 

        