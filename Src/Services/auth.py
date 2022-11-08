from  flask import Blueprint
from  flask import jsonify
from datetime import datetime as dt
# from Src.Constants.status_codes import HTTP_200_OK, HTTP_201_CREATED, HTTP_400_BAD_REQUEST, HTTP_401_UNAUTHORIZED, HTTP_409_CONFLICT
from  flask import request
from werkzeug.security import generate_password_hash,check_password_hash
# # from wtforms import Form, BooleanField, StringField, PasswordField, validators
from Src.Model.user import db, User

# register auth file
auth = Blueprint("auth", __name__)

# register route

@auth.post('/register')
def register():
    username = request.json['username']
    email = request.json['email']
    password = request.json['password']
    
    # elsse
    pwd_hash = generate_password_hash(password)
    if username and email and pwd_hash:
        new_user = User(
            name=username,
            email=email,
            password=pwd_hash
        )
    db.session.add(new_user)  # Adds new User record to database
    db.session.commit() 
    return jsonify({
        'message': "User created",
        'data' : {'name':username , 'email' : email},
        'status' :  200
    })




@auth.post('/login')
def login():
    useremail = request.json['email']
    userpassword = request.json['password']
    
    # elsse
    userobject = User.query.filter_by(email=useremail).first()
     
     

    if userobject.email == useremail and check_password_hash(userobject.password ,userpassword):
        
        return jsonify({
            'message': "User Authenticated Success",
            'data' : {'email' : useremail ,'name' :  userobject.name},
            'status' :  200
        })
    else:
         return jsonify({
            'message': "User denied",
            'status' :  500
        })
