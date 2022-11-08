from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()


class User(db.Model):

    __tablename__ = 'user'
    id = db.Column(
        db.Integer,
        primary_key=True
    )
    name = db.Column(
        db.String(6400),
        index=False,
        unique=True,
        nullable=False
    )
    email = db.Column(
        db.String(8000),
        unique=True,
        nullable=False
    )
    password = db.Column(
        db.String(800000),
        unique=True,
        nullable=False
    )
    Created_At = db.Column(
        db.DateTime,
        index=False,
        unique=False,
        default=datetime.now()
    )
    Updated_At = db.Column(
        db.DateTime,
        index=False,
        unique=False,
        nullable=True
    )

class Harvest(db.Model):

    __tablename__ = 'harvest'
    id = db.Column(
        db.Integer,
        primary_key=True
    )
    full_harvest = db.Column(
        db.Integer,
        index=False,
        unique=False,
        nullable=False
    )
    can_be_utilized = db.Column(
        db.Integer,
        unique=False,
        nullable=False
    )
    Created_At = db.Column(
        db.DateTime,
        index=False,
        unique=False,
        default=datetime.now()
    )
    Updated_At = db.Column(
        db.DateTime,
        index=False,
        unique=False,
        nullable=True
    )

class Quality(db.Model):

    __tablename__ = 'quality'
    id = db.Column(
        db.Integer,
        primary_key=True
    )
    
    quality_is = db.Column(
        db.String(1000),
        nullable=False
    )
    created_AT = db.Column(
        db.DateTime,
        index=False,
        unique=False,
        default=datetime.now(),
        nullable=False
    )
    Updated_At = db.Column(
        db.DateTime,
        index=False,
        unique=False,
        nullable=True
    )
    

class Deseases(db.Model):

    __tablename__ = 'deseases'
    id = db.Column(
        db.Integer,
        primary_key=True
    )
    
    desease = db.Column(
        db.String(1000),
        nullable=False
    )
    created_AT = db.Column(
        db.DateTime,
        index=False,
        unique=False,
        default=datetime.now(),
        nullable=False
    )
    Updated_At = db.Column(
        db.DateTime,
        index=False,
        unique=False,
        nullable=True
    )


class Clasification(db.Model):

    __tablename__ = 'desease_clasification'
    id = db.Column(
        db.Integer,
        primary_key=True
    )
    
    desease  = db.Column(
        db.String(1000),
        nullable=False
    )
    created_AT = db.Column(
        db.DateTime,
        index=False,
        unique=False,
        default=datetime.now(),
        nullable=False
    )
    Updated_At = db.Column(
        db.DateTime,
        index=False,
        unique=False,
        nullable=True
    )


    