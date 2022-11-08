from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()


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

    def __repr__(self):
        return '<User {}>'.format(self.username)