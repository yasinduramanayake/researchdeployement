from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()


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

