from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

db = SQLAlchemy()

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(128), nullable=False)

class Detection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False)
    age = db.Column(db.String(20))
    gender = db.Column(db.String(10))
    source = db.Column(db.String(20)) # 'image', 'video', 'live'
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
