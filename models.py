from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    google_id = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    name = db.Column(db.String(100))
    picture = db.Column(db.String(200))
    is_pro = db.Column(db.Boolean, default=False) # True if paid
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    consultations = db.relationship('Consultation', backref='user', lazy=True)

class Payment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    amount = db.Column(db.Integer, nullable=False)
    order_id = db.Column(db.String(100), unique=True)
    payment_id = db.Column(db.String(100))
    status = db.Column(db.String(20), default='created')
    date = db.Column(db.DateTime, default=datetime.utcnow)

class Consultation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    symptoms = db.Column(db.Text, nullable=False)
    diagnosis = db.Column(db.String(200))
    confidence = db.Column(db.Float)
    date = db.Column(db.DateTime, default=datetime.utcnow)