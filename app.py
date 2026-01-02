import os
from flask import Flask, render_template, jsonify, request, redirect, url_for, session
from dotenv import load_dotenv
from authlib.integrations.flask_client import OAuth
from flask_login import LoginManager, login_user, logout_user, current_user, login_required
# [CRITICAL FIX] This library handles the HTTPS headers from Render
from werkzeug.middleware.proxy_fix import ProxyFix 
import razorpay
from models import db, User, Payment, Consultation
from medical_model import medical_diagnosis

# --- LOAD ENV ---
load_dotenv()

# --- SECURITY BYPASS FOR LOCALHOST ---
# Only enable insecure transport if we are NOT on Render (Development mode)
if os.environ.get('FLASK_ENV') == 'development':
    os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

app = Flask(__name__)

# --- [CRITICAL FIX] PROXY FIX FOR RENDER DEPLOYMENT ---
# This tells Flask: "Trust that Render is handling HTTPS for us"
# Without this, Flask generates 'http://' URLs, causing Google Login to fail.
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# --- CONFIGURATION ---
app.secret_key = os.environ.get("SECRET_KEY", "super_secret_key_123")
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Cookie Settings
app.config['SESSION_COOKIE_NAME'] = 'mediai_session'
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = False # Keep False to support both Local and Render

# Database (Auto-detects Postgres on Render)
database_url = os.environ.get("DATABASE_URL")
if database_url and database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)
app.config['SQLALCHEMY_DATABASE_URI'] = database_url or 'sqlite:///mediai.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

# Login Manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# OAuth
oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id=os.environ.get("GOOGLE_CLIENT_ID"),
    client_secret=os.environ.get("GOOGLE_CLIENT_SECRET"),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

# Razorpay
razorpay_client = razorpay.Client(
    auth=(os.environ.get("RAZORPAY_KEY_ID"), os.environ.get("RAZORPAY_KEY_SECRET"))
)

# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html', user=current_user)

@app.route('/auth/login')
def login():
    # Automatically generates the correct URL (https:// on Render, http:// on Local)
    redirect_uri = url_for('auth_callback', _external=True)
    return google.authorize_redirect(redirect_uri)

@app.route('/auth/callback')
def auth_callback():
    try:
        # Exchange code for token
        token = google.authorize_access_token()
        
        # Get User Info (Full URL fixed)
        user_info = google.get('https://www.googleapis.com/oauth2/v3/userinfo').json()
        
        google_id = user_info.get('sub') or user_info.get('id')
        email = user_info.get('email')
        name = user_info.get('name')
        picture = user_info.get('picture')

        # Database Logic
        user = User.query.filter_by(google_id=google_id).first()
        
        if not user:
            # Check for admin email from .env
            is_admin = (email == os.environ.get("ADMIN_EMAIL"))
            user = User(google_id=google_id, email=email, name=name, picture=picture, is_admin=is_admin)
            db.session.add(user)
            db.session.commit()
        else:
            user.picture = picture
            db.session.commit()

        login_user(user)
        return redirect('/')
    
    except Exception as e:
        print(f"AUTH ERROR: {e}")
        return f"<h3>Authentication Error</h3><p>{e}</p><a href='/'>Back to Home</a>"

@app.route('/auth/logout')
@login_required
def logout():
    logout_user()
    session.clear()
    return redirect('/')

# --- MEDICAL API ---

@app.route('/api/diagnose', methods=['POST'])
def diagnose():
    if not current_user.is_authenticated:
        return jsonify({'error': 'Please login first'}), 401
    
    data = request.json
    symptoms = data.get('symptoms', '').strip()
    
    if not symptoms:
        return jsonify({'error': 'No symptoms provided'}), 400

    result = medical_diagnosis(symptoms)
    
    # Save History
    cons = Consultation(
        user_id=current_user.id,
        symptoms=symptoms,
        diagnosis=result.get('primary_diagnosis'),
        confidence=result.get('confidence')
    )
    db.session.add(cons)
    db.session.commit()
    
    return jsonify(result)

# --- PAYMENT API ---

@app.route('/api/create_subscription', methods=['POST'])
@login_required
def create_subscription():
    amount = 49900 # ₹499.00
    try:
        order_data = {'amount': amount, 'currency': 'INR', 'payment_capture': 1}
        order = razorpay_client.order.create(data=order_data)
        
        payment = Payment(user_id=current_user.id, amount=amount, order_id=order['id'])
        db.session.add(payment)
        db.session.commit()
        
        return jsonify({
            'order_id': order['id'],
            'amount': amount,
            'key': os.environ.get("RAZORPAY_KEY_ID")
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/verify_payment', methods=['POST'])
@login_required
def verify_payment():
    data = request.json
    try:
        razorpay_client.utility.verify_payment_signature({
            'razorpay_order_id': data.get('razorpay_order_id'),
            'razorpay_payment_id': data.get('razorpay_payment_id'),
            'razorpay_signature': data.get('razorpay_signature')
        })
        
        payment = Payment.query.filter_by(order_id=data.get('razorpay_order_id')).first()
        if payment:
            payment.payment_id = data.get('razorpay_payment_id')
            payment.status = 'paid'
            
            # Upgrade User
            current_user.is_pro = True
            db.session.commit()
            
            return jsonify({'success': True})
        return jsonify({'error': 'Order not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# --- ADMIN API ---

@app.route('/admin')
@login_required
def admin_panel():
    if not current_user.is_admin: return "Forbidden", 403
    return render_template('admin.html')

@app.route('/api/admin_stats')
@login_required
def admin_stats():
    if not current_user.is_admin: return jsonify({'error': 'Forbidden'}), 403
    
    users = User.query.all()
    payments = Payment.query.filter_by(status='paid').all()
    
    return jsonify({
        'users': [{'name': u.name, 'email': u.email, 'is_pro': u.is_pro} for u in users],
        'revenue': sum(p.amount for p in payments) / 100
    })

@app.route('/api/config')
def get_config():
    return jsonify({
        'user': {
            'name': current_user.name, 
            'picture': current_user.picture,
            'is_pro': current_user.is_pro,
            'is_admin': current_user.is_admin
        } if current_user.is_authenticated else None
    })

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    # Debug=True is fine for dev, but ProxyFix handles the Prod headers
    app.run(debug=True, port=5000)