from flask import Flask, request, jsonify, send_from_directory, session, redirect, url_for
from medical_model import medical_diagnosis, load_medical_model, train_medical_model
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
from dotenv import load_dotenv
from authlib.integrations.flask_client import OAuth
import razorpay
from models import db, User, Payment
from flask_login import LoginManager, login_user, logout_user, current_user, login_required

# --- CRITICAL: ALLOW HTTP FOR LOCALHOST ---
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)

# --- CONFIGURATION ---
# Default to 127.0.0.1
FRONTEND_URL = os.environ.get('FRONTEND_URL', 'http://127.0.0.1:5000/')
OAUTH_REDIRECT_URI = os.environ.get('OAUTH_REDIRECT_URI', 'http://127.0.0.1:5000/auth/callback')

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'super-secret-key')
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = False

# Database
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///' + os.path.join(basedir, 'ai_medical.db'))
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    try:
        return User.query.get(int(user_id))
    except Exception:
        return None

# OAuth
oauth = OAuth(app)
oauth.register(
    name='google',
    client_id=os.environ.get('GOOGLE_CLIENT_ID'),
    client_secret=os.environ.get('GOOGLE_CLIENT_SECRET'),
    access_token_url='https://oauth2.googleapis.com/token',
    access_token_params=None,
    authorize_url='https://accounts.google.com/o/oauth2/v2/auth',
    authorize_params=None,
    api_base_url='https://www.googleapis.com/oauth2/v2/',
    userinfo_endpoint='https://www.googleapis.com/oauth2/v2/userinfo',
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'},
)

# Razorpay
RAZORPAY_KEY_ID = os.environ.get('RAZORPAY_KEY_ID')
RAZORPAY_KEY_SECRET = os.environ.get('RAZORPAY_KEY_SECRET')
razorpay_client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET)) if RAZORPAY_KEY_ID else None

# Global variables
model_data = None

def initialize_model():
    global model_data
    try:
        model_data = load_medical_model()
        return True
    except Exception:
        try:
            train_medical_model()
            model_data = load_medical_model()
            return True
        except Exception:
            return False

# --- EMAIL FUNCTION ---
def send_payment_email(user_email, user_name, payment_id, amount):
    sender_email = os.environ.get('MAIL_USERNAME')
    sender_password = os.environ.get('MAIL_PASSWORD')
    
    if not sender_email or not sender_password:
        return

    subject = "MediAI Subscription Confirmed"
    body = f"""
    <h2>Hello {user_name},</h2>
    <p>Thank you for subscribing to MediAI Pro!</p>
    <p><strong>Payment ID:</strong> {payment_id}</p>
    <p><strong>Amount:</strong> ₹{amount/100}</p>
    <br><p>Regards,<br>The MediAI Team</p>
    """

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = user_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'html'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
    except Exception as e:
        logger.error(f"Failed to send email: {e}")

# --- ROUTES ---

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

# --- REPLACE THESE TWO FUNCTIONS IN app.py ---

@app.route('/auth/login')
def login():
    # Force the exact URI (bypass .env for now to be safe)
    # Ensure this matches Google Console EXACTLY
    forced_uri = 'http://127.0.0.1:5000/auth/callback'
    return oauth.google.authorize_redirect(forced_uri)

@app.route('/auth/callback')
def auth_callback():
    try:
        # Fetch token with the SAME hardcoded URI
        token = oauth.google.fetch_access_token(
            authorization_response=request.url,
            redirect_uri='http://127.0.0.1:5000/auth/callback'
        )
        
        # Get User Info
        resp = oauth.google.get('userinfo', token=token)
        user_info = resp.json()

        google_id = user_info.get('id')
        email = user_info.get('email')
        name = user_info.get('name')
        picture = user_info.get('picture')

        with app.app_context():
            user = User.query.filter_by(google_id=google_id).first()
            if not user:
                user = User(google_id=google_id, email=email, name=name, picture=picture)
                db.session.add(user)
                db.session.commit()
            else:
                user.email = email
                user.name = name
                user.picture = picture
                db.session.commit()
            
            user = User.query.filter_by(google_id=google_id).first()
            login_user(user)

        return redirect(FRONTEND_URL)
        
    except Exception as e:
        logger.error(f"LOGIN FAILED: {e}")
        # If this fails, do NOT refresh. Go back to homepage.
        return f"<h3>Login Error: {e}</h3><p>Do not refresh this page. <a href='/'>Return Home</a> and try again.</p>"
@app.route('/auth/logout')
def logout():
    logout_user()
    return redirect('/')

# --- PAYMENTS ---

@app.route('/payments/create-order', methods=['POST'])
@login_required
def create_order():
    if not razorpay_client: return jsonify({'error': 'Payment gateway error'}), 503
    
    amount = 499 * 100 
    try:
        order = razorpay_client.order.create({'amount': amount, 'currency': 'INR', 'payment_capture': 1})
        
        payment = Payment(
            user_id=current_user.id,
            order_id=order['id'],
            amount=amount,
            currency='INR',
            status='created'
        )
        db.session.add(payment)
        db.session.commit()
        
        return jsonify({'order': order, 'key': RAZORPAY_KEY_ID})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/payments/confirm', methods=['POST'])
@login_required
def confirm_payment():
    data = request.json
    razorpay_order_id = data.get('razorpay_order_id')
    razorpay_payment_id = data.get('razorpay_payment_id')
    
    payment = Payment.query.filter_by(order_id=razorpay_order_id).first()
    if payment:
        payment.payment_id = razorpay_payment_id
        payment.status = 'paid'
        db.session.commit()
        send_payment_email(current_user.email, current_user.name, razorpay_payment_id, payment.amount)
        return jsonify({'success': True})
    return jsonify({'error': 'Order not found'}), 404

# --- API ---

@app.route('/api/diagnose', methods=['POST'])
def diagnose():
    data = request.json
    symptoms = data.get('symptoms', '').strip()
    if not symptoms: return jsonify({'error': 'No symptoms'}), 400
    
    try:
        result = medical_diagnosis(symptoms)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/config')
def get_config():
    return jsonify({
        'user': current_user.to_dict() if current_user.is_authenticated else None
    })

if __name__ == '__main__':
    if initialize_model():
        with app.app_context():
            db.create_all()
        # Explicit 0.0.0.0 host
        app.run(debug=True, host='0.0.0.0', port=5000)