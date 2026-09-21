# 🏥 MediAI Pro - Advanced Medical Consultancy Platform

# Live Demo https://ai-medical-consultancy-1.onrender.com

**MediAI Pro** is a Flask-based healthcare application that uses Machine Learning to diagnose symptoms and provides a complete ecosystem for patients and doctors. It features secure Google OAuth login, Razorpay payment integration for Pro subscriptions, an Admin Dashboard, and a modern Dark Mode UI.

---

## 🚀 Features

* **🤖 AI Symptom Checker:** NLP-based diagnosis using `scikit-learn`.
* **🔐 Secure Auth:** Google OAuth 2.0 Login (Gmail).
* **💳 Payments:** Razorpay integration for Pro Subscriptions (₹499).
* **👑 Pro Membership:** Unlocks detailed reports, history tracking, and priority support.
* **🛡️ Admin Dashboard:** Track revenue, total users, and pro subscribers.
* **🎨 Modern UI:** Fully responsive Dark Mode design with Glassmorphism effects.
* **📧 Email Notifications:** Automated email receipts via SMTP.

---

## 🛠️ Tech Stack

* **Backend:** Python, Flask, SQLAlchemy (PostgreSQL/SQLite)
* **Frontend:** HTML5, Tailwind CSS, JavaScript (Vanilla)
* **AI/ML:** Scikit-learn, Pandas, NLTK
* **Payments:** Razorpay API
* **Deployment:** Ready for Render / Heroku

---

## ⚙️ Installation (Local)

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/mediai-pro.git](https://github.com/your-username/mediai-pro.git)
cd mediai-pro

```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

### 4. Setup `.env` File

Create a `.env` file in the root directory and fill in your keys:

```bash
# --- GENERAL ---
SECRET_KEY=your_random_secret_string
FLASK_ENV=development

# --- DATABASE (Auto-switches to SQLite locally) ---
# DATABASE_URL=sqlite:///mediai.db

# --- GOOGLE OAUTH ---
# Create credentials at: console.cloud.google.com
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret

# --- RAZORPAY ---
# Create keys at: dashboard.razorpay.com
RAZORPAY_KEY_ID=your_razorpay_key_id
RAZORPAY_KEY_SECRET=your_razorpay_key_secret

# --- ADMIN ACCESS ---
ADMIN_EMAIL=your_email@gmail.com

```

### 5. Run the App

```bash
python app.py

```

> Access the app at: **http://127.0.0.1:5000**

---

## ☁️ Deployment (Render.com)

1. **Push to GitHub:** Upload your code to a GitHub repository.
2. **Create Web Service:** Go to [Render](https://render.com) > New > Web Service.
3. **Connect Repo:** Select your repository.
4. **Settings:**
* **Runtime:** Python 3
* **Build Command:** `pip install -r requirements.txt`
* **Start Command:** `gunicorn app:app`


5. **Environment Variables:** Add all keys from your `.env` file to Render's "Environment" tab.
6. **Google Console Update:**
* Go to Google Cloud Console > Credentials.
* Add your Render URL to **Authorized Redirect URIs**:
* `https://your-app-name.onrender.com/auth/callback`





---

## 📂 Project Structure

```text
/mediai-pro
  ├── app.py                 # Main application logic
  ├── models.py              # Database models (User, Payment, Consultation)
  ├── medical_model.py       # ML Model training & prediction
  ├── requirements.txt       # Project dependencies
  ├── .env                   # Environment variables (Sensitive keys)
  ├── static/
  │   ├── style.css          # Custom CSS (Dark Theme)
  │   └── script.js          # Frontend Logic (Auth, Payments)
  └── templates/
      ├── index.html         # Main Dashboard
      └── admin.html         # Admin Panel

```

---

## 🚑 Disclaimer

**MediAI Pro** is an assistive tool for educational and informational purposes only. It does **not** replace professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or qualified health provider with any questions you may have regarding a medical condition.

---

### ❤️ Credits

Developed by **Krishna Patil**.

```

```

---

## Security

Please refer to [SECURITY.md](SECURITY.md) for vulnerability reporting guidelines.

## Contributing

Contributions are welcome! Please review [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and development process.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Developed and maintained by **[Krishna Patil](https://github.com/kriss2012)**.
