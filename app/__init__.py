from flask import Flask
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, firestore
from app.routes import initialize_routes

# Initialize the Flask app
app = Flask(__name__)
CORS(app)

# Load configuration
app.config.from_pyfile('config.py')

# Initialize Firestore
cred = credentials.Certificate(app.config['FIRESTORE_CREDENTIALS'])
firebase_admin.initialize_app(cred)
db = firestore.client()
app.firestore_db = db

# Initialize routes after app is fully initialized
initialize_routes(app)

# Add a simple route for testing
@app.route('/')
def hello():
    return "Hello, Flask with Firestore!"