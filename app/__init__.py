# app/__init__.py

from flask import Flask
from flask_cors import CORS
from app.extensions import db  # Import db from the extensions file
from app.routes import initialize_routes

# Initialize the Flask app
app = Flask(__name__)

# Enable Cross-Origin Resource Sharing (CORS)
CORS(app)

# SQLAlchemy Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:mysql09876@localhost/ase_schema'  # MySQL URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'jwtsecretkey12312323CASDdd23dddDFASF'


# Initialize the db object with the app
db.init_app(app)

# Initialize routes
initialize_routes(app)
