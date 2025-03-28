from flask import Blueprint, request, jsonify
from app.services.auth_service import register_user, login_user

# Create a Blueprint for authentication-related routes
auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/register', methods=['POST'])
def register():
    return register_user()

@auth_bp.route('/login', methods=['POST'])
def login():
    return login_user()
