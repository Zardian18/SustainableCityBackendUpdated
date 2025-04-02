from flask import jsonify, request
from dotenv import load_dotenv
from app import db
from app.models import Users
import bcrypt
import jwt
from datetime import datetime, timedelta
import os

load_dotenv()

# Helper function to generate JWT
def create_token(username, role, mode):
    if role is None:
        role = "user"  # Set a default value

    print(f"Encoding token with role: {role} (Type: {type(role)})")  # Debugging

    payload = {
        'username': username,
        'role': str(role),  # Ensure it's a string,
        'mode': str(mode),
        'exp': datetime.utcnow() + timedelta(hours=2)
    }

    token = jwt.encode(payload, 'jwtsecretkey12312323CASDdd23dddDFASF', algorithm='HS256')
    return token

# Register user function
def register_user():
    data = request.json
    supervisor_name = data['supervisor_name']
    username = data['username']
    password = data['password']
    role = data['role']
    mode = data['mode']
    security_question = data['security_question']
    security_question_answer = data['security_answer']

    try:
        # Ensure the Users table exists
        db.create_all()

        # Check if the username already exists
        existing_user = Users.query.filter_by(username=username).first()
        if existing_user:
            return jsonify({"error": "Username already exists"}), 409

        # Hash the password
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        # Create a new user instance
        new_user = Users(
            supervisor_name=supervisor_name,
            username=username,
            password=hashed_password.decode('utf-8'),  # Store as a string
            role=role,
            mode=mode,
            security_question=security_question,
            security_question_answer=security_question_answer
        )

        # Add to the session and commit to save
        db.session.add(new_user)
        db.session.commit()

        return jsonify({"message": "User registered successfully!"}), 201

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

# Login user function
def login_user():
    data = request.json
    username = data['username']
    password = data['password']

    try:
        user = Users.query.filter_by(username=username).first()
        if user and bcrypt.checkpw(password.encode('utf-8'), user.password.encode('utf-8')):  # Password matches
            token = create_token(username, user.role, user.mode)
            return jsonify({"message": "Login successful!", "token": token, "username": user.username, "role":user.mode}), 200
        else:
            return jsonify({"error": "Invalid username or password"}), 401
    except Exception as e:
        return jsonify({"error": str(e)}), 500
        
    
    
    
