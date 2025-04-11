from flask import jsonify, request
from dotenv import load_dotenv
from flask import current_app
import bcrypt
import jwt
from datetime import datetime, timedelta
import os

load_dotenv()

# Helper function to generate JWT
def create_token(username, role, mode):
    if role is None:
        role = "user"

    print(f"Encoding token with role: {role} (Type: {type(role)})")
    payload = {
        'username': username,
        'role': str(role),
        'mode': str(mode),
        'exp': datetime.utcnow() + timedelta(hours=2)
    }
    token = jwt.encode(payload, current_app.config['SECRET_KEY'], algorithm='HS256')
    return token

# Register user function
def register_user():
    # Access Firestore db via current_app
    db = current_app.firestore_db

    data = request.json
    supervisor_name = data['supervisor_name']
    username = data['username']
    password = data['password']
    role = data['role']
    mode = data['mode']
    security_question = data['security_question']
    security_question_answer = data['security_answer']

    try:
        # Check if the username already exists
        doc_ref = db.collection('users').document(username)
        if doc_ref.get().exists:
            return jsonify({"error": "Username already exists"}), 409

        # Hash the password
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

        # Create user data
        user_data = {
            'supervisor_name': supervisor_name,
            'username': username,
            'password': hashed_password,
            'role': role,
            'mode': mode,
            'security_question': security_question,
            'security_question_answer': security_question_answer
        }

        # Store in users/<username>
        doc_ref.set(user_data)
        user_data['user_id'] = username

        return jsonify({"message": "User registered successfully!", "user_id": username}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Login user function
def login_user():
    # Access Firestore db via current_app
    db = current_app.firestore_db

    data = request.json
    username = data['username']
    password = data['password']

    try:
        doc_ref = db.collection('users').document(username)
        doc = doc_ref.get()
        if not doc.exists:
            return jsonify({"error": "Username not found"}), 404

        user = doc.to_dict()

        if not bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
            return jsonify({"error": "Invalid password"}), 401

        token = create_token(username, user['role'], user['mode'])
        return jsonify({
            "message": "Login successful!",
            "token": token,
            "username": user['username'],
            "role": user['role'],
            "user_id": username
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500