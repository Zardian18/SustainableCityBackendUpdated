from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_mysqldb import MySQL
import bcrypt
import jwt  # For JSON Web Token (JWT) operations
import datetime  # For token expiration
from functools import wraps  # For token validation decorator

# app = Flask(__name__)
# CORS(app)  # Allow cross-origin requests from the React frontend

# # MySQL Configuration
# app.config['MYSQL_HOST'] = 'localhost'
# app.config['MYSQL_USER'] = 'root'  # Replace with your MySQL username
# app.config['MYSQL_PASSWORD'] = 'password'  # Replace with your MySQL password
# app.config['MYSQL_DB'] = 'ase_schema'  # Replace with your database name
# app.config['SECRET_KEY'] = 'thisisajwtsecretkey'  # Replace with a strong secret key for JWT

# mysql = MySQL(app)

# # Helper function to generate JWT
# def create_token(username):
#     payload = {
#         'username': username,
#         'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=2)  # Token expires in 2 hours
#     }
#     token = jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')
#     return token

# # Decorator to protect routes with JWT
# def token_required(f):
#     @wraps(f)
#     def decorated(*args, **kwargs):
#         token = request.headers.get('x-access-token')
#         if not token:
#             return jsonify({'error': 'Token is missing!'}), 403

#         try:
#             jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
#         except jwt.ExpiredSignatureError:
#             return jsonify({'error': 'Token has expired!'}), 401
#         except jwt.InvalidTokenError:
#             return jsonify({'error': 'Invalid token!'}), 401

#         return f(*args, **kwargs)
#     return decorated

# # Register route
# @app.route('/api/auth/register', methods=['POST'])
# def register():
#     data = request.json
#     username = data['username']
#     password = data['password']
#     supervisor_name = data['supervisor_name']
#     mode_of_transport = data['mode_of_transport']
#     security_question = data['security_question']

#     try:
#         # Create users table if it doesn't exist
#         cursor = mysql.connection.cursor()
#         cursor.execute("""
#             CREATE TABLE IF NOT EXISTS users (
#                 user_id INT AUTO_INCREMENT PRIMARY KEY,
#                 username VARCHAR(255) NOT NULL UNIQUE,
#                 password VARCHAR(255) NOT NULL,
#                 supervisor_name VARCHAR(255),
#                 mode_of_transport VARCHAR(255),
#                 security_question VARCHAR(255)
#             )
#         """)
#         mysql.connection.commit()

#         # Hash the password
#         hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

#         # Insert user into database
#         cursor.execute("""
#             INSERT INTO users (username, password, supervisor_name, mode_of_transport, security_question)
#             VALUES (%s, %s, %s, %s, %s)
#         """, (username, hashed_password, supervisor_name, mode_of_transport, security_question))
#         mysql.connection.commit()

#         return jsonify({"message": "User registered successfully!"}), 201
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# # Login route
# @app.route('/api/auth/login', methods=['POST'])
# def login():
#     data = request.json
#     username = data['username']
#     password = data['password']

#     try:
#         # Fetch user from the database
#         cursor = mysql.connection.cursor()
#         cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
#         user = cursor.fetchone()

#         if user and bcrypt.checkpw(password.encode('utf-8'), user[2].encode('utf-8')):  # Password matches
#             token = create_token(username)
#             return jsonify({"message": "Login successful!", "token": token}), 200
#         else:
#             return jsonify({"error": "Invalid username or password"}), 401
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# # Protected dashboard route
# @app.route('/api/dashboard', methods=['GET'])
# @token_required
# def dashboard():
#     return jsonify({"message": "Welcome to the Dashboard!"}), 200

# # Logout route (optional for token-based systems)
# @app.route('/api/logout', methods=['POST'])
# def logout():
#     # In a token-based system, logout is handled client-side by deleting the token.
#     return jsonify({"message": "Logout successful!"}), 200

# # Start the Flask server
# if __name__ == '__main__':
#     app.run(debug=True)
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from flask_mysqldb import MySQL
# import bcrypt
# import jwt
# import datetime
# from functools import wraps

app = Flask(__name__)
CORS(app)

# MySQL Configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'  # Replace with your MySQL username
app.config['MYSQL_PASSWORD'] = 'mysql09876'  # Replace with your MySQL password
app.config['MYSQL_DB'] = 'ase_schema'  # Replace with your database name
app.config['SECRET_KEY'] = 'jwtsecretkey12312323CASDdd23dddDFASF'  # Replace with a strong secret key for JWT

mysql = MySQL(app)

# Helper function to generate JWT
def create_token(username, user_type):
    payload = {
        'username': username,
        'user_type': user_type,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=2)
    }
    token = jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')
    return token

# Decorator to protect routes with JWT
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('x-access-token')
        if not token:
            return jsonify({'error': 'Token is missing!'}), 403
        try:
            decoded_token = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            request.user_type = decoded_token['user_type']
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token has expired!'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token!'}), 401
        return f(*args, **kwargs)
    return decorated

# Register route
@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.json
    username = data['username']
    password = data['password']
    user_type = data['user_type']
    security_question = data['security_question']
    security_answer = data['security_answer']

    try:
        cursor = mysql.connection.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(255) NOT NULL UNIQUE,
                password VARCHAR(255) NOT NULL,
                user_type ENUM('supervisor', 'manager', 'normal') NOT NULL,
                security_question VARCHAR(255),
                security_answer VARCHAR(255)
            )
        """)
        mysql.connection.commit()

        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        cursor.execute("""
            INSERT INTO users (username, password, user_type, security_question, security_answer)
            VALUES (%s, %s, %s, %s, %s)
        """, (username, hashed_password, user_type, security_question, security_answer))
        mysql.connection.commit()

        return jsonify({"message": "User registered successfully!"}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Login route
@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.json
    username = data['username']
    password = data['password']

    try:
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()

        if user and bcrypt.checkpw(password.encode('utf-8'), user[2].encode('utf-8')):
            token = create_token(username, user[3])
            return jsonify({"message": "Login successful!", "token": token, "user_type": user[3]}), 200
        else:
            return jsonify({"error": "Invalid username or password"}), 401
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Protected dashboard route
@app.route('/api/dashboard', methods=['GET'])
@token_required
def dashboard():
    return jsonify({"message": "Welcome to the Dashboard!", "user_type": request.user_type}), 200

# Logout route
@app.route('/api/logout', methods=['POST'])
def logout():
    return jsonify({"message": "Logout successful!"}), 200

if __name__ == '__main__':
    app.run(debug=True)
