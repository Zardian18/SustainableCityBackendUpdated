from app import db

class Users(db.Model):
    __tablename__ = 'users'

    user_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(255), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(255))
    security_question = db.Column(db.String(255))
    security_question_answer = db.Column(db.String(255))

    def __init__(self, username, password, role, security_question, security_question_answer):
        self.username = username
        self.password = password
        self.role = role
        self.security_question = security_question
        self.security_question_answer = security_question_answer
