from app import db

class Users(db.Model):
    __tablename__ = 'users'

    user_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    supervisor_name = db.Column(db.String(255), nullable=False)
    username = db.Column(db.String(255), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(255))
    mode = db.Column(db.String(255))
    security_question = db.Column(db.String(255))
    security_question_answer = db.Column(db.String(255))

    def __init__(self, supervisor_name, username, password, role, mode, security_question, security_question_answer):
        self.supervisor_name = supervisor_name
        self.username = username
        self.password = password
        self.role = role
        self.mode = mode
        self.security_question = security_question
        self.security_question_answer = security_question_answer
