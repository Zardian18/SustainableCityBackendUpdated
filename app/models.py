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

class Notification(db.Model):
    __tablename__ = 'notification'

    notification_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    manager_name = db.Column(db.String(255), nullable=False)
    status = db.Column(db.String(50), nullable=False, default='pending')  # e.g., 'pending', 'approved', 'rejected'
    timestamp = db.Column(db.DateTime, nullable=False)  # Changed to db.DateTime
    mode_of_transport = db.Column(db.String(50), nullable=False, default='bus')
    route_id = db.Column(db.String(255), nullable=True)  # Optional, can be set later
    bus_id = db.Column(db.Integer, nullable=False)  # To identify the bus
    start_lat = db.Column(db.Float, nullable=False)  # Starting latitude
    start_lng = db.Column(db.Float, nullable=False)  # Starting longitude
    end_lat = db.Column(db.Float, nullable=False)   # Destination latitude
    end_lng = db.Column(db.Float, nullable=False)   # Destination longitude

    def __init__(self, manager_name, status, timestamp, bus_id, start_lat, start_lng, end_lat, end_lng, mode_of_transport='bus', route_id=None):
        self.manager_name = manager_name
        self.status = status
        self.timestamp = timestamp  # Expects a datetime object now
        self.mode_of_transport = mode_of_transport
        self.route_id = route_id
        self.bus_id = bus_id
        self.start_lat = start_lat
        self.start_lng = start_lng
        self.end_lat = end_lat
        self.end_lng = end_lng

    def to_dict(self):
        return {
            'notification_id': self.notification_id,
            'manager_name': self.manager_name,
            'status': self.status,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,  # Convert to ISO string for JSON
            'mode_of_transport': self.mode_of_transport,
            'route_id': self.route_id,
            'bus_id': self.bus_id,
            'start_lat': self.start_lat,
            'start_lng': self.start_lng,
            'end_lat': self.end_lat,
            'end_lng': self.end_lng
    }