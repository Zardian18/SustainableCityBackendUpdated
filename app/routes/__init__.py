from app.extensions import db 
from app.routes.auth_routes import auth_bp
from app.routes.prediction_routes import prediction_bp
from app.routes.dashboard_route import dashboard_bp
from app.routes.notification_routes import notification_bp

def initialize_routes(app):
    # Register the blueprints
    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(prediction_bp, url_prefix='/api/predict')
    app.register_blueprint(dashboard_bp, url_prefix='/api/dashboard')
    app.register_blueprint(notification_bp, url_prefix='/api/notification')
