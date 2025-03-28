# __init__.py for the services module
# This file initializes the services module and allows you to import functions easily

from .auth_service import register_user, login_user
from .prediction_service import make_prediction