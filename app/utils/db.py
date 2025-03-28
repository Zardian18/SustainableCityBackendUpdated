from app import db
from app.models import User

# Utility function to commit changes to the database
def commit_to_db():
    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        raise Exception(f"Error committing to database: {str(e)}")

# Function to check if a user exists by username
def user_exists(username):
    user = User.query.filter_by(username=username).first()
    return user is not None
