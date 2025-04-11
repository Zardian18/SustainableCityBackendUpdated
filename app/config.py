import os
from dotenv import load_dotenv

load_dotenv()


# Firestore Configuration
FIRESTORE_CREDENTIALS = os.getenv('FIRESTORE_CREDENTIALS', 'serviceAccount.json')

# Flask Configuration
SECRET_KEY = os.getenv('SECRET_KEY', 'jwtsecretkey12312323CASDdd23dddDFASF')