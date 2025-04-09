from flask import Blueprint, request, jsonify  # Assuming these exist
from datetime import datetime
from app import db  # Assuming db is initialized in app/__init__.py
from app.models import Notification  # Assuming Notification model is defined in models.py

notification_bp = Blueprint('notification', __name__)

@notification_bp.route('/reroute-request', methods=['POST'])
def reroute():
    try:
        # Ensure the Notification table exists
        db.create_all()

        # Get JSON data from the request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Extract required fields
        bus_id = data.get('busId')
        start = data.get('start', {})
        end = data.get('end', {})
        start_lat = start.get('lat')
        start_lng = start.get('lng')
        end_lat = end.get('lat')
        end_lng = end.get('lng')
        manager_name = data.get('manager_name')

        # Validate required fields
        if not all([bus_id, start_lat, start_lng, end_lat, end_lng, manager_name]):
            return jsonify({'error': 'Missing required fields'}), 400

        # Create a new notification entry
        timestamp = datetime.utcnow()  # Use datetime object
        notification = Notification(
            manager_name=manager_name,
            status='pending',
            timestamp=timestamp,
            bus_id=bus_id,
            start_lat=start_lat,
            start_lng=start_lng,
            end_lat=end_lat,
            end_lng=end_lng
        )

        # Add to the database and commit
        db.session.add(notification)
        db.session.commit()

        # Return success response
        return jsonify({
            'message': 'Reroute request created successfully',
            'notification': notification.to_dict()
        }), 201

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@notification_bp.route('/fetch-notification', methods=['GET'])
def fetch_notification():
    try:
        db.create_all()  # Ensure table exists

        # Filter by query params
        bus_id = request.args.get('bus_id', type=int)
        status = request.args.get('status')
        manager_name = request.args.get('manager_name')  # Add manager_name filter

        query = Notification.query
        if bus_id:
            query = query.filter_by(bus_id=bus_id)
        if status:
            query = query.filter_by(status=status)
        if manager_name:
            query = query.filter_by(manager_name=manager_name)  # Filter by manager

        notifications = query.all()

        return jsonify({
            'notifications': [n.to_dict() for n in notifications]
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@notification_bp.route('/update-notification', methods=['POST'])
def update_notification():
    try:
        db.create_all()  # Ensure table exists
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        notification_id = data.get('notification_id')
        new_status = data.get('status')

        if not all([notification_id, new_status]):
            return jsonify({'error': 'Missing required fields'}), 400

        if new_status not in ['approved', 'rejected']:
            return jsonify({'error': 'Invalid status value'}), 400

        # Find the notification
        notification = Notification.query.filter_by(notification_id=notification_id).first()
        if not notification:
            return jsonify({'error': 'Notification not found'}), 404

        # Update status
        notification.status = new_status
        db.session.commit()

        return jsonify({
            'message': 'Notification status updated successfully',
            'notification': notification.to_dict()
        }), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500