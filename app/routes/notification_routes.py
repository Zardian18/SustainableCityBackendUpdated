from flask import Blueprint, request, jsonify
from datetime import datetime
from app import db
from app.models import Notification

notification_bp = Blueprint('notification', __name__)

@notification_bp.route('/reroute-request', methods=['POST'])
def reroute():
    try:
        db.create_all()
        data = request.get_json()
        print("Received data:", data)  # Debugging
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        bus_id = data.get('busId')
        bike_id = data.get('bike_id')
        start = data.get('start', {})
        end = data.get('end', {})
        start_lat = start.get('lat')
        start_lng = start.get('lng')
        end_lat = end.get('lat')
        end_lng = end.get('lng')
        manager_name = data.get('manager_name')
        mode_of_transport = data.get('mode_of_transport', 'bus')
        station_name = data.get('station_name')
        event_name = data.get('event_name')  # Extract event_name

        if not all([manager_name, start_lat, start_lng, end_lat, end_lng]):
            return jsonify({'error': 'Missing required fields: manager_name and coordinates are required'}), 400
        
        if mode_of_transport == 'bus' and bus_id is None:
            return jsonify({'error': 'bus_id is required for bus mode'}), 400
        if mode_of_transport == 'bike' and bike_id is None:
            return jsonify({'error': 'bike_id is required for bike mode'}), 400
        if mode_of_transport == 'event' and event_name is None:  # Require event_name for events
            return jsonify({'error': 'event_name is required for event mode'}), 400

        timestamp = datetime.utcnow()
        notification = Notification(
            manager_name=manager_name,
            status='pending',
            timestamp=timestamp,
            mode_of_transport=mode_of_transport,
            bus_id=bus_id if mode_of_transport == 'bus' else None,
            bike_id=bike_id if mode_of_transport == 'bike' else None,
            start_lat=start_lat,
            start_lng=start_lng,
            end_lat=end_lat,
            end_lng=end_lng,
            station_name=station_name if mode_of_transport == 'bike' else None,
            event_name=event_name if mode_of_transport == 'event' else None  # Set for event mode
        )

        db.session.add(notification)
        db.session.commit()

        return jsonify({
            'message': 'Reroute request created successfully',
            'notification': notification.to_dict()
        }), 201

    except Exception as e:
        db.session.rollback()
        print("Exception:", str(e))
        return jsonify({'error': str(e)}), 500

@notification_bp.route('/fetch-notification', methods=['GET'])
def fetch_notification():
    try:
        db.create_all()
        bus_id = request.args.get('bus_id', type=int)
        bike_id = request.args.get('bike_id', type=int)
        status = request.args.get('status')
        manager_name = request.args.get('manager_name')

        query = Notification.query
        if bus_id:
            query = query.filter_by(bus_id=bus_id)
        if bike_id:
            query = query.filter_by(bike_id=bike_id)
        if status:
            query = query.filter_by(status=status)
        if manager_name:
            query = query.filter_by(manager_name=manager_name)

        notifications = query.all()

        return jsonify({
            'notifications': [n.to_dict() for n in notifications]
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@notification_bp.route('/update-notification', methods=['POST'])
def update_notification():
    try:
        db.create_all()
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        notification_id = data.get('notification_id')
        new_status = data.get('status')

        if not all([notification_id, new_status]):
            return jsonify({'error': 'Missing required fields'}), 400

        if new_status not in ['approved', 'rejected']:
            return jsonify({'error': 'Invalid status value'}), 400

        notification = Notification.query.filter_by(notification_id=notification_id).first()
        if not notification:
            return jsonify({'error': 'Notification not found'}), 404

        notification.status = new_status
        db.session.commit()

        return jsonify({
            'message': 'Notification status updated successfully',
            'notification': notification.to_dict()
        }), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500