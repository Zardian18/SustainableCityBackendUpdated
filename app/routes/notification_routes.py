from flask import Blueprint, request, jsonify, current_app
from datetime import datetime

notification_bp = Blueprint('notification', __name__)

@notification_bp.route('/reroute-request', methods=['POST'])
def reroute():
    try:
        # Access Firestore db via current_app
        db = current_app.firestore_db

        data = request.get_json()
        print("Received data:", data)
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
        event_name = data.get('event_name')

        if not all([manager_name, start_lat, start_lng, end_lat, end_lng]):
            return jsonify({'error': 'Missing required fields: manager_name and coordinates are required'}), 400
        
        if mode_of_transport == 'bus' and bus_id is None:
            return jsonify({'error': 'bus_id is required for bus mode'}), 400
        if mode_of_transport == 'bike' and bike_id is None:
            return jsonify({'error': 'bike_id is required for bike mode'}), 400

        # Verify that the manager_name corresponds to an existing user
        user_ref = db.collection('users').document(manager_name)
        if not user_ref.get().exists:
            return jsonify({'error': 'User not found'}), 404

        timestamp = datetime.utcnow().isoformat()
        notification_data = {
            'manager_name': manager_name,
            'status': 'pending',
            'timestamp': timestamp,
            'mode_of_transport': mode_of_transport,
            'bus_id': bus_id if mode_of_transport == 'bus' else None,
            'bike_id': bike_id if mode_of_transport == 'bike' else None,
            'start_lat': start_lat,
            'start_lng': start_lng,
            'end_lat': end_lat,
            'end_lng': end_lng,
            'station_name': station_name if mode_of_transport == 'bike' else None,
            'event_name': event_name if mode_of_transport == 'pedestrian' else None,
            'route_id': None
        }

        # Store in notifications collection with auto-generated ID
        doc_ref = db.collection('notifications').add(notification_data)[1]
        notification_data['notification_id'] = doc_ref.id

        return jsonify({
            'message': 'Reroute request created successfully',
            'notification': notification_data
        }), 201

    except Exception as e:
        print("Exception:", str(e))
        return jsonify({'error': str(e)}), 500

@notification_bp.route('/fetch-notification', methods=['GET'])
def fetch_notification():
    try:
        # Access Firestore db via current_app
        db = current_app.firestore_db

        bus_id = request.args.get('bus_id', type=int)
        bike_id = request.args.get('bike_id', type=int)
        status = request.args.get('status')
        manager_name = request.args.get('manager_name')

        query = db.collection('notifications')
        if manager_name:
            query = query.where('manager_name', '==', manager_name)
        if bus_id:
            query = query.where('bus_id', '==', bus_id)
        if bike_id:
            query = query.where('bike_id', '==', bike_id)
        if status:
            query = query.where('status', '==', status)

        docs = query.stream()
        notifications = [{**doc.to_dict(), 'notification_id': doc.id} for doc in docs]

        return jsonify({
            'notifications': notifications
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@notification_bp.route('/update-notification', methods=['POST'])
def update_notification():
    try:
        # Access Firestore db via current_app
        db = current_app.firestore_db

        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        notification_id = data.get('notification_id')
        new_status = data.get('status')
        manager_name = data.get('manager_name')

        if not all([notification_id, new_status]):
            return jsonify({'error': 'Missing required fields: notification_id and status'}), 400

        if new_status not in ['approved', 'rejected']:
            return jsonify({'error': 'Invalid status value'}), 400

        doc_ref = db.collection('notifications').document(notification_id)
        doc = doc_ref.get()
        if not doc.exists:
            return jsonify({'error': 'Notification not found'}), 404

        # Verify manager_name matches
        if manager_name and doc.to_dict().get('manager_name') != manager_name:
            return jsonify({'error': 'Notification does not belong to this user'}), 403

        doc_ref.update({'status': new_status})
        updated_doc = doc_ref.get().to_dict()
        updated_doc['notification_id'] = notification_id

        return jsonify({
            'message': 'Notification status updated successfully',
            'notification': updated_doc
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500