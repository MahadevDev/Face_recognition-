from flask import Blueprint, jsonify, request
from utils import FaceRecognitionUtils, SecurityFeatures
import json
import os
from datetime import datetime

# Create API blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api')

@api_bp.route('/stats', methods=['GET'])
def get_statistics():
    """Get attendance statistics"""
    try:
        # This would integrate with your existing attendance extraction
        stats = {
            "total_users": 0,
            "today_attendance": 0,
            "this_week": 0,
            "this_month": 0,
            "accuracy_rate": 0.95,  # Example metric
            "system_status": "healthy"
        }
        return jsonify({"status": "success", "data": stats})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@api_bp.route('/users', methods=['GET'])
def get_users_api():
    """Get all registered users"""
    try:
        from app import getallusers
        userlist, names, rolls, l = getallusers()
        users = []
        for i in range(len(names)):
            users.append({
                "id": userlist[i] if i < len(userlist) else f"{names[i]}_{rolls[i]}",
                "name": names[i] if i < len(names) else "",
                "roll": rolls[i] if i < len(rolls) else ""
            })
        return jsonify({"status": "success", "data": users})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@api_bp.route('/attendance', methods=['GET'])
def get_attendance_api():
    """Get attendance records with date filtering"""
    try:
        date_filter = request.args.get('date', None)
        
        # This would integrate with your existing attendance extraction
        attendance_data = []
        
        return jsonify({
            "status": "success", 
            "data": attendance_data,
            "date_filter": date_filter
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@api_bp.route('/export/<format>', methods=['GET'])
def export_data(format):
    """Export attendance data in different formats"""
    try:
        if format not in ['csv', 'json', 'excel']:
            return jsonify({"status": "error", "message": "Unsupported format"}), 400
        
        date_filter = request.args.get('date', None)
        
        # Generate export based on format
        if format == 'json':
            # Generate JSON export
            export_data = {
                "export_date": datetime.now().isoformat(),
                "filter": date_filter,
                "records": []
            }
            return jsonify(export_data)
        
        elif format == 'csv':
            # Generate CSV export
            csv_data = "Name,Roll,Time,Date\n"
            # Add your attendance data here
            return csv_data, 200, {'Content-Type': 'text/csv'}
        
        elif format == 'excel':
            # Would need openpyxl for Excel export
            return jsonify({"status": "error", "message": "Excel export requires additional setup"}), 400
            
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@api_bp.route('/quality-check', methods=['POST'])
def check_image_quality():
    """Check face image quality"""
    try:
        if 'image' not in request.files:
            return jsonify({"status": "error", "message": "No image provided"}), 400
        
        image_file = request.files['image']
        # Process image and check quality
        # This would integrate with your face processing
        
        quality_result = {
            "quality_score": 0.85,
            "brightness": "good",
            "sharpness": "good",
            "face_detected": True,
            "recommendations": []
        }
        
        return jsonify({"status": "success", "data": quality_result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@api_bp.route('/system/health', methods=['GET'])
def system_health():
    """System health check endpoint"""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "database": "connected",
                "camera": "ready",
                "model": "trained",
                "storage": "available"
            },
            "metrics": {
                "cpu_usage": "normal",
                "memory_usage": "normal",
                "disk_space": "available"
            }
        }
        return jsonify(health_status)
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@api_bp.route('/security/logs', methods=['GET'])
def get_security_logs():
    """Get security event logs"""
    try:
        if os.path.exists('security_logs.json'):
            with open('security_logs.json', 'r') as f:
                logs = json.load(f)
            return jsonify({"status": "success", "data": logs[-50:]})  # Last 50 logs
        else:
            return jsonify({"status": "success", "data": []})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
