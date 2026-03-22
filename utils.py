import cv2
import numpy as np
from datetime import datetime, timedelta
import json
import os

class FaceRecognitionUtils:
    @staticmethod
    def enhance_image(img):
        """Enhance image quality for better face detection"""
        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L-channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # Merge channels and convert back
        lab = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return enhanced
    
    @staticmethod
    def validate_face_quality(face_img):
        """Validate if face image meets quality standards"""
        if face_img is None or face_img.size == 0:
            return False, "No face detected"
        
        # Check image size
        height, width = face_img.shape[:2]
        if height < 50 or width < 50:
            return False, "Face too small"
        
        # Check brightness
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        if brightness < 50 or brightness > 200:
            return False, "Poor lighting conditions"
        
        # Check blur using Laplacian variance
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score < 100:
            return False, "Image too blurry"
        
        return True, "Good quality"
    
    @staticmethod
    def generate_attendance_report(attendance_data, date_range=None):
        """Generate comprehensive attendance report"""
        report = {
            "generated_at": datetime.now().isoformat(),
            "total_records": len(attendance_data),
            "unique_users": len(set(record.get('name', '') for record in attendance_data)),
            "date_range": date_range,
            "statistics": {},
            "records": attendance_data
        }
        
        if attendance_data:
            # Calculate statistics
            user_counts = {}
            time_stats = []
            
            for record in attendance_data:
                name = record.get('name', '')
                time = record.get('time', '')
                
                if name:
                    user_counts[name] = user_counts.get(name, 0) + 1
                
                if time:
                    try:
                        hour = int(time.split(':')[0])
                        time_stats.append(hour)
                    except:
                        pass
            
            report["statistics"] = {
                "most_frequent": max(user_counts.items(), key=lambda x: x[1]) if user_counts else None,
                "peak_hour": max(set(time_stats), key=time_stats.count) if time_stats else None,
                "average_attendance_per_user": len(attendance_data) / len(user_counts) if user_counts else 0
            }
        
        return report
    
    @staticmethod
    def backup_data(backup_dir="backups"):
        """Create backup of important data"""
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_info = {
            "timestamp": timestamp,
            "files_backed_up": []
        }
        
        # Backup attendance records if they exist
        attendance_files = []
        if os.path.exists('Attendance'):
            attendance_files = [f for f in os.listdir('Attendance') if f.endswith('.csv')]
        
        # Backup model if it exists
        model_path = 'static/face_recognition_model.pkl'
        if os.path.exists(model_path):
            backup_info["files_backed_up"].append(model_path)
        
        # Save backup info
        backup_file = os.path.join(backup_dir, f"backup_info_{timestamp}.json")
        with open(backup_file, 'w') as f:
            json.dump(backup_info, f, indent=2)
        
        return backup_file

class SecurityFeatures:
    @staticmethod
    def detect_spoofing(face_img):
        """Basic anti-spoofing detection"""
        if face_img is None:
            return False, "No face provided"
        
        # Convert to grayscale
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Check for eye reflection (simplified)
        eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        eyes = eyes_cascade.detectMultiScale(gray, 1.1, 5)
        
        # Basic liveness check - require at least one eye
        if len(eyes) == 0:
            return False, "No eyes detected - possible spoof"
        
        return True, "Liveness check passed"
    
    @staticmethod
    def log_security_event(event_type, details):
        """Log security-related events"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "details": details
        }
        
        log_file = "security_logs.json"
        logs = []
        
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            except:
                pass
        
        logs.append(log_entry)
        
        # Keep only last 1000 entries
        if len(logs) > 1000:
            logs = logs[-1000:]
        
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
