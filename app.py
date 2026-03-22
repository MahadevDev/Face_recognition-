import cv2
import os
from flask import Flask, request, render_template, Response, session, redirect, url_for, jsonify
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
from pymongo import MongoClient
from api import api_bp
from utils import FaceRecognitionUtils, SecurityFeatures

# Defining Flask App
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'change-this-secret')

# Register API blueprint
app.register_blueprint(api_bp)

nimgs = 10

# Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

# MongoDB configuration
def get_mongodb_connection():
    try:
        # Using direct connection instead of SRV to avoid DNS resolution issues
        MONGO_URI = os.getenv('MONGO_URI', 'mongodb+srv://mahadevags573_db_user:hytPJrCfkTa48YXZ@cluster2.gal6a9g.mongodb.net/')
        
        # Replace the SRV URI with a direct connection string
        if 'mongodb+srv://' in MONGO_URI:
            MONGO_URI = MONGO_URI.replace('mongodb+srv://', 'mongodb://')
            
        # Add connection timeout and retry parameters
        client = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=5000,  # 5 second timeout
            connectTimeoutMS=10000,         # 10 second connection timeout
            socketTimeoutMS=45000,          # 45 second socket timeout
            retryWrites=True,
            w='majority'
        )
        
        # Test the connection
        client.admin.command('ping')
        return client
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return None

try:
    # Initialize MongoDB connection
    mongo_client = get_mongodb_connection()
    if mongo_client is None:
        print("Warning: Could not connect to MongoDB. Running in offline mode.")
        # Create a dummy client to prevent errors
        class DummyCollection:
            def __getattr__(self, name):
                return lambda *args, **kwargs: []
        
        class DummyDB:
            def __getitem__(self, name):
                return DummyCollection()
        
        class DummyClient:
            def __getitem__(self, name):
                return DummyDB()
            
            def __enter__(self):
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
        
        mongo_client = DummyClient()
    
    MONGO_DB_NAME = os.getenv('MONGO_DB_NAME', 'attendance_db')
    MONGO_COLLECTION_NAME = os.getenv('MONGO_COLLECTION_NAME', 'attendance')
    
    mongo_db = mongo_client[MONGO_DB_NAME]
    attendance_col = mongo_db[MONGO_COLLECTION_NAME]
    
except Exception as e:
    print(f"Error initializing MongoDB: {e}")
    # Fallback to in-memory storage if MongoDB is not available
    class MemoryStorage:
        def __init__(self):
            self.data = {}
            
        def find(self, query=None, **kwargs):
            if query is None:
                query = {}
            if '_id' in query:
                query.pop('_id')
            return [doc for doc in self.data.values() if all(doc.get(k) == v for k, v in query.items())]
            
        def find_one(self, query=None, **kwargs):
            results = self.find(query, **kwargs)
            return results[0] if results else None
            
        def insert_one(self, document):
            doc_id = str(len(self.data) + 1)
            document['_id'] = doc_id
            self.data[doc_id] = document
            return type('Result', (), {'inserted_id': doc_id})
            
        def delete_one(self, filter):
            to_delete = [k for k, v in self.data.items() if all(v.get(key) == value for key, value in filter.items())]
            if to_delete:
                del self.data[to_delete[0]]
                return type('Result', (), {'deleted_count': 1})
            return type('Result', (), {'deleted_count': 0})
    
    # Create in-memory collections
    mongo_db = type('DB', (), {'__getitem__': lambda *args: MemoryStorage()})()
    attendance_col = mongo_db['attendance']


# Initializing VideoCapture object to access WebCam
try:
    cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
    face_detector = cv2.CascadeClassifier(cascade_path)
except Exception:
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# If these directories don't exist, create them
# (Attendance CSV directory no longer required with MongoDB, left for backward compatibility)
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
# Deprecated: CSV attendance file creation (kept to avoid breaking existing setups)
try:
    if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
            f.write('Name,Roll,Time')
except Exception:
    pass


# get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))


# extract the face from an image
def extract_faces(img):
    try:
        # Enhance image for better face detection
        enhanced_img = FaceRecognitionUtils.enhance_image(img)
        gray = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points if isinstance(face_points, (list, tuple, np.ndarray)) else []
    except Exception:
        return []


# Identify face using ML model
def identify_face(facearray):
    model_path = 'static/face_recognition_model.pkl'
    if not os.path.exists(model_path):
        raise FileNotFoundError('Trained model not found')
    model = joblib.load(model_path)
    return model.predict(facearray)


# A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    user_faces_dir = 'static/faces'
    if not os.path.isdir(user_faces_dir):
        return
    userlist = os.listdir(user_faces_dir)
    for user in userlist:
        user_dir = os.path.join(user_faces_dir, user)
        if not os.path.isdir(user_dir):
            continue
        for imgname in os.listdir(user_dir):
            img_path = os.path.join(user_dir, imgname)
            img = cv2.imread(img_path)
            if img is None:
                continue
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    if len(faces) == 0:
        # No data to train on; remove stale model if exists
        model_path = 'static/face_recognition_model.pkl'
        if os.path.exists(model_path):
            os.remove(model_path)
        return
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')


# Extract today's attendance from storage
def extract_attendance():
    try:
        # Check if using in-memory storage
        if hasattr(attendance_col, 'data'):
            records = []
            for record_id, record in attendance_col.data.items():
                if record.get('date_key') == datetoday:
                    records.append(record)
            # Sort by time
            records.sort(key=lambda x: x.get('time', ''))
        else:  # Using MongoDB
            cursor = attendance_col.find({ 'date_key': datetoday }, { '_id': 0 }).sort('time', 1)
            records = list(cursor)
            
        if not records:
            return [], [], [], 0
            
        names = [rec.get('name', '') for rec in records]
        rolls = [str(rec.get('roll', '')) for rec in records]
        times = [rec.get('time', '') for rec in records]
        l = len(records)
        return names, rolls, times, l
    except Exception as e:
        print(f"Error in extract_attendance: {str(e)}")
        return [], [], [], 0


# Add Attendance of a specific user to MongoDB
def add_attendance(name):
    if '_' not in name:
        return
    username, userid = name.split('_', 1)
    current_time = datetime.now().strftime("%H:%M:%S")
    current_date = datetime.now().strftime("%m_%d_%y")
    current_date_display = datetime.now().strftime("%d-%B-%Y")
    
    # Create attendance record
    attendance_record = {
        'name': username,
        'roll': userid,
        'time': current_time,
        'date_key': current_date,
        'date_display': current_date_display
    }
    
    try:
        # For in-memory storage
        if hasattr(attendance_col, 'data'):
            import uuid
            record_id = str(uuid.uuid4())
            attendance_col.data[record_id] = attendance_record
            print(f"Added to in-memory storage: {attendance_record}")  # Debug log
        # For MongoDB
        elif hasattr(attendance_col, 'insert_one'):
            attendance_col.insert_one(attendance_record)
        return True
    except Exception as e:
        print(f"Error adding attendance: {str(e)}")
        return False


## A function to get names and rol numbers of all users
def getallusers():
    faces_dir = 'static/faces'
    if not os.path.isdir(faces_dir):
        return [], [], [], 0
    userlist = [d for d in os.listdir(faces_dir) if os.path.isdir(os.path.join(faces_dir, d))]
    names = []
    rolls = []
    for folder_name in userlist:
        if '_' in folder_name:
            name, roll = folder_name.split('_', 1)
            names.append(name)
            rolls.append(roll)
        else:
            # Skip invalid folder names
            continue
    l = len(names)
    return userlist, names, rolls, l


## A function to delete a user folder 
def deletefolder(duser):
    if not os.path.isdir(duser):
        return
    pics = os.listdir(duser)
    for i in pics:
        try:
            os.remove(os.path.join(duser, i))
        except Exception:
            pass
    try:
        os.rmdir(duser)
    except Exception:
        pass




################## ROUTING FUNCTIONS #########################

# Our main page
@app.route('/')
def home():
    if not session.get('authenticated'):
        return redirect(url_for('login'))
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

# Admin dashboard route
@app.route('/admin')
def admin():
    if not session.get('authenticated'):
        return redirect(url_for('login'))
    names, rolls, times, l = extract_attendance()
    return render_template('admin.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)


## List users page
@app.route('/listusers')
def listusers():
    if not session.get('authenticated'):
        return redirect(url_for('login', next=url_for('listusers')))
    userlist, names, rolls, l = getallusers()
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2)


## Delete functionality
@app.route('/deleteuser', methods=['GET', 'POST'])
def deleteuser():
    if request.method == 'POST':
        duser = request.form.get('userid')
    else:
        duser = request.args.get('user')
        
    if duser:
        deletefolder('static/faces/'+duser)

        ## if all the face are deleted, delete the trained file...
        if os.path.exists('static/faces/') and os.path.isdir('static/faces/') and os.listdir('static/faces/')==[]:
            if os.path.exists('static/face_recognition_model.pkl'):
                os.remove('static/face_recognition_model.pkl')
        
        try:
            train_model()
        except:
            pass

    return redirect(url_for('listusers'))
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2)


# Our main Face Recognition functionality. 
# This function will run when we click on Take Attendance Button.
@app.route('/start', methods=['GET'])
def start():
    if not session.get('authenticated'):
        return redirect(url_for('login', next=url_for('start')))
    names, rolls, times, l = extract_attendance()

    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='There is no trained model in the static folder. Please add a new face to continue.')

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='Unable to access the webcam.')
    logged_this_session = set()
    ret = True
    while ret:
        ret, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
            cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
            try:
                face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
                identified_person = identify_face(face.reshape(1, -1))[0]
                label = identified_person if isinstance(identified_person, str) else str(identified_person)
                if '_' in label and label not in logged_this_session:
                    if add_attendance(label):
                        logged_this_session.add(label)
                        # Store the success message in session
                        session['attendance_success'] = f"Attendance recorded for {label.replace('_', ' ')}"
                cv2.putText(frame, f'{label}', (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            except Exception:
                continue
        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    
    # Check if we have a success message to show
    success_message = session.pop('attendance_success', None)
    if success_message:
        return redirect(url_for('home', success='true', message=success_message))
        
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)


# Attendance history page with date filter
@app.route('/history', methods=['GET'])
def history():
    if not session.get('authenticated'):
        return redirect(url_for('login', next=url_for('history', date=request.args.get('date', datetoday))))
    
    date_key = request.args.get('date', datetoday)
    try:
        # Try to fetch records from MongoDB if available, otherwise use in-memory storage
        if hasattr(attendance_col, 'data'):  # Using in-memory storage
            records = []
            for record_id, record in attendance_col.data.items():
                if record.get('date_key') == date_key:
                    records.append(record)
            # Sort by time
            records.sort(key=lambda x: x.get('time', ''))
        else:  # Using MongoDB
            cursor = attendance_col.find({ 'date_key': date_key }, { '_id': 0 }).sort('time', 1)
            records = list(cursor)
        
        # Extract data from records
        names = [r.get('name', '') for r in records]
        rolls = [str(r.get('roll', '')) for r in records]
        times = [r.get('time', '') for r in records]
        l = len(records)
        
        # Handle display date with error checking
        if records and 'date_display' in records[0]:
            display_date = records[0]['date_display']
        else:
            try:
                # Try to parse the date_key
                display_date = datetime.strptime(date_key, '%m_%d_%y').strftime('%d-%B-%Y')
            except ValueError:
                # Fallback to current date if parsing fails
                display_date = datetime.now().strftime('%d-%B-%Y')
        
        return render_template('history.html', 
                             names=names, 
                             rolls=rolls, 
                             times=times, 
                             l=l, 
                             totalreg=totalreg(), 
                             datetoday2=display_date, 
                             date_key=date_key)
    except Exception as e:
        # Log the error for debugging
        print(f"Error in history route: {str(e)}")
        # Return empty results on error
        return render_template('history.html', 
                             names=[], 
                             rolls=[], 
                             times=[], 
                             l=0, 
                             totalreg=totalreg(), 
                             datetoday2=datetime.now().strftime('%d-%B-%Y'), 
                             date_key=date_key)


# CSV export for any date (defaults to today)
@app.route('/delete_attendance', methods=['POST'])
def delete_attendance():
    try:
        print("[DEBUG] Delete attendance endpoint hit")
        
        # Check authentication
        if not session.get('authenticated'):
            print("[AUTH] User not authenticated")
            return jsonify({"status": "error", "message": "Authentication required"}), 401
        
        # Get form data with error handling
        try:
            form_data = request.get_json() or request.form
            print(f"[DEBUG] Raw request data: {request.data}")
            print(f"[DEBUG] Parsed form data: {dict(form_data)}")
            
            date_key = form_data.get('date')
            roll = form_data.get('roll')
            time = form_data.get('time')
            
            print(f"[DEBUG] Extracted values - Date: {date_key}, Roll: {roll}, Time: {time}")
            
            # Validate required fields
            if not all([date_key, roll, time]):
                error_msg = f"Missing required parameters. Date: {date_key}, Roll: {roll}, Time: {time}"
                print(f"[ERROR] {error_msg}")
                return jsonify({"status": "error", "message": error_msg}), 400
            
            # Delete the attendance record
            delete_query = {
                'date_key': date_key,
                'roll': roll,
                'time': time
            }
            print(f"[DEBUG] Delete query: {delete_query}")
            
            # Check if we're using in-memory storage
            if hasattr(attendance_col, 'delete_one'):
                print("[DEBUG] Using MongoDB delete operation")
                result = attendance_col.delete_one(delete_query)
                print(f"[DEBUG] MongoDB delete result: {result.raw_result if hasattr(result, 'raw_result') else result}")
                deleted_count = result.deleted_count if hasattr(result, 'deleted_count') else 0
            elif hasattr(attendance_col, 'data'):
                print("[DEBUG] Using in-memory storage")
                deleted_count = 0
                items = list(attendance_col.data.values())
                print(f"[DEBUG] Current items in memory: {items}")
                for item_id, item in list(attendance_col.data.items()):
                    print(f"[DEBUG] Checking item {item_id}: {item}")
                    if all(str(item.get(k)) == str(v) for k, v in delete_query.items()):
                        print(f"[DEBUG] Match found, deleting item {item_id}")
                        del attendance_col.data[item_id]
                        deleted_count += 1
            else:
                print("[ERROR] No valid storage method found")
                return jsonify({"status": "error", "message": "No valid storage method available"}), 500
            
            print(f"[DEBUG] Deleted {deleted_count} record(s)")
            
            if deleted_count == 0:
                error_msg = f"No record found matching the criteria: {delete_query}"
                print(f"[WARNING] {error_msg}")
                return jsonify({"status": "error", "message": "Record not found"}), 404
            
            print(f"[SUCCESS] Deleted {deleted_count} record(s) successfully")
            return jsonify({
                "status": "success", 
                "message": "Record deleted successfully",
                "redirect": url_for('history', date=date_key, _external=True)
            })
            
        except Exception as e:
            error_msg = f"Error processing request: {str(e)}"
            print(f"[ERROR] {error_msg}")
            import traceback
            traceback.print_exc()
            return jsonify({"status": "error", "message": str(e)}), 500
            
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f"[CRITICAL] {error_msg}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": "An unexpected error occurred"}), 500


@app.route('/export', methods=['GET'])
def export_attendance():
    if not session.get('authenticated'):
        return redirect(url_for('login', next=url_for('export_attendance', date=request.args.get('date', datetoday))))
    date_key = request.args.get('date', datetoday)
    cursor = attendance_col.find({ 'date_key': date_key }, { '_id': 0 }).sort('time', 1)
    rows = list(cursor)
    csv_lines = ['Name,Roll,Time']
    for r in rows:
        csv_lines.append(f"{r.get('name','')},{r.get('roll','')},{r.get('time','')}")
    csv_data = '\n'.join(csv_lines)
    filename = f'Attendance-{date_key}.csv'
    return Response(
        csv_data,
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename={filename}'}
    )


# Take attendance landing page (with button to start camera)
@app.route('/take', methods=['GET'])
def take():
    if not session.get('authenticated'):
        return redirect(url_for('login', next=url_for('take')))
    return render_template('take.html')


# Clear attendance records for a given date
@app.route('/clear', methods=['POST'])
def clear_attendance():
    if not session.get('authenticated'):
        return redirect(url_for('login'))
    date_key = request.form.get('date', datetoday)
    attendance_col.delete_many({ 'date_key': date_key })
    return render_template('history.html', names=[], rolls=[], times=[], l=0, totalreg=totalreg(), datetoday2=datetime.strptime(date_key, '%m_%d_%y').strftime('%d-%B-%Y'), date_key=date_key, mess=f'Cleared attendance for {date_key}.')


# Inspection page to review stored data and times (optionally filter by roll)
@app.route('/inspect', methods=['GET'])
def inspect():
    if not session.get('authenticated'):
        next_url = request.full_path if request.query_string else url_for('inspect')
        return redirect(url_for('login', next=next_url))
    date_key = request.args.get('date', datetoday)
    roll = request.args.get('roll')
    query = { 'date_key': date_key }
    if roll:
        query['roll'] = str(roll)
    cursor = attendance_col.find(query, { '_id': 0 }).sort('time', 1)
    records = list(cursor)
    return render_template('inspect.html', records=records, date_key=date_key, roll=roll or '')


# Simple session-based authentication for admin actions
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        expected_user = os.getenv('ADMIN_USER', 'admin')
        expected_pass = os.getenv('ADMIN_PASS', 'admin123')
        if username == expected_user and password == expected_pass:
            session['authenticated'] = True
            next_url = request.args.get('next') or url_for('home')
            return redirect(next_url)
        return render_template('login.html', error='Invalid username or password')
    # For GET request, just show the login form
    return render_template('login.html')
    return render_template('login.html', error=None)


@app.route('/logout', methods=['POST', 'GET'])
def logout():
    session.clear()
    return redirect(url_for('home'))


# A function to add a new user.
# This function will run when we add a new user.
@app.route('/add', methods=['GET', 'POST'])
def add():
    if not session.get('authenticated'):
        return redirect(url_for('login', next=url_for('add')))
    if request.method == 'GET':
        names, rolls, times, l = extract_attendance()
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='Submit the form to add a new user.')
    newusername = request.form.get('newusername', '').strip()
    newuserid = request.form.get('newuserid', '').strip()
    if not newusername or not newuserid:
        names, rolls, times, l = extract_attendance()
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='Username and User ID are required.')
    userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    i, j = 0, 0
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        names, rolls, times, l = extract_attendance()
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='Unable to access the webcam.')
    while 1:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            if j % 5 == 0:
                name = newusername+'_'+str(i)+'.jpg'
                cv2.imwrite(userimagefolder+'/'+name, frame[y:y+h, x:x+w])
                i += 1
            j += 1
        if j == nimgs*5:
            break
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    try:
        train_model()
    except Exception:
        pass
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)



# Our main function which runs the Flask App
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
