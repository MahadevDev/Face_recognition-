import cv2
import os
from flask import Flask, request, render_template, Response, session, redirect, url_for
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
from pymongo import MongoClient

# Defining Flask App
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'change-this-secret')

nimgs = 10

# Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

# MongoDB configuration
MONGO_URI = os.getenv('MONGO_URI', 'mongodb+srv://mahadevags573_db_user:hytPJrCfkTa48YXZ@cluster2.gal6a9g.mongodb.net/')
MONGO_DB_NAME = os.getenv('MONGO_DB_NAME', 'attendance_db')
MONGO_COLLECTION_NAME = os.getenv('MONGO_COLLECTION_NAME', 'attendance')

mongo_client = MongoClient(MONGO_URI)
mongo_db = mongo_client[MONGO_DB_NAME]
attendance_col = mongo_db[MONGO_COLLECTION_NAME]


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
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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


# Extract today's attendance from MongoDB
def extract_attendance():
    cursor = attendance_col.find({ 'date_key': datetoday }, { '_id': 0 }).sort('time', 1)
    records = list(cursor)
    if not records:
        return [], [], [], 0
    names = [rec.get('name', '') for rec in records]
    rolls = [str(rec.get('roll', '')) for rec in records]
    times = [rec.get('time', '') for rec in records]
    l = len(records)
    return names, rolls, times, l


# Add Attendance of a specific user to MongoDB
def add_attendance(name):
    if '_' not in name:
        return
    username, userid = name.split('_', 1)
    current_time = datetime.now().strftime("%H:%M:%S")

    # Avoid duplicate roll for the same day
    existing = attendance_col.find_one({ 'date_key': datetoday, 'roll': str(userid) })
    if existing:
        return
    doc = {
        'date_key': datetoday,
        'date_display': datetoday2,
        'name': username,
        'roll': str(userid),
        'time': current_time
    }
    try:
        attendance_col.insert_one(doc)
    except Exception:
        # Ignore write failure silently (no offline queue)
        pass


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


## List users page
@app.route('/listusers')
def listusers():
    if not session.get('authenticated'):
        return redirect(url_for('login', next=url_for('listusers')))
    userlist, names, rolls, l = getallusers()
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2)


## Delete functionality
@app.route('/deleteuser', methods=['GET'])
def deleteuser():
    duser = request.args.get('user')
    deletefolder('static/faces/'+duser)

    ## if all the face are deleted, delete the trained file...
    if os.listdir('static/faces/')==[]:
        os.remove('static/face_recognition_model.pkl')
    
    try:
        train_model()
    except:
        pass

    userlist, names, rolls, l = getallusers()
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
                    add_attendance(label)
                    logged_this_session.add(label)
                cv2.putText(frame, f'{label}', (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            except Exception:
                continue
        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)


# Attendance history page with date filter
@app.route('/history', methods=['GET'])
def history():
    if not session.get('authenticated'):
        return redirect(url_for('login', next=url_for('history', date=request.args.get('date', datetoday))))
    date_key = request.args.get('date', datetoday)
    cursor = attendance_col.find({ 'date_key': date_key }, { '_id': 0 }).sort('time', 1)
    records = list(cursor)
    names = [r.get('name', '') for r in records]
    rolls = [str(r.get('roll', '')) for r in records]
    times = [r.get('time', '') for r in records]
    l = len(records)
    display_date = records[0]['date_display'] if records else datetime.strptime(date_key, '%m_%d_%y').strftime('%d-%B-%Y')
    return render_template('history.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=display_date, date_key=date_key)


# CSV export for any date (defaults to today)
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
        expected_user = os.getenv('ADMIN_USER', 'Mahadeva')
        expected_pass = os.getenv('ADMIN_PASS', 'Madev123')
        if username == expected_user and password == expected_pass:
            session['authenticated'] = True
            next_url = request.args.get('next') or url_for('add')
            return redirect(next_url)
        return render_template('login.html', error='Invalid credentials')
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
    app.run(debug=True)
