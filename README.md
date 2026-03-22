# Face Recognition Based Attendance System  

Do visit my blog for better explanations: https://machinelearningprojects.net/face-recognition-based-attendance-system/

![Face Recognition Based Attendance System](ss.png)

## Deploy on Render.com

This application is configured for easy deployment on Render.com.

### Quick Deploy:
1. Push your code to GitHub
2. Go to [Render.com](https://render.com)
3. Click "New +" → "Web Service"
4. Connect your GitHub repository
5. Render will auto-detect the `render.yaml` configuration

### Environment Variables Required:
- `MONGO_URI` - Your MongoDB connection string
- `MONGO_DB_NAME` - Database name (default: attendance_db)
- `MONGO_COLLECTION_NAME` - Collection name (default: attendance)
- `SECRET_KEY` - Flask secret key (auto-generated)
- `ADMIN_USER` - Admin username (default: admin)
- `ADMIN_PASS` - Admin password (auto-generated)

### Features:
- Face recognition attendance system
- Real-time camera processing
- MongoDB data storage
- User management
- Attendance history and export

### Dependencies:
- Flask
- OpenCV
- scikit-learn
- MongoDB
- NumPy, Pandas
