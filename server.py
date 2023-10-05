import datetime
import os
import sqlite3
import threading

import cv2
import numpy as np
from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/attendance', methods=['POST'])
def success():
    if request.method == 'POST':
        try:
            f = request.files['file']
            f.save('uploads/upload.jpg')

            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read('recognizer/model.yml')
            img = detect()
            if img is None:
                return {
                    "message": "No face found"
                }, 404
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            id_, conf = recognizer.predict(gray)
            if conf > 70:
                return {
                    "message": "User not found"
                }, 404

            conn = get_db_connection()
            result = conn.execute(
                'SELECT * FROM students WHERE id = ?', (id_,)).fetchone()
            if result is None:
                return {
                    "message": "User not found"
                }, 404

            current_datetime = datetime.datetime.now()
            sqlite_date_format = current_datetime.strftime('%Y-%m-%d')
            student_id = result['student_id']
            result = conn.execute(
                'SELECT * FROM attendance WHERE student_id = ? AND date = ?', (student_id, sqlite_date_format)).fetchone()
            if result is not None:
                return {
                    "message": "student attendance is marked"
                }, 201

            conn.execute('INSERT INTO attendance (student_id, date) VALUES (?, ?)',
                         (student_id, sqlite_date_format))
            conn.commit()
            conn.close()
            return {
                "message": "Attendance marked successfully"
            }, 200
        except:
            return {
                "message": "Internal server error"
            }, 500

    else:
        return {
            "message": "Invalid request"
        }, 400


@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        try:
            images = request.files.getlist('images')
            rollno = request.form.get('rollno')
            name = request.form.get('name')
            email = request.form.get('email')

            if images is None:
                return {
                    "message": "No images found"
                }, 400
            if len(images) > 10:
                return {
                    "message": "Maximum 10 images are allowed"
                }, 400
            if rollno is None or name is None or email is None:
                return {
                    "message": "Invalid data"
                }, 400
            conn = get_db_connection()
            result = conn.execute(
                'SELECT * FROM students WHERE student_id = ?', (rollno,)).fetchone()
            if result is not None:
                return {
                    "message": "User already exists"
                }, 400
            for image, i in zip(images, range(10)):
                image.save('uploads/upload'+str(i)+'.jpg')
                face = detect('uploads/upload'+str(i)+'.jpg')
                user_directory = 'images/'+rollno
                if not os.path.exists(user_directory):
                    os.makedirs(user_directory)
                if face is not None:
                    cv2.imwrite(os.path.join(
                        user_directory, rollno + "_" + str(i) + '.jpg'), face)

            conn.execute('INSERT INTO students (student_id, name, email) VALUES (?, ?, ?)',
                         (rollno, name, email))
            conn.commit()
            conn.close()

            train_thread = threading.Thread(target=train_model)
            train_thread.start()
            return {
                "message": "User added successfully & training started"
            }, 200
        except:
            return {
                "message": "Internal server error"
            }, 500
    else:
        return {
            "message": "Invalid request"
        }, 400


def detect(image='uploads/upload.jpg', cascade='cascades/haarcascade_frontalface_default.xml'):
    faceCascade = cv2.CascadeClassifier(cascade)
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
    if faces is None or len(faces) == 0:
        return None
    max_area = faces[0][2] * faces[0][3]
    face = faces[0]
    for f in faces:
        if f[2] * f[3] > max_area:
            max_area = f[2] * f[3]
            face = f
    x, y, w, h = face
    return img[y:y+h, x:x+w]


def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer().create()
    conn = get_db_connection()
    result = conn.execute('SELECT * FROM students').fetchall()
    conn.commit()
    conn.close()
    if result is None or len(result) == 0:
        return
    faces = []
    ids = []
    for row in result:
        path = 'images/'+row['student_id']
        images = os.listdir(path)
        for image in images:
            face = cv2.imread(path+'/'+image)
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            faces.append(face)
            ids.append(row['id'])
    recognizer.train(faces, np.array(ids))
    recognizer.save('recognizer/model.yml')


def get_db_connection():
    conn = sqlite3.connect('db/database.db')
    conn.row_factory = sqlite3.Row
    return conn


port = int(os.environ.get('PORT', 5000))
app.run(debug=True, host='0.0.0.0', port=port)
