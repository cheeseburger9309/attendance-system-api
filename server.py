import datetime
import os
import sqlite3
import threading

import cv2
import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route('/attendance', methods=['POST'])
def success():
    if request.method == 'POST':
        try:
            f = request.files['file']
            f.save('uploads/upload.jpg')

            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read('recognizer/trainingData.yml')
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

            idresult = conn.execute(
                'SELECT COUNT(*) FROM attendance').fetchone()
            id = len(idresult) + 1
            conn.execute('INSERT INTO attendance (id, student_id, date) VALUES (?, ?, ?)',
                         (id, student_id, sqlite_date_format))
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

            for image, i in zip(images, range(10)):
                image.save('uploads/upload'+str(i)+'.jpg')
                face = detect('uploads/upload'+str(i)+'.jpg')
                user_directory = 'images/'+rollno
                if not os.path.exists(user_directory):
                    os.makedirs(user_directory)
                if face is not None:
                    cv2.imwrite(os.path.join(
                        user_directory, rollno + "_" + str(i) + '.jpg'), face)

            conn = get_db_connection()
            result = conn.execute(
                'SELECT * FROM students WHERE student_id = ?', (rollno,)).fetchone()
            if result is not None:
                return {
                    "message": "User already exists"
                }, 400

            result = conn.execute('SELECT COUNT(*) FROM students').fetchone()
            id = len(result) + 1
            conn.execute('INSERT INTO students (id, student_id, name, email) VALUES (?, ?, ?, ?)',
                         (id, rollno, name, email))
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
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    if faces is None or len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    return img[y:y+h, x:x+w]


def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer().create()
    conn = get_db_connection()
    result = conn.execute('SELECT * FROM students').fetchall()
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
    conn.close()
    recognizer.train(faces, np.array(ids))
    recognizer.save('recognizer/model.yml')


def get_db_connection():
    conn = sqlite3.connect('db/database.db')
    conn.row_factory = sqlite3.Row
    return conn
