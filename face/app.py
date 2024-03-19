import os
import cv2
import face_recognition
import mysql.connector
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import time
from win10toast import ToastNotifier

video_capture = cv2.VideoCapture(0)
frame_width = int(video_capture.get(3))
frame_height = int(video_capture.get(4))
IMAGE_BASE_PATH = 'image/'
IMAGE_SAVE_PATH = 'saved_images/'

if not os.path.exists(IMAGE_SAVE_PATH):
    os.makedirs(IMAGE_SAVE_PATH)

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="nuntaburee-tis"
)
mycursor = mydb.cursor()
toaster = ToastNotifier()
last_detection = {}

def draw_text(image, text, position, font_path="THSarabunNew.ttf", font_size=32, color=(255, 255, 255)):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

def show_notification(message):
    toaster.show_toast("Face Recognition", message, duration=5)

def fetch_registered_users():
    mycursor.execute("SELECT id_user, user_name, last_name, picture, position FROM teacher_personnel_information")
    return mycursor.fetchall()

def load_registered_faces():
    registered_faces = fetch_registered_users()
    registered_encodings = {}
    for id_user, first_name, last_name, image_path, position in registered_faces:
        full_path = os.path.join(IMAGE_BASE_PATH, image_path)
        try:
            image = face_recognition.load_image_file(full_path)
            encoding = face_recognition.face_encodings(image)[0]
            registered_encodings[id_user] = (first_name, last_name, encoding, position)
        except Exception as e:
            print(f"Error loading {full_path}: {e}")
    return registered_encodings

def save_image(image, name):
    now = datetime.now()
    filename = f"{name}_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
    filepath = os.path.join(IMAGE_SAVE_PATH, filename)
    cv2.imwrite(filepath, image)
    return filepath

def user_has_checked_in_today(user_id):
    query = "SELECT COUNT(*) FROM face_recognition_data WHERE user_id = %s AND DATE(attend_work) = CURDATE()"
    mycursor.execute(query, (user_id,))
    result = mycursor.fetchone()
    return result[0] > 0

registered_encodings = load_registered_faces()

while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    frame = cv2.resize(frame, (frame_width, frame_height))
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        match = False
        for user_id, (first_name, last_name, known_encoding, position) in registered_encodings.items():
            if face_recognition.compare_faces([known_encoding], face_encoding)[0]:
                match = True
                current_time = time.time() # Use time.time() to get the current time
                if user_id not in last_detection or (current_time - last_detection[user_id]) > 5:
                    last_detection[user_id] = current_time
                    if not user_has_checked_in_today(user_id):
                        cropped_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)[top:bottom, left:right]
                        image_path = save_image(cropped_frame, first_name)
                        sql = "INSERT INTO face_recognition_data (user_id, name, last_name, image_path, position, attend_work) VALUES (%s, %s, %s, %s, %s, %s)"
                        val = (user_id, first_name, last_name, image_path, position, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                        mycursor.execute(sql, val)
                        mydb.commit()
                        show_notification(f"เข้างาน: {first_name} {last_name}")
                    else:
                        sql_update = "UPDATE face_recognition_data SET leaving_work = %s WHERE user_id = %s AND DATE(attend_work) = CURDATE()"
                        val_update = (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), user_id)
                        mycursor.execute(sql_update, val_update)
                        mydb.commit()
                        show_notification(f"ออกงาน: {first_name} {last_name}")
                    break

        if not match:
            frame = draw_text(frame, "ไม่รู้จัก", (left, top - 10), "THSarabunNew.ttf", 32)
    
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
