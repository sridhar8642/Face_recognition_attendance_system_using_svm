from flask import Flask, request, render_template, redirect, url_for, flash, Response, jsonify, send_file
import os
import cv2
import numpy as np
import base64
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle
from imutils import paths
import dlib
import logging
import mysql.connector
import csv

# Configuration
app = Flask(__name__)
app.secret_key = 'your_secret_key'

DATASET_DIR = 'dataset'
MODEL_PATH = 'model'
OUTPUT_PATH = 'output'
CASCADE_PATH = 'haarcascade_frontalface_default.xml'
EMBEDDING_FILE = os.path.join(OUTPUT_PATH, "embeddings.pickle")
RECOGNIZER_FILE = os.path.join(OUTPUT_PATH, "recognizer.pickle")
LABEL_ENCODER_FILE = os.path.join(OUTPUT_PATH, "le.pickle")
SHAPE_PREDICTOR = os.path.join(MODEL_PATH, "shape_predictor_68_face_landmarks.dat")
FACE_RECOGNITION_MODEL = os.path.join(MODEL_PATH, "dlib_face_recognition_resnet_model_v1.dat")
CONFIDENCE_THRESHOLD = 0.5

# MySQL Configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '1234',
    'database': 'attendance_system_new'
}

# Logging Configuration
logging.basicConfig(level=logging.DEBUG)

# Ensure necessary directories exist
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Load Dlib models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR)
embedder = dlib.face_recognition_model_v1(FACE_RECOGNITION_MODEL)

# Database connection
def get_db_connection():
    return mysql.connector.connect(**db_config)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/create_dataset')
def create_dataset():
    return render_template('create_dataset.html')


@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    try:
        # Validate form data
        name = request.form.get('name')
        roll_number = request.form.get('roll_number')

        if not name or not roll_number:
            return jsonify({"error": "Name and Roll Number are required!"}), 400
        
        # Connect to the database
        connection = get_db_connection()
        cursor = connection.cursor()

        # Check if the student already exists in the database
        cursor.execute("SELECT * FROM students WHERE roll_number = %s", (roll_number,))
        existing_student = cursor.fetchone()

        if existing_student:
            return jsonify({"error": "Student with this roll number already exists!"}), 400

        # Insert student details into the database
        cursor.execute("INSERT INTO students (name, roll_number) VALUES (%s, %s)", (name, roll_number))
        connection.commit()

        # Create user-specific directory
        user_dir = os.path.join(DATASET_DIR, name)
        os.makedirs(user_dir, exist_ok=True)

        # Process each image from the request
        for key in request.form:
            if key.startswith('image_'):
                img_data = request.form[key].split(",")[1]
                img_bytes = base64.b64decode(img_data)
                img_path = os.path.join(user_dir, f"{key}.png")
                with open(img_path, "wb") as img_file:
                    img_file.write(img_bytes)

        return jsonify({"message": "Dataset uploaded and student registered successfully!"}), 200

    except Exception as e:
        logging.exception("Error during dataset upload.")
        return jsonify({"error": str(e)}), 500

    finally:
        if connection:
            cursor.close()
            connection.close()

# Route to display the registered students
@app.route('/view_students', methods=['GET'])
def view_students():
    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        # Fetch all students from the database
        cursor.execute("SELECT id, name, roll_number FROM students")
        students = cursor.fetchall()

        # Return a list of students
        return render_template('view_students.html', students=students)

    except Exception as e:
        logging.exception("Error fetching students.")
        return jsonify({"error": str(e)}), 500

    finally:
        if connection:
            cursor.close()
            connection.close()
            
@app.route('/preprocess_embeddings', methods=['POST'])
def preprocess_embeddings():
    try:
        image_paths = list(paths.list_images(DATASET_DIR))
        known_embeddings = []
        known_names = []

        for (i, image_path) in enumerate(image_paths):
            logging.info(f"Processing image {i + 1}/{len(image_paths)}: {image_path}")
            name = os.path.basename(os.path.dirname(image_path))
            image = cv2.imread(image_path)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            boxes = detector(rgb_image, 1)

            for box in boxes:
                shape = predictor(rgb_image, box)
                face_descriptor = embedder.compute_face_descriptor(rgb_image, shape)
                known_names.append(name)
                known_embeddings.append(np.array(face_descriptor))

        data = {"embeddings": known_embeddings, "names": known_names}
        with open(EMBEDDING_FILE, "wb") as f:
            pickle.dump(data, f)

        flash("Embeddings preprocessed successfully!", "success")
        return redirect(url_for('index'))

    except Exception as e:
        logging.exception("Error during preprocessing embeddings.")
        flash(f"Error: {str(e)}", "danger")
        return redirect(url_for('index'))


@app.route('/train_model', methods=['POST'])
def train_model():
    try:
        if not os.path.exists(EMBEDDING_FILE):
            flash("Embeddings file not found. Please preprocess embeddings first.", "danger")
            return redirect(url_for('index'))

        with open(EMBEDDING_FILE, "rb") as f:
            data = pickle.load(f)

        known_names = data["names"]
        if len(set(known_names)) < 2:
            flash("Error: At least two unique individuals required for training.", "danger")
            return redirect(url_for('index'))

        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(known_names)

        recognizer = SVC(C=1.0, kernel="linear", probability=True)
        recognizer.fit(data["embeddings"], labels)

        with open(RECOGNIZER_FILE, "wb") as f:
            pickle.dump(recognizer, f)
        with open(LABEL_ENCODER_FILE, "wb") as f:
            pickle.dump(label_encoder, f)

        flash("Model trained successfully!", "success")
        return redirect(url_for('index'))

    except Exception as e:
        logging.exception("Error during model training.")
        flash(f"Error: {str(e)}", "danger")
        return redirect(url_for('index'))



@app.route('/recognize', methods=['GET', 'POST'])
def recognize():
    if request.method == 'GET':
        return render_template('recognize.html')  # For GET requests, render the page.
    elif request.method == 'POST':
        try:
            # Decode the incoming base64 frame
            frame_data = request.json.get('frame')
            if not frame_data:
                return jsonify({"error": "No frame data received"}), 400

            # Process the frame (base64 to image)
            frame_data = frame_data.split(',')[1]  # Strip base64 prefix
            frame_bytes = base64.b64decode(frame_data)
            frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Load recognizer and label encoder (ensure they exist)
            recognizer = None
            le = None
            if os.path.exists(RECOGNIZER_FILE) and os.path.exists(LABEL_ENCODER_FILE):
                with open(RECOGNIZER_FILE, "rb") as f:
                    recognizer = pickle.load(f)
                with open(LABEL_ENCODER_FILE, "rb") as f:
                    le = pickle.load(f)

            if recognizer is None or le is None:
                return jsonify({"error": "Recognizer or label encoder not found. Please train the model first."}), 400

            # Detect faces in the frame
            faces = detector(rgb_frame)
            results = []

            for face in faces:
                shape = predictor(rgb_frame, face)
                face_embedding = embedder.compute_face_descriptor(rgb_frame, shape)

                # Perform recognition
                predictions = recognizer.predict_proba([np.array(face_embedding)])[0]
                max_index = np.argmax(predictions)
                confidence = predictions[max_index]
                name = le.classes_[max_index] if confidence > CONFIDENCE_THRESHOLD else "Unknown"

                if name != "Unknown":
                    status = log_attendance(name)
                    if status == "marked":
                        display_text = f"{name} ({confidence * 100:.2f}%): Attendance Marked"
                    else:
                        display_text = f"{name} ({confidence * 100:.2f}%): Already Marked"
                else:
                    display_text = "Unknown"

                results.append({
                    "name": name,
                    "confidence": round(confidence * 100, 2),
                    "status": display_text,
                    "box": [face.left(), face.top(), face.right(), face.bottom()]
                })

            return jsonify({"results": results})

        except Exception as e:
            return jsonify({"error": f"Error processing frame: {str(e)}"}), 500
def log_attendance(name):
    # Log attendance in database
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM attendance WHERE name=%s AND DATE(timestamp) = CURDATE()", (name,))
    existing_record = cursor.fetchone()

    if existing_record:
        status = "already_marked"
    else:
        timestamp = datetime.now()
        cursor.execute("INSERT INTO attendance (name, timestamp) VALUES (%s, %s)", (name, timestamp))
        conn.commit()
        status = "marked"

    cursor.close()
    conn.close()
    return status


@app.route('/view_attendance', methods=['GET'])
def view_attendance():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT name, timestamp FROM attendance ORDER BY timestamp DESC")
    records = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template('view_attendance.html', records=records)


@app.route('/export_attendance')
def export_attendance():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT name, timestamp FROM attendance")
    rows = cursor.fetchall()

    csv_file = 'attendance.csv'
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Name', 'Timestamp'])
        writer.writerows(rows)

    cursor.close()
    conn.close()

    return send_file(csv_file, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
