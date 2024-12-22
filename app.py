from flask import Flask, request, jsonify, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import base64
from flask_cors import CORS


app = Flask(__name__)
CORS(app)


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=2)
mp_drawing = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)



def process_image(image):
    
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = pose.process(imgRGB)

    
    output_img = image.copy()
    height, width, _ = image.shape

    
    if result.pose_landmarks:
        mp_drawing.draw_landmarks(
            output_img,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=5, circle_radius=3),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2),
        )

    return output_img



@app.route('/pose', methods=['POST'])
def detect_pose_from_image():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"error": "Invalid request"}), 400

    try:
       
        image_data = base64.b64decode(data['image'])
        np_image = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

        
        output_img = process_image(image)

        
        _, buffer = cv2.imencode('.jpg', output_img)
        base64_output = base64.b64encode(buffer).decode('utf-8')

        return jsonify({"image": base64_output})
    except Exception as e:
        return jsonify({"error": str(e)}), 500



def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

       
        frame = cv2.flip(frame, 1)

        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=5, circle_radius=3),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2),
            )

       
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

       
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n\r\n")



@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/')
def index():
    return render_template('index1.html')  



@app.teardown_appcontext
def cleanup(_=None):
    cap.release()


if __name__ == '__main__':
    app.run(debug=True)
