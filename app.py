from flask import Flask, render_template, Response
import cv2
import pandas as pd
from datetime import datetime
import winsound

app = Flask(__name__)

# ---------------- LOAD CASCADES ----------------

face_cascade = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml'
)

eye_cascade = cv2.CascadeClassifier(
    'haarcascade_eye.xml'
)

# ---------------- CAMERA ----------------

camera = cv2.VideoCapture(0)

# ---------------- VARIABLES ----------------

closed_frames = 0
THRESHOLD_FRAMES = 15

alarm_played = False

status = "ACTIVE"

# ---------------- CSV LOG ----------------

log_file = "driver_log.csv"

try:
    pd.read_csv(log_file)

except:
    df = pd.DataFrame(columns=["Time", "Event"])
    df.to_csv(log_file, index=False)

# ---------------- VIDEO STREAM ----------------

def generate_frames():

    global closed_frames
    global alarm_played
    global status

    while True:

        success, frame = camera.read()

        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Face Detection
        faces = face_cascade.detectMultiScale(
            gray,
            1.3,
            5
        )

        status = "ACTIVE"
        color = (0, 255, 0)

        for (x, y, w, h) in faces:

            # Face Box
            cv2.rectangle(
                frame,
                (x, y),
                (x+w, y+h),
                (255, 0, 0),
                2
            )

            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            # ---------------- EYE DETECTION ----------------

            eyes = eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.1,
                minNeighbors=5
            )

            # Draw eye boxes
            for (ex, ey, ew, eh) in eyes:

                cv2.rectangle(
                    roi_color,
                    (ex, ey),
                    (ex+ew, ey+eh),
                    (0, 255, 0),
                    2
                )

            # ---------------- BETTER DROWSINESS LOGIC ----------------

            # If fewer than 2 eyes detected
            if len(eyes) < 2:
                closed_frames += 1

            else:
                closed_frames = 0
                alarm_played = False

            # ---------------- ALERT ----------------

            if closed_frames >= THRESHOLD_FRAMES:

                status = "DROWSY"
                color = (0, 0, 255)

                if not alarm_played:

                    winsound.Beep(1000, 1000)

                    alarm_played = True

                    # Save log
                    new_log = pd.DataFrame([{
                        "Time": datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "Event": "Drowsy Detected"
                    }])

                    old_logs = pd.read_csv(log_file)

                    updated_logs = pd.concat(
                        [old_logs, new_log],
                        ignore_index=True
                    )

                    updated_logs.to_csv(
                        log_file,
                        index=False
                    )

        # ---------------- UI ----------------

        cv2.rectangle(
            frame,
            (0, 0),
            (640, 80),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            "AI Driver Safety Monitoring System",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Status: {status}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            3
        )

        cv2.putText(
            frame,
            f"Closed Frames: {closed_frames}",
            (350, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # Encode Frame
        ret, buffer = cv2.imencode('.jpg', frame)

        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame +
            b'\r\n'
        )

# ---------------- ROUTES ----------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():

    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

# ---------------- RUN ----------------

if __name__ == "__main__":

    app.run(debug=True)