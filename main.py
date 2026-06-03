import cv2
import pandas as pd
from datetime import datetime
import winsound

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml'
)

eye_cascade = cv2.CascadeClassifier(
    'haarcascade_eye.xml'
)

# Start webcam
cap = cv2.VideoCapture(0)

closed_frames = 0
THRESHOLD_FRAMES = 20

alarm_played = False

# CSV file
log_file = "driver_log.csv"

try:
    pd.read_csv(log_file)

except:
    df = pd.DataFrame(columns=["Time", "Event"])
    df.to_csv(log_file, index=False)

# ---------------- MAIN LOOP ----------------

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ORIGINAL FACE SETTINGS
    faces = face_cascade.detectMultiScale(
        gray,
        1.3,
        5
    )

    status = "ACTIVE"
    color = (0, 255, 0)

    for (x, y, w, h) in faces:

        # Face box
        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (255, 0, 0),
            2
        )

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # ORIGINAL EYE SETTINGS (BEST)
        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            1.1,
            10
        )

        eye_closed = True

        for (ex, ey, ew, eh) in eyes:

            # Eye rectangle
            cv2.rectangle(
                roi_color,
                (ex, ey),
                (ex+ew, ey+eh),
                (0, 255, 0),
                2
            )

            # ORIGINAL WORKING LOGIC
            if eh > 15:
                eye_closed = False

        # Drowsiness logic
        if eye_closed:
            closed_frames += 1

        else:
            closed_frames = 0
            alarm_played = False

        # Alert logic
        if closed_frames >= THRESHOLD_FRAMES:

            status = "DROWSY"
            color = (0, 0, 255)

            # Alarm once
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

        else:
            status = "ACTIVE"
            color = (0, 255, 0)

    # ---------------- UI ----------------

    # Top black bar
    cv2.rectangle(
        frame,
        (0, 0),
        (640, 80),
        (0, 0, 0),
        -1
    )

    # Title
    cv2.putText(
        frame,
        "Driver Safety Monitoring System",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    # Status
    cv2.putText(
        frame,
        f"Status: {status}",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        3
    )

    # Counter
    cv2.putText(
        frame,
        f"Closed Frames: {closed_frames}",
        (360, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # Exit text
    cv2.putText(
        frame,
        "Press Q or ESC to Exit",
        (170, 460),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (200, 200, 200),
        2
    )

    # Show window
    cv2.imshow("Driver Monitor", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27 or key == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()