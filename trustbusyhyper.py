import cv2
from ultralytics import YOLO
import numpy as np

# Load YOLO model
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

prev_center = None
alert_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=0.4, verbose=False)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Calculate center of detected person
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            current_center = np.array([center_x, center_y])

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if prev_center is not None:
                movement = np.linalg.norm(current_center - prev_center)

                # Show movement value
                cv2.putText(frame, f"Movement: {int(movement)}",
                            (x1, y1 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 0), 2)

                # --- Hyper motion detection ---
                if movement > 40:      # threshold (adjustable)
                    alert_counter += 1
                else:
                    alert_counter = 0

                # If high movement continues → show alert
                if alert_counter > 5:
                    cv2.putText(frame, "⚠ CHILD MOVING TOO MUCH!",
                                (40, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 3)

            prev_center = current_center

    cv2.imshow("Trust Bus Safety Monitor", frame)

    # Press ESC to exit
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
