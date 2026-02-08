import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n-face.pt")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model.predict(frame, conf=0.4, verbose=False)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # --- SHRINK THE BOX ---
            shrink = 0.01  # 10% shrink from each side

            w = x2 - x1
            h = y2 - y1

            x1 += int(w * shrink)
            y1 += int(h * shrink)
            x2 -= int(w * shrink)
            y2 -= int(h * shrink)

            # Make sure coordinates stay valid
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            # Crop face
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            # --- APPLY BLUR ---
            face_blur = cv2.GaussianBlur(face, (61, 61), 20)

            # Put blurred region back
            frame[y1:y2, x1:x2] = face_blur

    # Show output
    cv2.imshow("Face Blur (Shrunk Box)", frame)

    # Press ESC to exit
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
