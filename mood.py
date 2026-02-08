import cv2
from deepface import DeepFace
import time

cap = cv2.VideoCapture(0)

last_analysis_time = 0
analysis_interval = 2   # analyze every 2 seconds

happy_percent = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    # Run emotion analysis every few seconds (heavy model)
    if current_time - last_analysis_time > analysis_interval:
        last_analysis_time = current_time

        try:
            results = DeepFace.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=False
            )

            if not isinstance(results, list):
                results = [results]

            total_faces = len(results)
            happy_count = 0

            for res in results:
                emotion = res['dominant_emotion']
                if emotion == "happy":
                    happy_count += 1

            if total_faces > 0:
                happy_percent = int((happy_count / total_faces) * 100)

        except:
            pass

    # Display result
    cv2.putText(frame, f"Bus Happiness: {happy_percent}%",
                (30,50), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0,255,0), 3)

    cv2.imshow("Bus Emotion AI", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
