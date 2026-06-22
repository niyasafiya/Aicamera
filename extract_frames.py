import cv2, sys
sys.path.insert(0, ".")

cap = cv2.VideoCapture("demo_plates.mp4")
frames_to_test = [30, 90, 150, 210, 300, 400, 500]

for fn in frames_to_test:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(f"test_frame_{fn}.jpg", frame)
        print(f"Saved frame {fn}")

cap.release()
print("Frames extracted OK")
