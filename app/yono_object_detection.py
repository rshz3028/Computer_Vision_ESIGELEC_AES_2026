import cv2
from ultralytics import YOLO

def main():
    # Pick a model: nano is fastest, small is a bit more accurate
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try changing the index (0,1,2...)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference (returns a list; webcam frame is a single image)
        results = model(frame, verbose=False)

        # Draw boxes
        annotated = results[0].plot()

        cv2.imshow("YOLOv8 Object Detection", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # q or ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
