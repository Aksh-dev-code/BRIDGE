import os
import cv2
import time

# CONFIGURATION

DATA_DIR = "./Dataset"
DATASET_SIZE = 200
CAPTURE_DELAY = 0
IMAGE_SIZE = 224


# CAMERA SOURCE
CAMERA_SOURCE =  "http://10.118.2.94:8080/video"
 
MAX_READ_RETRIES = 30
RECONNECT_DELAY = 1.5  

# Hand Placement Box - ROI
X1, Y1 = 300, 50
X2, Y2 = 750, 500


# Classes

CLASS_NAMES = {}

for i, ch in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    CLASS_NAMES[i] = ch

for i, digit in enumerate("123456789"):
    CLASS_NAMES[26 + i] = digit

NUMBER_OF_CLASSES = len(CLASS_NAMES)


# Folder creation
os.makedirs(DATA_DIR, exist_ok=True)

for label in CLASS_NAMES.values():
    os.makedirs(os.path.join(DATA_DIR, label), exist_ok=True)


# Camera connection helpers

def open_camera(source):
    """Open a camera / stream and verify it actually delivers frames."""
    print(f"Connecting to camera source: {source} ...")
    cam = cv2.VideoCapture(source)

    if not cam.isOpened():
        raise RuntimeError(
            f"Cannot open camera source '{source}'.\n"
            f"If this is a phone stream, check that:\n"
            f"  1. The phone app's server is running.\n"
            f"  2. Phone and laptop are on the SAME WiFi network.\n"
            f"  3. The URL matches exactly what the app shows "
            f"(including '/video' if required)."
        )

    # Try a few times to get the first real frame - phone streams
    # can take a moment to start delivering data.
    for _ in range(MAX_READ_RETRIES):
        ret, frame = cam.read()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            print(f"Camera connected. Frame size: {w}x{h}")
            return cam
        time.sleep(0.1)

    cam.release()
    raise RuntimeError(
        "Camera opened but no frames were received. "
        "Check the stream URL / app and try again."
    )


def read_frame_safe(cam, source):
    """
    Read a frame, retrying and reconnecting if needed.
    Returns (cam, frame). Raises SystemExit if it truly can't recover.
    """
    for attempt in range(MAX_READ_RETRIES):
        ret, frame = cam.read()
        if ret and frame is not None:
            return cam, frame
        time.sleep(0.05)

    # Frame reads kept failing -> try a full reconnect (common with
    # phone WiFi streams that briefly drop).
    print("Lost connection to camera, attempting to reconnect...")
    cam.release()
    time.sleep(RECONNECT_DELAY)

    try:
        cam = open_camera(source)
    except RuntimeError as e:
        print(f"Reconnect failed: {e}")
        cv2.destroyAllWindows()
        raise SystemExit

    ret, frame = cam.read()
    if not ret or frame is None:
        cv2.destroyAllWindows()
        raise SystemExit("Camera reconnected but still no frames. Exiting.")

    return cam, frame


# Open camera

cap = open_camera(CAMERA_SOURCE)

print("\n=== ISL Dataset Collection ===")
print(f"Classes      : {NUMBER_OF_CLASSES}")
print(f"Images/Class : {DATASET_SIZE}")
print(f"Save Path    : {os.path.abspath(DATA_DIR)}\n")



# Main loop - start / skip / exit per class

for class_id in range(NUMBER_OF_CLASSES):

    class_label = CLASS_NAMES[class_id]
    class_dir = os.path.join(DATA_DIR, class_label)

    print(f"\n[{class_id+1}/{NUMBER_OF_CLASSES}] Class: {class_label}")
    print("Press Q = Start | S = Skip | ESC = Exit")

    # ---- wait screen ----
    while True:

        cap, frame = read_frame_safe(cap, CAMERA_SOURCE)
        frame = cv2.flip(frame, 1)

        cv2.rectangle(frame, (X1, Y1), (X2, Y2), (0, 255, 0), 2)

        cv2.putText(
            frame, "PLACE HAND HERE", (X1, Y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

        cv2.putText(
            frame, f"Class: {class_label}", (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )

        cv2.putText(
            frame, "Q=Start  S=Skip  ESC=Exit", (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )

        cv2.imshow("ISL Dataset Collection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q")):
            break

        elif key in (ord("s"), ord("S")):
            print(f"Skipped {class_label}")
            class_label = None
            break

        elif key == 27:
            cap.release()
            cv2.destroyAllWindows()
            raise SystemExit

    if class_label is None:
        continue

    # ---- countdown ----
    for countdown in range(3, 0, -1):

        cap, frame = read_frame_safe(cap, CAMERA_SOURCE)
        frame = cv2.flip(frame, 1)

        cv2.rectangle(frame, (X1, Y1), (X2, Y2), (0, 255, 0), 2)

        cv2.putText(
            frame, str(countdown), (280, 250),
            cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 6
        )

        cv2.imshow("ISL Dataset Collection", frame)
        cv2.waitKey(1000)

    # ---- capture images ----
    print(f"Capturing {DATASET_SIZE} images...")

    count = 0

    while count < DATASET_SIZE:

        cap, frame = read_frame_safe(cap, CAMERA_SOURCE)
        frame = cv2.flip(frame, 1)

        roi = frame[Y1:Y2, X1:X2]

        if roi.size == 0:
            # ROI fell outside the frame 
            print(
                "Warning: ROI box is outside the camera frame. "
                "Adjust X1,Y1,X2,Y2 to match your camera's resolution."
            )
            continue

        roi = cv2.resize(roi, (IMAGE_SIZE, IMAGE_SIZE))

        img_path = os.path.join(class_dir, f"{count}.jpg")
        cv2.imwrite(img_path, roi)

        display = frame.copy()

        cv2.rectangle(display, (X1, Y1), (X2, Y2), (0, 255, 0), 2)

        progress = int((count / DATASET_SIZE) * 300)

        cv2.rectangle(display, (20, 420), (320, 450), (255, 255, 255), 2)
        cv2.rectangle(display, (20, 420), (20 + progress, 450), (0, 255, 0), -1)

        cv2.putText(
            display, f"{class_label} [{count+1}/{DATASET_SIZE}]", (20, 400),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
        )

        cv2.imshow("ISL Dataset Collection", display)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            raise SystemExit

        count += 1
        time.sleep(CAPTURE_DELAY)

    print(f"✓ Saved {count} images for {class_label}")


cap.release()
cv2.destroyAllWindows()

print("\n================================")
print("Dataset Collection Complete!")
print(f"Saved To: {os.path.abspath(DATA_DIR)}")
print("================================")