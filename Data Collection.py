import os
import cv2
import time


# CONFIGURATION

DATA_DIR = "./Custom_Data"
DATASET_SIZE = 50
CAMERA_INDEX = 0
CAPTURE_DELAY = .5
IMAGE_SIZE = 224

# Hand Placement Box -  ROI 
X1, Y1 = 150, 50
X2, Y2 = 450, 350


# Class_names
CLASS_NAMES = {}

for i, ch in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    CLASS_NAMES[i] = ch

for i, digit in enumerate("123456789"):
    CLASS_NAMES[26 + i] = digit

NUMBER_OF_CLASSES = len(CLASS_NAMES)


# folder creation
os.makedirs(DATA_DIR, exist_ok=True)

for label in CLASS_NAMES.values():
    os.makedirs(os.path.join(DATA_DIR, label), exist_ok=True)

# Open camera

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError(
        f"Cannot open camera index {CAMERA_INDEX}"
    )

print("\n=== ISL Dataset Collection ===")
print(f"Classes      : {NUMBER_OF_CLASSES}")
print(f"Images/Class : {DATASET_SIZE}")
print(f"Save Path    : {os.path.abspath(DATA_DIR)}\n")


# loop for start , skip and exit 

for class_id in range(NUMBER_OF_CLASSES):

    class_label = CLASS_NAMES[class_id]
    class_dir = os.path.join(DATA_DIR, class_label)

    print(f"\n[{class_id+1}/{NUMBER_OF_CLASSES}] Class: {class_label}")
    print("Press Q = Start | S = Skip | ESC = Exit")

    # wait screen
    
    while True:

        ret, frame = cap.read()

        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        # ROI box
        cv2.rectangle(
            frame,
            (X1, Y1),
            (X2, Y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "PLACE HAND HERE",
            (X1, Y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Class: {class_label}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Q=Start  S=Skip  ESC=Exit",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.imshow("ISL Dataset Collection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == ord("Q"):
            break

        elif key == ord("s") or key == ord("S"):
            print(f"Skipped {class_label}")
            class_label = None
            break

        elif key == 27:
            cap.release()
            cv2.destroyAllWindows()
            raise SystemExit

    if class_label is None:
        continue


    # countdown 
    
    for countdown in range(3, 0, -1):

        ret, frame = cap.read()

        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        cv2.rectangle(
            frame,
            (X1, Y1),
            (X2, Y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            str(countdown),
            (280, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            4,
            (0, 0, 255),
            6
        )

        cv2.imshow("ISL Dataset Collection", frame)
        cv2.waitKey(1000)

    #  captures images

    print(f"Capturing {DATASET_SIZE} images...")

    count = 0

    while count < DATASET_SIZE:

        ret, frame = cap.read()

        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        roi = frame[Y1:Y2, X1:X2]
        roi = cv2.resize(
            roi,
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        img_path = os.path.join(
            class_dir,
            f"{count}.jpg"
        )

        cv2.imwrite(img_path, roi)

        display = frame.copy()

        # ROI BOX
        cv2.rectangle(
            display,
            (X1, Y1),
            (X2, Y2),
            (0, 255, 0),
            2
        )

        # progress bar
        progress = int(
            (count / DATASET_SIZE) * 300
        )

        cv2.rectangle(
            display,
            (20, 420),
            (320, 450),
            (255, 255, 255),
            2
        )

        cv2.rectangle(
            display,
            (20, 420),
            (20 + progress, 450),
            (0, 255, 0),
            -1
        )

        cv2.putText(
            display,
            f"{class_label} [{count+1}/{DATASET_SIZE}]",
            (20, 400),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.imshow(
            "ISL Dataset Collection",
            display
        )

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            raise SystemExit

        count += 1

        time.sleep(CAPTURE_DELAY)

    print(f"✓ Saved {count} images for {class_label}")

# clean
cap.release()
cv2.destroyAllWindows()

print("\n================================")
print("Dataset Collection Complete!")
print(f"Saved To: {os.path.abspath(DATA_DIR)}")
print("================================")