import cv2
import random
from object_detection import ObjectDetection
from sort import *

# Initialize Object Detection
od = ObjectDetection()

cap = cv2.VideoCapture("test.mp4")

# Initialize count
count = 0
center_points_prev_frame = []

tracking_objects = {}
track_id = 0

mot_tracker = Sort()

# Initialize random colors
color_list = []
for j in range(1000):
    color_list.append(((int)(random.randrange(255)),(int)(random.randrange(255)),(int)(random.randrange(255))))

while True:
    ret, frame = cap.read()
    count += 1
    if not ret:
        break

    # Detect objects on frame
    (class_ids, scores, boxes) = od.detect(frame)

    indexes_of_person_classes = []
    index_of_person_class = 0
    for class_id in class_ids:
        if class_id == 0:
            indexes_of_person_classes.append(index_of_person_class)
        index_of_person_class += 1

    if len(indexes_of_person_classes) != 0:
        # Persons are found on the frame
        detects = np.zeros((len(indexes_of_person_classes), 5))
        count_detections = 0

        # Getting the parameters of the founded objects for the tracker
        for i in range(0, len(indexes_of_person_classes)):
            (x, y, w, h, s) = np.append(boxes[indexes_of_person_classes[i]], scores[indexes_of_person_classes[i]])
            detection = np.array([w, h, x, y, s])
            detects[count_detections, :] = detection[:]
            count_detections += 1

        trackers = mot_tracker.update(detects)
        for tracker in trackers:
            (w, h, x, y, id) = tracker
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)
            id = int(id)
            cx = int((x + x + w) / 2)
            cy = int((y + y + h) / 2)
            center_points_prev_frame.append((cx, cy))
            print("FRAME N°", count, " ", x, y, w, h)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color_list[(int)(id)], 2)

    cv2.imshow("Frame", frame)

    ret, frame = cap.read()
    ret, frame = cap.read()
    ret, frame = cap.read()
    ret, frame = cap.read()
    ret, frame = cap.read()
    ret, frame = cap.read()
    ret, frame = cap.read()
    ret, frame = cap.read()
    key = cv2.waitKey(0)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
