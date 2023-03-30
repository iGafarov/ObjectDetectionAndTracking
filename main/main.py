import cv2
import random
from object_detection import ObjectDetection
from object_tracking import *
from tracker import Tracker

weights_path = "C:\\Users\\iskan\\PycharmProjects\\ObjectDetectionAndTracking\\dnn_model\\yolov4.weights"
cfg_path = "C:\\Users\\iskan\\PycharmProjects\\ObjectDetectionAndTracking\\dnn_model\\yolov4.cfg"
path_to_video = "C:\\Users\\iskan\\PycharmProjects\\ObjectDetectionAndTracking\\resources\\Walkers.mp4"
classes_path = "C:\\Users\\iskan\\PycharmProjects\\ObjectDetectionAndTracking\\dnn_model\\classes.txt"

# Initialize Object Detection
od = ObjectDetection(weights_path, cfg_path, classes_path)
# Initialize Deep Sort Tracker
tracker = Tracker()

# Initialize count
count = 0
center_points_prev_frame = []

tracking_objects = {}
track_id = 0


# Initialize random colors
color_list = []
for j in range(1000):
    color_list.append(((int)(random.randrange(255)),(int)(random.randrange(255)),(int)(random.randrange(255))))

trajectories = {}

cap = cv2.VideoCapture(path_to_video)

ret, frame = cap.read()
while True:
    count += 1
    if not ret:
        break

    # Detect objects on frame
    (class_ids, scores, boxes) = od.detect(frame)

    detects = []
    for i in range(0, len(class_ids)):
        if class_ids[i] == 0:
            (x, y, w, h, s) = np.append(boxes[i], scores[i])
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)
            detection = [x, y, w, h, s]
            detects.append(detection)
    #     ===============================================
    # indexes_of_person_classes = []
    # index_of_person_class = 0
    # for class_id in class_ids:
    #     if class_id == 0:
    #         indexes_of_person_classes.append(index_of_person_class)
    #     index_of_person_class += 1
    #
    # if len(indexes_of_person_classes) != 0:
    #     # Persons are found on the frame
    #     detects = np.zeros((len(indexes_of_person_classes), 5))
    #     count_detections = 0
    #
    #     # Getting the parameters of the founded objects for the tracker
    #     for i in range(0, len(indexes_of_person_classes)):
    #         (x, y, w, h, s) = np.append(boxes[indexes_of_person_classes[i]], scores[indexes_of_person_classes[i]])
    #         detection = np.array([w, h, x, y, s])
    #         detects[count_detections, :] = detection[:]
    #         count_detections += 1
    #     ===================================================
    tracker.update(frame, detects)

    print("frame  x  y  w  h  id")

    for track in tracker.tracks:
        (x1, y1, x2, y2) = track.bbox

        x = int(x1)
        y = int(y1)
        w = int(x2 - x1)
        h = int(y2 - y1)
        id = track.track_id

        bbox = (x, y, w, h)

        # if count == 1 or not trajectories.keys().__contains__(id):
        #     list_of_tracks = [bbox]
        #     trajectories[id] = list_of_tracks
        # else:
        #     list_of_tracks = trajectories[id]
        #     list_of_tracks.append(bbox)

        cx = int((x + x + w) / 2)
        cy = int((y + y + h) / 2)
        center_points_prev_frame.append((cx, cy))
        print(count, " : ", x, y, w, h, id)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color_list[(int)(id)], 2)

    cv2.imshow("Frame", frame)

    ret, frame = cap.read()
    key = cv2.waitKey(1)
    if key == 27:
        # print(trajectories)
        break

cap.release()
cv2.destroyAllWindows()
