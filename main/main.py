import cv2
import random
import numpy as np
from detector import Detector
from tracker import Tracker
from utils.constants import *

# Initialize Object Detection
detector = Detector(weights_path, cfg_path, classes_path)
# Initialize Deep Sort Tracker
tracker = Tracker(encoder_model_path)
# Initialize frame counter
frame_counter = 0
# Initialize random colors
color_list = []
for j in range(1000):
    color_list.append(((int)(random.randrange(255)), (int)(random.randrange(255)), (int)(random.randrange(255))))
# Initialize trajectories chain
trajectories = {}
tracks_per_frame = {}
# Start video analysis
cap = cv2.VideoCapture(path_to_video)
ret, frame = cap.read()
skip = 100


def detect_manipulations(tracks_per_frame, manipulations_percent):
    if len(tracks_per_frame) != 0:
        detected_manipulations = []
        ids_on_prev_frame = []
        prev_frame_number = 0
        saved_ids_count = 0
        for frame_number in tracks_per_frame.keys():
            tracks_on_frame = tracks_per_frame[frame_number]
            ids_on_current_frame = []

            for (x, y, w, h, id) in tracks_on_frame:
                ids_on_current_frame.append(id)

            if len(ids_on_prev_frame) != 0:
                for current_id in ids_on_current_frame:
                    if ids_on_prev_frame.__contains__(current_id):
                        saved_ids_count = saved_ids_count + 1
                saved_ids_percent = (saved_ids_count / len(ids_on_prev_frame)) * 100
                if 100 - saved_ids_percent >= manipulations_percent:
                    detected_manipulations.append((prev_frame_number, frame_number))

            ids_on_prev_frame = ids_on_current_frame
            prev_frame_number = frame_number
            saved_ids_count = 0
        return detected_manipulations


while True:
    frame_counter += 1
    if not ret:
        break
    # if frame_counter > 0:
    if frame_counter > skip:
        # Detect objects on frame
        (class_ids, scores, boxes) = detector.detect(frame)

        detects = []
        for i in range(0, len(class_ids)):
            # Save only Person detection info
            if class_ids[i] == 0:
                (x, y, w, h, s) = np.append(boxes[i], scores[i])
                x = int(x)
                y = int(y)
                w = int(w)
                h = int(h)
                detection = [x, y, w, h, s]
                detects.append(detection)

        tracker.update(frame, detects)

        print("frame  x  y  w  h  id")

        for track in tracker.tracks:
            (x1, y1, x2, y2) = track.bbox

            x = int(x1)
            y = int(y1)
            w = int(x2 - x1)
            h = int(y2 - y1)
            id = track.track_id

            tracked_object = (x, y, w, h, id)

            if not tracks_per_frame.__contains__(frame_counter):
                tracks_per_frame[frame_counter] = [tracked_object]
            else:
                tracks_per_frame[frame_counter].append(tracked_object)

            print(frame_counter, " : ", x, y, w, h, id)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color_list[(int)(id)], 2)

    cv2.imshow("Frame", frame)

    ret, frame = cap.read()
    if frame_counter > skip:
        key = cv2.waitKey(1)
    else:
        key = cv2.waitKey(1)
    if key == 27:
        print(tracks_per_frame)
        break
print(detect_manipulations(tracks_per_frame, 70))
cap.release()
cv2.destroyAllWindows()
