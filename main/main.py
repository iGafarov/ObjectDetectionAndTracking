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
# Start video analysis
cap = cv2.VideoCapture(path_to_video)
ret, frame = cap.read()
skip = 105


def analyze_trajectories(trajectories, step):
    person_count = len(trajectories)
    if person_count != 0:
        for object_id in trajectories.keys():
            trajectory = trajectories[object_id]

            frames = list(trajectory.keys())
            first_detected_frame = frames[0]
            (prev_x, prev_y, prev_w, prev_h) = trajectory[first_detected_frame]
            frame_from = first_detected_frame
            for frame_key in frames:
                (cur_x, cur_y, cur_w, cur_h) = trajectory[frame_key]
                if abs(cur_x - prev_x) > step or abs(cur_y - prev_y) > step or abs(cur_w - prev_w) > step or abs(
                        cur_h - prev_h) > step:
                    # if frame_from == 0:
                    print('Video montage FOUNDED!')
                    print('Person id: ', object_id)
                    print('From frame ', frame_from, '. bbox: ', prev_x, ' ', prev_y, ' ', prev_w, ' ', prev_h)
                    print('To frame ', frame_key, '. bbox: ', cur_x, ' ', cur_y, ' ', cur_w, ' ', cur_h)
                # if abs(cur_x - prev_x) < 2 and abs(cur_y - prev_y) < 2 and abs(cur_w - prev_w) < 2 and abs(
                #         cur_h - prev_h) < 2:
                #     if frame_from == 0:
                #         frame_from = frame_count - 1
                #     frame_count = frame_count + 1
                #     continue
                frame_from = frame_key
                prev_x = cur_x
                prev_y = cur_y
                prev_w = cur_w
                prev_h = cur_h


# def detect_manipulations(trajectories):
#     person_count = len(trajectories)
#     if person_count != 0:
#         for object_id in trajectories.keys():
#             trajectory = trajectories[object_id]
#
#             frames = list(trajectory.keys())
#             first_detected_frame = frames[0]
#             (prev_x, prev_y, prev_w, prev_h) = trajectory[first_detected_frame]
#             frame_from = first_detected_frame
#             for frame_key in frames:
#                 (cur_x, cur_y, cur_w, cur_h) = trajectory[frame_key]
#                 if abs(cur_x - prev_x) > step or abs(cur_y - prev_y) > step or abs(cur_w - prev_w) > step or abs(
#                         cur_h - prev_h) > step:
#                     # if frame_from == 0:
#                     print('Video montage FOUNDED!')
#                     print('Person id: ', object_id)
#                     print('From frame ', frame_from, '. bbox: ', prev_x, ' ', prev_y, ' ', prev_w, ' ', prev_h)
#                     print('To frame ', frame_key, '. bbox: ', cur_x, ' ', cur_y, ' ', cur_w, ' ', cur_h)
#                 # if abs(cur_x - prev_x) < 2 and abs(cur_y - prev_y) < 2 and abs(cur_w - prev_w) < 2 and abs(
#                 #         cur_h - prev_h) < 2:
#                 #     if frame_from == 0:
#                 #         frame_from = frame_count - 1
#                 #     frame_count = frame_count + 1
#                 #     continue
#                 frame_from = frame_key
#                 prev_x = cur_x
#                 prev_y = cur_y
#                 prev_w = cur_w
#                 prev_h = cur_h


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

            bbox = (x, y, w, h)

            if frame_counter == 1 or not trajectories.keys().__contains__(id):
                dictionary_of_tracks = {frame_counter: bbox}
                trajectories[id] = dictionary_of_tracks
            else:
                dictionary_of_tracks = trajectories[id]
                dictionary_of_tracks[frame_counter] = bbox

            print(frame_counter, " : ", x, y, w, h, id)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color_list[(int)(id)], 2)

    cv2.imshow("Frame", frame)
    print('Frame: ', frame_counter)

    ret, frame = cap.read()
    if frame_counter > skip:
        key = cv2.waitKey(1)
    else:
        key = cv2.waitKey(1)
    if key == 27:
        print(trajectories)
        analyze_trajectories(trajectories, 38)
        break
cap.release()
cv2.destroyAllWindows()
