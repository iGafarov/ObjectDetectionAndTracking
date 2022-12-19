import cv2
import numpy as np
from object_detection import ObjectDetection
import math

# Initialize Object Detection
od = ObjectDetection()

# cap = cv2.VideoCapture("kakoitomyzhik.mp4")
# cap = cv2.VideoCapture("kandibober_edit.mp4")
cap = cv2.VideoCapture("test.mp4")

# Initialize count
count = 0
center_points_prev_frame = []

tracking_objects = {}
track_id = 0

while True:
    ret, frame = cap.read()
    count += 1
    if not ret:
        break

    # Point current frame
    center_points_cur_frame = []

    # Detect objects on frame
    (class_ids, scores, boxes) = od.detect(frame)

    indexes_of_person_classes = []
    index_of_person_class = 0
    for class_id in class_ids:
        if class_id == 0:
            indexes_of_person_classes.append(index_of_person_class)
        index_of_person_class += 1
    if len(indexes_of_person_classes) != 0:
        for i in range(0, len(indexes_of_person_classes)):
            (x, y, w, h) = boxes[indexes_of_person_classes[i]]
            cx = int((x + x + w) / 2)
            cy = int((y + y + h) / 2)
            # center_points_cur_frame.append((cx, cy))
            center_points_prev_frame.append((cx, cy))
            print("FRAME N°", count, " ", x, y, w, h)

            # print(cv2.circle(frame, (cx, cy), 3, (0, 0, 255), 1))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (128, 0, 128), 2)
        for pt in center_points_prev_frame:
            cv2.circle(frame, pt, 3, (0, 0, 255), 1)
        # Only at the beginning we compare previous and current frame
        # if count <= 2:
        #     for pt in center_points_cur_frame:
        #         for pt2 in center_points_prev_frame:
        #             distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
        #
        #             if distance < 20:
        #                 tracking_objects[track_id] = pt
        #                 track_id += 1
        # else:
        #
        #     tracking_objects_copy = tracking_objects.copy()
        #     center_points_cur_frame_copy = center_points_cur_frame.copy()
        #
        #     for object_id, pt2 in tracking_objects_copy.items():
        #         object_exists = False
        #         for pt in center_points_cur_frame_copy:
        #             distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
        #
        #             # Update IDs position
        #             if distance < 20:
        #                 tracking_objects[object_id] = pt
        #                 object_exists = True
        #                 if pt in center_points_cur_frame:
        #                     center_points_cur_frame.remove(pt)
        #                 continue
        #
        #         # Remove IDs lost
        #         if not object_exists:
        #             tracking_objects.pop(object_id)
        #
        #     # Add new IDs found
        #     for pt in center_points_cur_frame:
        #         tracking_objects[track_id] = pt
        #         track_id += 1
        #
        # for object_id, pt in tracking_objects.items():
        #     cv2.circle(frame, pt, 5, (0, 0, 255), -1)
        #     cv2.putText(frame, str(object_id), (pt[0], pt[1] - 7), 0, 1, (0, 0, 255), 2)

        # print("Tracking objects")
        # print(tracking_objects)
        #
        #
        # print("CUR FRAME LEFT PTS")
        # print(center_points_cur_frame)


    cv2.imshow("Frame", frame)

    # Make a copy of the points
    # center_points_prev_frame = center_points_cur_frame.copy()
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
