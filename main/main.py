from detector import Detector
from tracker import Tracker
from manipulations_detector import ManipulationsDetector
from utils.constants import *
from video_processor import *

# Initialize YOLO object detector
detector = Detector(weights_path, cfg_path, classes_path)
# Initialize Deep Sort Tracker
tracker = Tracker(encoder_model_path)

if __name__ == '__main__':
    video_processor = VideoProcessor(path_to_video, 1000)
    tracks_per_frame = video_processor.process_video(0, detector, tracker)

    manipulations_detector = ManipulationsDetector(tracks_per_frame)
    print(manipulations_detector.detect_manipulations(70))
