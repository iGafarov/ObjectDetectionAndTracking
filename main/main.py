from detector import Detector
from tracker import Tracker
from manipulations_detector import ManipulationsDetector
from utils.constants import *
from video_processor import VideoProcessor
from utils.video_paths_collector import VideoPathsCollector
from utils.excel_writer import ExcelWriter

# Initialize YOLO Object Detector
detector = Detector(weights_path, cfg_path, classes_path)
# Initialize Deep Sort Tracker
tracker = Tracker(encoder_model_path)
# Initialize Videos Collector
video_paths_collector = VideoPathsCollector(path_to_videos)
# Initialize Excel Writer
excel_writer = ExcelWriter("C:\\Users\\iskan\\PycharmProjects\\ObjectDetectionAndTracking\\resources\\results.xlsx")


def get_video_name(path: str):
    parsed = path.split("\\")
    return parsed[len(parsed) - 1]


if __name__ == '__main__':
    videos_paths = video_paths_collector.collect()
    all_manipulations = {}
    for video_path in videos_paths:
        video_processor = VideoProcessor(video_path, 1000)
        tracks_per_frame = video_processor.process_video(50, detector, tracker)

        manipulations_detector = ManipulationsDetector(tracks_per_frame)
        manipulations = manipulations_detector.detect_manipulations(70)
        print('Manipulations on ', video_path, ':\n', manipulations)
        all_manipulations[get_video_name(video_path)] = manipulations
    if len(all_manipulations) != 0:
        ExcelWriter.write(excel_writer, all_manipulations, True)


