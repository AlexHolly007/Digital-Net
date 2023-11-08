from ultralytics import YOLO
import cv2
import numpy as np
from tqdm import tqdm
from typing import List
import argparse

from supervision.utils.video import get_video_frames_generator, VideoInfo, VideoSink
from supervision.detection.core import Detections
from supervision.detection.annotate import BoxAnnotator
from supervision.detection.line_counter import LineZone, LineZoneAnnotator
from supervision.geometry.core import Point

from yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass


###########
###########
###
# Byte tracker classes and functions
@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False

def detections2boxes(detections: Detections) -> np.ndarray:
    return np.hstack((
        detections.xyxy,
        detections.confidence[:, np.newaxis]
    ))

def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([
        track.tlbr
        for track
        in tracks
    ], dtype=float)

def match_detections_with_tracks(
    detections: Detections,
    tracks: List[STrack]
) -> Detections:
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return np.empty((0,))

    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detections.xyxy)
    track2detection = np.argmax(iou, axis=1)

    tracker_ids = [None] * len(detections)

    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id

    return tracker_ids
# Byte tracker classes and functions
###
#########
#########






def main(SOURCE_VID, MODEL, VIDEO_OUTPUT):

    #######
    #######
    ## model, box annotation, and tracking initializing
    model = YOLO(MODEL)
    model.fuse()

    video_info = VideoInfo.from_video_path(SOURCE_VID)
    print("VIDEO INFO :", video_info)
    generator = get_video_frames_generator(SOURCE_VID)
    byte_tracker = BYTETracker(BYTETrackerArgs())
    box_annotator = BoxAnnotator(thickness=2, text_thickness=1, text_scale=.3)
    CLASS_NAMES_DICT = model.model.names

    LINE_START, LINE_END = Point(370,830), Point(1920-370,830)
    LINE_START2, LINE_END2 = Point(1920-370,250), Point(370,250)
    LINE_START3, LINE_END3 = Point(370,250), Point(370, 830)
    LINE_START4, LINE_END4 = Point(1920-370, 830), Point(1920-370, 250)

    line_counter, line_counter2 = LineZone(start=LINE_START, end=LINE_END), LineZone(start=LINE_START2, end=LINE_END2)
    line_counter3, line_counter4 = LineZone(start=LINE_START3, end=LINE_END3), LineZone(start=LINE_START4, end=LINE_END4)
    line_annotator = line_annotator2 = line_annotator3 = line_annotator4 = LineZoneAnnotator(thickness=3, text_thickness=1, text_scale=.4)
    ## model, box annotation, and tracking initializing
    #######
    #######



    with VideoSink(VIDEO_OUTPUT, video_info) as sink:
        for frame in tqdm(generator, total=video_info.total_frames): 

            results = model(frame)

            detections = Detections(
                xyxy=results[0].boxes.xyxy.cpu().numpy(),
                confidence=results[0].boxes.conf.cpu().numpy(),
                class_id=results[0].boxes.cls.cpu().numpy().astype(int)
            )


            tracks = byte_tracker.update(
                output_results=detections2boxes(detections=detections),
                img_info=frame.shape,
                img_size=frame.shape
            )

            tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
            detections.tracker_id = np.array(tracker_id)
        
            labels = [
                f"#{tracker_id}# {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
                for _, _, confidence, class_id, tracker_id
                in detections
            ]
        
            line_counter.trigger(detections=detections)
            line_counter2.trigger(detections=detections)
            line_counter3.trigger(detections=detections)
            line_counter4.trigger(detections=detections)

            frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
            line_annotator.annotate(frame=frame, line_counter=line_counter)
            line_annotator2.annotate(frame=frame, line_counter=line_counter2)
            line_annotator3.annotate(frame=frame, line_counter=line_counter3)
            line_annotator4.annotate(frame=frame, line_counter=line_counter4)
            sink.write_frame(frame)




##########
#######
## ARGUMENT COLLECTION AND MAIN CALL
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script is run using an input video, a yolov8 trained model, and an output video path. This is done \
            to detect, count, and label images within the video.")
    parser.add_argument("-v","--video", type=str, required=True, help="The path to the video in use.")
    parser.add_argument("-m","--model", type=str, required=True, help="The path to the model being used for detection on the video")
    parser.add_argument("-o","--output", type=str, required=True, help="Output path for the finsihed video")
    args = parser.parse_args()

    SOURCE_VID = args.video
    MODEL = args.model
    VIDEO_OUTPUT = args.output

    main(SOURCE_VID, MODEL, VIDEO_OUTPUT)
####
##########
########### end of file ####
