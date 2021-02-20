import logging as log
import sys
import time
import json
import os
from argparse import ArgumentParser, SUPPRESS

import cv2
import numpy as np
from gesture_recognition.common import load_ie_core
from gesture_recognition.video_stream import VideoStream
from gesture_recognition.tracking import Tracker
from gesture_recognition.action_recognition import ActionRecognizerOV
from gesture_recognition.visualizer import Visualizer
from gesture_recognition.person_detection import PersonDetector
from gesture_recognition.config import (
    DETECTOR_OUTPUT_SHAPE, SAMPLES_MAX_WINDOW_SIZE, SAMPLES_TRG_FPS, TRACKER_IOU_THRESHOLD, TRACKER_SCORE_THRESHOLD,
    ACTION_IMAGE_SCALE, ACTION_NET_INPUT_FPS, VISUALIZER_TRG_FPS, OBJECT_IDS, labels_map
)



def build_argparser():
    """ Returns argument parser. """

    parser = ArgumentParser()
    args = parser.add_argument_group('Options')
    args.add_argument('-m_a', '--action_model',
                      help='Required. Path to an .xml file with a trained gesture recognition model.',
                      required=True, type=str)
    args.add_argument('-m_d', '--detection_model',
                      help='Required. Path to an .xml file with a trained person detector model.',
                      required=True, type=str)
    args.add_argument('-i', '--input',
                      help='Required. Path to a video file or a device node of a web-camera.',
                      required=True, type=str)
    args.add_argument('-t', '--action_threshold',
                      help='Optional. Threshold for the predicted score of an action.',
                      default=0.8, type=float)
    args.add_argument('-d', '--device',
                      help='Optional. Specify the target device to infer on: CPU, GPU, FPGA, HDDL '
                           'or MYRIAD. The demo will look for a suitable plugin for device '
                           'specified (by default, it is CPU).',
                      default='CPU', type=str)
    args.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. Absolute path to "
                           "a shared library with the kernels implementations.", type=str,
                      default=None)
    args.add_argument('--no_show', help="Optional. Switch online/salient mode", action='store_true', required=False)

    return parser


def main():
    """ Main function. """

    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()

    ie_core = load_ie_core(args.device, args.cpu_extension)

    person_detector = PersonDetector(args.detection_model, args.device, ie_core,
                                     num_requests=2, output_shape=DETECTOR_OUTPUT_SHAPE)
    action_recognizer = ActionRecognizerOV(args.action_model, args.device, ie_core,
                                         num_requests=2, img_scale=ACTION_IMAGE_SCALE,
                                         num_classes=len(labels_map))
    person_tracker = Tracker(person_detector, TRACKER_SCORE_THRESHOLD, TRACKER_IOU_THRESHOLD)

    video_stream = VideoStream(args.input, ACTION_NET_INPUT_FPS, action_recognizer.input_length)
    video_stream.start()

    visualizer = Visualizer(VISUALIZER_TRG_FPS)
    visualizer.register_window('Gesture recognition')

    visualizer.start()

    last_caption = None
    active_object_id = -1
    tracker_labels_map = dict()
    tracker_labels = set()

    start_time = time.perf_counter()
    while True:
        frame = video_stream.get_live_frame()
        batch = video_stream.get_batch()
        if frame is None or batch is None:
            break

        detections, tracker_labels_map = person_tracker.add_frame(
            frame, len(OBJECT_IDS), tracker_labels_map)
        if detections is None:
            active_object_id = -1
            last_caption = None

        if len(detections) == 1:
            active_object_id = 0

        if active_object_id >= 0:
            cur_det = [det for det in detections if det.id == active_object_id]
            if len(cur_det) != 1:
                active_object_id = -1
                last_caption = None
                continue

            recognizer_result = action_recognizer(batch, cur_det[0].roi.reshape(-1))
            if recognizer_result is not None:
                action_class_id = np.argmax(recognizer_result)
                action_class_label = labels_map[action_class_id]

                action_class_score = np.max(recognizer_result)
                if action_class_score > args.action_threshold:
                    last_caption = 'Last gesture: {} '.format(action_class_label)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        start_time = end_time
        if active_object_id >= 0:
            current_fps = 1.0 / elapsed_time
            cv2.putText(frame, 'FPS: {:.2f}'.format(current_fps), (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if detections is not None:
            tracker_labels = set(det.id for det in detections)

            for det in detections:
                roi_color = (0, 255, 0) if active_object_id == det.id else (128, 128, 128)
                border_width = 2 if active_object_id == det.id else 1
                person_roi = det.roi[0]
                cv2.rectangle(frame, (person_roi[0], person_roi[1]),
                              (person_roi[2], person_roi[3]), roi_color, border_width)
                cv2.putText(frame, str(det.id), (person_roi[0] + 10, person_roi[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, roi_color, 2)

        if last_caption is not None:
            cv2.putText(frame, last_caption, (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if args.no_show:
            continue

        visualizer.put_queue(frame, 'Gesture recognition')
        key = visualizer.get_key()

        if key == 27:  # esc
            break
        elif key == ord(' '):  # space
            active_object_id = -1
            last_caption = None
        elif key == 13:  # enter
            last_caption = None
        elif key in OBJECT_IDS:  # 0-9
            local_bbox_id = int(chr(key))
            if local_bbox_id in tracker_labels:
                active_object_id = local_bbox_id

    visualizer.release()
    video_stream.release()


if __name__ == '__main__':
    sys.exit(main() or 0)
