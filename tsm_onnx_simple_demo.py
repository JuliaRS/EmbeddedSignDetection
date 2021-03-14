import os
import time
import argparse
import numpy as np
import cv2
import light_control
from action_recognizer import TSMActionRecognizer
from detector import Detector, Tracker

SOFTMAX_THRES = 0.8
HISTORY_LOGIT = True
REFINE_OUTPUT = True
TRACKER_SCORE_THRESHOLD = 0.4
TRACKER_IOU_THRESHOLD = 0.3
WINDOW_NAME = 'Video Gesture Recognition'

categories = [
    "Doing other things",  # 0
    "Drumming Fingers",  # 1
    "No gesture",  # 2
    "Pulling Hand In",  # 3
    "Pulling Two Fingers In",  # 4
    "Pushing Hand Away",  # 5
    "Pushing Two Fingers Away",  # 6
    "Rolling Hand Backward",  # 7
    "Rolling Hand Forward",  # 8
    "Shaking Hand",  # 9
    "Sliding Two Fingers Down",  # 10
    "Sliding Two Fingers Left",  # 11
    "Sliding Two Fingers Right",  # 12
    "Sliding Two Fingers Up",  # 13
    "Stop Sign",  # 14
    "Swiping Down",  # 15
    "Swiping Left",  # 16
    "Swiping Right",  # 17
    "Swiping Up",  # 18
    "Thumb Down",  # 19
    "Thumb Up",  # 20
    "Turning Hand Clockwise",  # 21
    "Turning Hand Counterclockwise",  # 22
    "Zooming In With Full Hand",  # 23
    "Zooming In With Two Fingers",  # 24
    "Zooming Out With Full Hand",  # 25
    "Zooming Out With Two Fingers"  # 26
]


def build_argparser():
    parser = argparse.ArgumentParser('TSM simple demo')
    parser.add_argument('-m', '--model', help='path to onnx model', required=True)
    parser.add_argument('-s', '--stream', help='mode for demo video stream', choices=['gui', 'video', 'none'],
                        default='gui')
    parser.add_argument('-m_d', '--detection_model', help='path to onnx detector model', required=True)
    parser.add_argument('-d', '--device', help='device for execution', required=False, default='CPU')                          

    return parser

def main():
    args = build_argparser().parse_args()
    print("Build Executor...")
    action_recognizer = TSMActionRecognizer(args.model, args.device, refine_output=REFINE_OUTPUT, softmax_thresh=SOFTMAX_THRES, history_logit=HISTORY_LOGIT)
    detector = Detector(args.detection_model, args.device)
    tracker = Tracker(detector, TRACKER_SCORE_THRESHOLD, TRACKER_IOU_THRESHOLD)
    idx = 0
    print("Open camera...")
    cap = cv2.VideoCapture(0)

    # set a lower resolution for speed up
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, 7)
    cap.read()  # for settings to take place
    if args.stream == 'gui':
        # env variables
        full_screen = False
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 640, 480)
        cv2.moveWindow(WINDOW_NAME, 0, 0)
        cv2.setWindowTitle(WINDOW_NAME, WINDOW_NAME)

    light_controler = light_control.LightContoller()
    t = None

    if args.stream == 'video':
        fps = 7
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        demo_video = cv2.VideoWriter('demo.avi', fourcc, fps, (640, 528))

    os.system(f'v4l2-ctl -c auto_exposure=0')
    os.system(f'v4l2-ctl -c white_balance_auto_preset=1')
    t1 = time.time()
    i_frame = -1
    tracker_labels_map = dict()
    tracker_labels = set()
    last_caption = None
    active_object_id = -1
    print("Ready!")
    while True:
        try:
            i_frame += 1
            ret, img = cap.read()  # (480, 640, 3) 0 ~ 255
            if not ret:
                break
            detections, tracker_labels_map = tracker.add_frame(img, 1, tracker_labels_map)
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

                idx = action_recognizer(img, cur_det[0].roi.reshape(-1))
                print(f"{i_frame} {categories[idx]}")
            light_controler.encode_gesture(idx)
            if i_frame % 5 == 0:
                t2 = time.time()
                current_time = (t2 - t1) / 5
                t1 = t2
            if args.stream == 'none':
                continue

            img = cv2.resize(img, (640, 480))
            img = img[:, ::-1]
            height, width, _ = img.shape
            label = np.zeros([height // 10, width, 3]).astype('uint8') + 255

            cv2.putText(label, 'Prediction: ' + categories[idx],
                        (0, int(height / 16)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 0), 2)
            cv2.putText(label, '{:.1f} Vid/s'.format(1 / current_time),
                        (width - 170, int(height / 16)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 0), 2)

            img = np.concatenate((img, label), axis=0)
            if args.stream == 'gui':
                cv2.imshow(WINDOW_NAME, img)

                key = cv2.waitKey(1)
                if key & 0xFF == ord('q') or key == 27:  # exit
                    break
                elif key == ord('F') or key == ord('f'):  # full screen
                    print('Changing full screen option!')
                    full_screen = not full_screen
                    if full_screen:
                        print('Setting FS!!!')
                        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                              cv2.WINDOW_FULLSCREEN)
                    else:
                        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                              cv2.WINDOW_NORMAL)
            elif args.stream == 'video':
                demo_video.write(img)

        except KeyboardInterrupt:
            if args.stream == 'video':
                demo_video.release()
            cap.release()
            cv2.destroyAllWindows()
            light_controler.shutdown()


if __name__ == '__main__':
    main()
