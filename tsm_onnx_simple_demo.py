import os
from typing import Tuple
import io
import time
import argparse
import numpy as np
import cv2
import onnxruntime.backend as backend
import onnx
import light_control

SOFTMAX_THRES = 0.8
HISTORY_LOGIT = True
REFINE_OUTPUT = True
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

    return parser


def get_executor(model):
    beckend_rep = backend.prepare(model=str(model))
    sess = beckend_rep._session
    inputs_info = sess.get_inputs()
    input_names = [input_layer.name for input_layer in inputs_info]
    outputs = sess.get_outputs()
    output_names = [output.name for output in outputs]
    return sess, input_names, output_names


def transform(frame: np.ndarray):
    # 480, 640, 3, 0 ~ 255
    frame = cv2.resize(frame, (224, 224))  # (224, 224, 3) 0 ~ 255
    frame = frame.astype(np.float32) / 255.0  # (224, 224, 3) 0 ~ 1.0
    frame -= np.array([0.485, 0.456, 0.406])
    frame /= np.array([0.229, 0.224, 0.225])
    frame = np.transpose(frame, axes=[2, 0, 1])  # (3, 224, 224) 0 ~ 1.0
    frame = np.expand_dims(frame, axis=0)  # (1, 3, 480, 640) 0 ~ 1.0
    return frame


def process_output(idx_, history):
    # idx_: the output of current frame
    # history: a list containing the history of predictions
    if not REFINE_OUTPUT:
        return idx_, history

    max_hist_len = 20  # max history buffer

    # mask out illegal action
    if idx_ in [7, 8, 21, 22, 3]:
        idx_ = history[-1]

    # use only single no action class
    if idx_ == 0:
        idx_ = 2

    # history smoothing
    if idx_ != history[-1]:
        if not (history[-1] == history[-2]):  # and history[-2] == history[-3]):
            idx_ = history[-1]

    history.append(idx_)
    history = history[-max_hist_len:]

    return history[-1], history


def main():
    args = build_argparser().parse_args()
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
    print("Build transformer...")
    print("Build Executor...")
    executor, input_names, output_names = get_executor(args.model)
    buffer = [
        np.zeros((1, 3, 56, 56), dtype=np.float32),
        np.zeros((1, 4, 28, 28), dtype=np.float32),
        np.zeros((1, 4, 28, 28), dtype=np.float32),
        np.zeros((1, 8, 14, 14), dtype=np.float32),
        np.zeros((1, 8, 14, 14), dtype=np.float32),
        np.zeros((1, 8, 14, 14), dtype=np.float32),
        np.zeros((1, 12, 14, 14), dtype=np.float32),
        np.zeros((1, 12, 14, 14), dtype=np.float32),
        np.zeros((1, 20, 7, 7), dtype=np.float32),
        np.zeros((1, 20, 7, 7), dtype=np.float32)
    ]
    idx = 0
    history = [2]
    history_logit = []
    history_timing = []

    if args.stream == 'video':
        fps = 7
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        demo_video = cv2.VideoWriter('demo.avi', fourcc, fps, (640, 528))

    os.system(f'v4l2-ctl -c auto_exposure=0')
    os.system(f'v4l2-ctl -c white_balance_auto_preset=1')
    t1 = time.time()
    i_frame = -1
    print("Ready!")
    while True:
        try:
            i_frame += 1
            ret, img = cap.read()  # (480, 640, 3) 0 ~ 255
            if not ret:
                break
            img_tran = transform(img)
            inputs = [img_tran] + buffer
            outputs = executor.run(output_names, dict(zip(input_names, inputs)))
            feat, buffer = outputs[0], outputs[1:]

            if SOFTMAX_THRES > 0:
                feat_np = np.reshape(feat, -1)
                feat_np -= feat_np.max()
                softmax = np.exp(feat_np) / np.sum(np.exp(feat_np))
                if max(softmax) > SOFTMAX_THRES:
                    idx_ = np.argmax(feat, axis=1)[0]
                else:
                    idx_ = idx
            else:
                idx_ = np.argmax(feat, axis=1)[0]

            if HISTORY_LOGIT:
                history_logit.append(feat)
                history_logit = history_logit[-12:]
                avg_logit = sum(history_logit)
                idx_ = np.argmax(avg_logit, axis=1)[0]

            idx, history = process_output(idx_, history)

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
