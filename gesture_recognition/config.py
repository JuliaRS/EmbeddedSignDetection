DETECTOR_OUTPUT_SHAPE = -1, 5
TRACKER_SCORE_THRESHOLD = 0.4
TRACKER_IOU_THRESHOLD = 0.3
ACTION_NET_INPUT_FPS = 15
ACTION_IMAGE_SCALE = 256
SAMPLES_MAX_WINDOW_SIZE = 1000
SAMPLES_TRG_FPS = 20
VISUALIZER_TRG_FPS = 60
OBJECT_IDS = [ord(str(n)) for n in range(10)]

labels_map = ["Swiping Left",
"Swiping Right",
"Swiping Down",
"Swiping Up",
"Pushing Hand Away",
"Pulling Hand In",
"Sliding Two Fingers Left",
"Sliding Two Fingers Right",
"Sliding Two Fingers Down",
"Sliding Two Fingers Up",
"Pushing Two Fingers Away",
"Pulling Two Fingers In",
"Rolling Hand Forward",
"Rolling Hand Backward",
"Turning Hand Clockwise",
"Turning Hand Counterclockwise",
"Zooming In With Full Hand",
"Zooming Out With Full Hand",
"Zooming In With Two Fingers",
"Zooming Out With Two Fingers",
"Thumb Up",
"Thumb Down",
"Shaking Hand",
"Stop Sign",
"Drumming Fingers",
"No gesture",
"Doing other things"
]