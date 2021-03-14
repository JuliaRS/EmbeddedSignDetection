import cv2
import numpy as np
from common_onnx import ONNXModel

def process_output(idx_, history, refine_output=True):
    # idx_: the output of current frame
    # history: a list containing the history of predictions
    if not refine_output:
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

class TSMActionRecognizer(ONNXModel):
    def __init__(self, model, device, img_scale=256, num_classes=27, refine_output=True, softmax_thresh=0.8, history_logit=True):
        super().__init__(model, device)
        self.buffer = [
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
        self.input_height = 224
        self.input_width = 224
        self.img_scale = img_scale
        self.num_test_classes = num_classes
        self.history = [2]
        self.history_logit = []
        self.history_timing = []
        self.softmax_thresh = softmax_thresh
        self.use_history_logit = history_logit
        self.refine_output = refine_output

    @staticmethod
    def _convert_to_central_roi(src_roi, input_height, input_width, img_scale):
        """Extracts from the input ROI the central square part with specified side size"""

        src_roi_height, src_roi_width = src_roi[3] - src_roi[1], src_roi[2] - src_roi[0]
        src_roi_center_x = 0.5 * (src_roi[0] + src_roi[2])
        src_roi_center_y = 0.5 * (src_roi[1] + src_roi[3])

        height_scale = float(input_height) / float(img_scale)
        width_scale = float(input_width) / float(img_scale)
        assert height_scale < 1.0
        assert width_scale < 1.0

        min_roi_size = min(src_roi_height, src_roi_width)
        trg_roi_height = int(height_scale * min_roi_size)
        trg_roi_width = int(width_scale * min_roi_size)

        trg_roi = [int(src_roi_center_x - 0.5 * trg_roi_width),
                   int(src_roi_center_y - 0.5 * trg_roi_height),
                   int(src_roi_center_x + 0.5 * trg_roi_width),
                   int(src_roi_center_y + 0.5 * trg_roi_height)]

        return trg_roi

    def _process_image(self, input_image, roi):
        """Converts input image according to model requirements"""

        cropped_image = input_image[roi[1]:roi[3], roi[0]:roi[2]]
        resized_image = cv2.resize(cropped_image, (self.input_width, self.input_height))
        out_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        out_image /= 255
        out_image -= np.array([0.406, 0.456, 0.485])
        out_image /= np.array([0.225, 0.224, 0.229])
        out_image = np.transpose(out_image, axes=[2, 0, 1])  # (3, 224, 224) 0 ~ 1.0
        out_image = np.expand_dims(out_image, axis=0)  # (1, 3, 480, 640) 0 ~ 1.0
        return out_image

    def __call__(self, frame, person_roi=None):
        """Runs model on the specified input"""

        if person_roi is None:
            person_roi = [0, 0, frame.shape[0], frame.shape[1]]
        central_roi = self._convert_to_central_roi(person_roi,
                                                   self.input_height, self.input_width,
                                                   self.img_scale)
        data = self._process_image(frame, central_roi)
        input_data = [data] + self.buffer
        results = self.infer(input_data)
        self.buffer = results[1:]
        return self._process_output(results[0])

    def _process_output(self, feat):
        if self.softmax_thresh > 0:
            feat_np = np.reshape(feat, -1)
            feat_np -= feat_np.max()
            softmax = np.exp(feat_np) / np.sum(np.exp(feat_np))
            if max(softmax) > self.softmax_thresh:
                idx_ = np.argmax(feat, axis=1)[0]
            else:
                idx_ = self.history[-1]
        else:
            idx_ = np.argmax(feat, axis=1)[0]

        if self.use_history_logit:
            self.history_logit.append(feat)
            self.history_logit = self.history_logit[-12:]
            avg_logit = sum(self.history_logit)
            idx_ = np.argmax(avg_logit, axis=1)[0]

        idx, self.history = process_output(idx_, self.history, self.refine_output)

        return idx
