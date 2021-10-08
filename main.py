"""
- Load model
- Read frame from camera (OpenCV)
- Convert to RGB
- Detect
"""

import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from mask_detection.utils import load_model, detect_fn, show_result

path_dict = {
    'checkpoint': 'configs/my_ckpt/ckpt-11', # Use last ckpt
    'pipeline': 'configs/custom.config',
    'label_map': 'configs/label_map.pbtxt'
}

categorical_index = label_map_util.create_category_index_from_labelmap(path_dict['label_map'])
model = load_model(path_dict)


vid = cv2.VideoCapture(0)

while True:
    ret, frame = vid.read()
    show_result(frame, model, categorical_index, enabled_warning=False)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()