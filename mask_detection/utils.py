import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import logging
import threading
import time
import pyttsx3


def load_model(path_dict):
    config = config_util.get_configs_from_pipeline_file(path_dict['pipeline'])
    model = model_builder.build(config['model'], is_training=False)
    ckpt = tf.compat.v2.train.Checkpoint(model=model)
    ckpt.restore(path_dict['checkpoint']).expect_partial()
    return model


@tf.function
def detect_fn(image_np, model):
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = tf.expand_dims(input_tensor, axis=0)
    input_tensor = tf.cast(input_tensor, tf.float32)
    preprocessed_tensor, shape = model.preprocess(input_tensor)
    prediction_dict = model.predict(preprocessed_tensor, shape)
    results = model.postprocess(prediction_dict, shape)
    return results


def check_need_mask(results, threshold=0.6):
    need_mask = False
    scores = tf.convert_to_tensor(results['detection_scores'])
    boxes = tf.convert_to_tensor(results['detection_boxes'])
    labels = tf.convert_to_tensor(results['detection_classes'])
    selected_indices = tf.image.non_max_suppression(boxes, scores, max_output_size=10,
                                                    score_threshold=threshold)
    boxes = tf.gather(boxes, selected_indices)
    labels = tf.gather(labels, selected_indices)

    # Find without mask
    results = []
    for label, bbox in zip(labels.numpy().astype(int), boxes.numpy()):
        if label != 1:
            need_mask = True
        results.append((label, bbox))
    return need_mask, results


engine = pyttsx3.init()
engine.setProperty("rate", 150)


def speak():
    global engine
    text = "You are not wearing a mask or wearing a mask incorrectly. Please put a mask on."
    engine.say(text)
    engine.runAndWait()


def show_result(img, model, categorical_index, enabled_warning=False):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_np = np.array(rgb)
    results = detect_fn(img_np, model)
    num_detections = int(results.pop('num_detections'))
    results = {key: value[0, :num_detections].numpy()
               for key, value in results.items()}

    need_mask, filtered_results = check_need_mask(results)

    speaking = False

    def speak_thread():
        global speaking
        logging.info("Thread starting")
        speaking = True
        speak()
        time.sleep(2)
        speaking = False
        logging.info("Thread finishing")

    if need_mask and not speaking and enabled_warning:
        x = threading.Thread(target=speak_thread)
        x.start()

    img_with_bbox = viz_utils.visualize_boxes_and_labels_on_image_array(
        img,
        boxes=results['detection_boxes'],
        classes=results['detection_classes'].astype(int) + 1,
        scores=results['detection_scores'],
        category_index=categorical_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=15,
        min_score_thresh=0.6,
        agnostic_mode=False,
    )
    cv2.imshow('frame', img)
