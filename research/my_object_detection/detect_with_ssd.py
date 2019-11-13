# -*- coding: utf-8 -*-
"""
Usage:
  # detect one image using trained model:
  python detect_with_ssd.py --model=/train_model/xxx.pb --label=xxx_label_map.pbtxt --mode=image --path=/path/to/image

  # detect video using trained model:
  python detect_with_ssd.py --model=/train_model/xxx.pb --label=xxx_label_map.pbtxt --mode=video --path=/path/to/video
"""
import cv2
import numpy as np
import tensorflow as tf
import time
import os
import argparse
import utils.label_map_util as label_map_util
from object_detection.utils import visualization_utils as vis_util

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='path to model.pb')
parser.add_argument('--label', type=str, help='path to label_map.pbtxt')
parser.add_argument('--mode', type=str, help='detect mode: image or video')
parser.add_argument('--path', type=str, help='path to image or video')
args = parser.parse_args()


class TOD(object):
    def __init__(self):
        self.PATH_TO_CKPT = args.model
        self.PATH_TO_LABEL = args.label
        self.NUM_CLASSES = 1
        self.detection_graph = self._load_model()
        self.category_index = self._load_label_map()

    def _load_model(self):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        # print('model has been loaded!')
        return detection_graph

    def _load_label_map(self):
        label_map = label_map_util.load_labelmap(self.PATH_TO_LABEL)
        categories = label_map_util.convert_label_map_to_categories(label_map,
                                                                    self.NUM_CLASSES,
                                                                    True)
        category_index = label_map_util.create_category_index(categories)
        return category_index

    def detect(self, image):
        # with tf.device('/gpu:0'):
        with self.detection_graph.as_default():
            with tf.compat.v1.Session(graph=self.detection_graph) as sess:
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image, axis=0)
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
                # Actual detection.
                start = time.time()
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                end = time.time()
                # print(np.squeeze(boxes))
                print(np.squeeze(scores))
                print('Running time: %s Seconds' % (end - start))
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    self.category_index,
                    use_normalized_coordinates=True,
                    line_thickness=2)
        cv2.imshow("detection", image)
        cv2.waitKey(0)

    def detect_video(self, video_path):
        self.vc = cv2.VideoCapture(video_path)
        # print('video has been loaded!')
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
                # print('----------1----------')
                while True:
                    ret, frame = self.vc.read()
                    if not ret:
                        break
                    # image size: 2048 * 1536, 增加边界扩充为2048 * 2048（根据样本不同可以修改该句）
                    frame = cv2.copyMakeBorder(frame, 256, 256, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                    frame = cv2.resize(frame, (512, 512), cv2.INTER_CUBIC)
                    image_np_expanded = np.expand_dims(frame, axis=0)
                    (boxes, scores, classes, nums) = sess.run(
                        [boxes, scores, classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})
                    # length = len(boxes)
                    # for i, box in zip(range(length), boxes):
                    #     if scores[i] >= 0.1:
                    #         cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), thickness=2)
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        frame,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        self.category_index,
                        use_normalized_coordinates=True,
                        line_thickness=2)
                    cv2.imshow('Frame', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break


if __name__ == '__main__':
    detector = TOD()
    if args.mode == 'video':
        detector.detect_video(args.path)
    elif args.mode == 'image':
        img = cv2.imread(args.path)
        board = cv2.copyMakeBorder(img, 256, 256, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        img = cv2.resize(img, (512, 512), cv2.INTER_CUBIC)
        detector.detect(img)
    # for img in os.listdir('val'):
    #     if '.jpg' in img:
    #         img_path = os.path.join('val', img)
    #         image = cv2.imread(img_path)
    #         board = cv2.copyMakeBorder(image, 256, 256, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    #         image = cv2.resize(image, (512, 512), cv2.INTER_CUBIC)
    #         detector.detect(image)
    # detector.detect_video('video/test1.avi')
