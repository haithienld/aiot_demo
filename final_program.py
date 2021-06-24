# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import collections
import common
import cv2
import numpy as np
import os
from PIL import Image
import re
import tflite_runtime.interpreter as tflite
from tflite_runtime.interpreter import load_delegate

# import the necessary packages
import face_recognition
import pickle
import argparse
import cv2
from imutils.video import VideoStream
import imutils
import time

#===============================================

Object = collections.namedtuple('Object', ['id', 'score', 'bbox'])

def load_labels(path):
    p = re.compile(r'\s*(\d+)(.+)')
    with open(path, 'r', encoding='utf-8') as f:
       lines = (p.match(line).groups() for line in f.readlines())
       return {int(num): text.strip() for num, text in lines}

class BBox(collections.namedtuple('BBox', ['xmin', 'ymin', 'xmax', 'ymax'])):

    __slots__ = ()

def get_output(interpreter, score_threshold, top_k, image_scale=1.0):
    """Returns list of detected objects."""
    boxes = common.output_tensor(interpreter, 0)
    class_ids = common.output_tensor(interpreter, 1)
    scores = common.output_tensor(interpreter, 2)
    count = int(common.output_tensor(interpreter, 3))

    def make(i):
        ymin, xmin, ymax, xmax = boxes[i]
        return Object(
            id=int(class_ids[i]),
            score=scores[i],
            bbox=BBox(xmin=np.maximum(0.0, xmin),
                      ymin=np.maximum(0.0, ymin),
                      xmax=np.minimum(1.0, xmax),
                      ymax=np.minimum(1.0, ymax)))

    return [make(i) for i in range(top_k) if scores[i] >= score_threshold]

def decode_bbox(anchors, raw_outputs, variances=[0.1, 0.1, 0.2, 0.2]):
    '''
    Decode the actual bbox according to the anchors.
    the anchor value order is:[xmin,ymin, xmax, ymax]
    :param anchors: numpy array with shape [batch, num_anchors, 4]
    :param raw_outputs: numpy array with the same shape with anchors
    :param variances: list of float, default=[0.1, 0.1, 0.2, 0.2]
    :return:
    '''
    anchor_centers_x = (anchors[:, :, 0:1] + anchors[:, :, 2:3]) / 2
    anchor_centers_y = (anchors[:, :, 1:2] + anchors[:, :, 3:]) / 2
    anchors_w = anchors[:, :, 2:3] - anchors[:, :, 0:1]
    anchors_h = anchors[:, :, 3:] - anchors[:, :, 1:2]
    raw_outputs_rescale = raw_outputs * np.array(variances)
    predict_center_x = raw_outputs_rescale[:, :, 0:1] * anchors_w + anchor_centers_x
    predict_center_y = raw_outputs_rescale[:, :, 1:2] * anchors_h + anchor_centers_y
    predict_w = np.exp(raw_outputs_rescale[:, :, 2:3]) * anchors_w
    predict_h = np.exp(raw_outputs_rescale[:, :, 3:]) * anchors_h
    predict_xmin = predict_center_x - predict_w / 2
    predict_ymin = predict_center_y - predict_h / 2
    predict_xmax = predict_center_x + predict_w / 2
    predict_ymax = predict_center_y + predict_h / 2
    predict_bbox = np.concatenate([predict_xmin, predict_ymin, predict_xmax, predict_ymax], axis=-1)
    return predict_bbox

def generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios, offset=0.5):
    '''
    generate anchors.
    :param feature_map_sizes: list of list, for example: [[40,40], [20,20]]
    :param anchor_sizes: list of list, for example: [[0.05, 0.075], [0.1, 0.15]]
    :param anchor_ratios: list of list, for example: [[1, 0.5], [1, 0.5]]
    :param offset: default to 0.5
    :return:
    '''
    anchor_bboxes = []
    for idx, feature_size in enumerate(feature_map_sizes):
        cx = (np.linspace(0, feature_size[0] - 1, feature_size[0]) + 0.5) / feature_size[0]
        cy = (np.linspace(0, feature_size[1] - 1, feature_size[1]) + 0.5) / feature_size[1]
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid_expend = np.expand_dims(cx_grid, axis=-1)
        cy_grid_expend = np.expand_dims(cy_grid, axis=-1)
        center = np.concatenate((cx_grid_expend, cy_grid_expend), axis=-1)

        num_anchors = len(anchor_sizes[idx]) +  len(anchor_ratios[idx]) - 1
        center_tiled = np.tile(center, (1, 1, 2* num_anchors))
        anchor_width_heights = []

        # different scales with the first aspect ratio
        for scale in anchor_sizes[idx]:
            ratio = anchor_ratios[idx][0] # select the first ratio
            width = scale * np.sqrt(ratio)
            height = scale / np.sqrt(ratio)
            anchor_width_heights.extend([-width / 2.0, -height / 2.0, width / 2.0, height / 2.0])

        # the first scale, with different aspect ratios (except the first one)
        for ratio in anchor_ratios[idx][1:]:
            s1 = anchor_sizes[idx][0] # select the first scale
            width = s1 * np.sqrt(ratio)
            height = s1 / np.sqrt(ratio)
            anchor_width_heights.extend([-width / 2.0, -height / 2.0, width / 2.0, height / 2.0])

        bbox_coords = center_tiled + np.array(anchor_width_heights)
        bbox_coords_reshape = bbox_coords.reshape((-1, 4))
        anchor_bboxes.append(bbox_coords_reshape)
    anchor_bboxes = np.concatenate(anchor_bboxes, axis=0)
    return anchor_bboxes

def single_class_non_max_suppression(bboxes, confidences, conf_thresh=0.2, iou_thresh=0.5, keep_top_k=-1):
    '''
    do nms on single class.
    Hint: for the specific class, given the bbox and its confidence,
    1) sort the bbox according to the confidence from top to down, we call this a set
    2) select the bbox with the highest confidence, remove it from set, and do IOU calculate with the rest bbox
    3) remove the bbox whose IOU is higher than the iou_thresh from the set,
    4) loop step 2 and 3, util the set is empty.
    :param bboxes: numpy array of 2D, [num_bboxes, 4]
    :param confidences: numpy array of 1D. [num_bboxes]
    :param conf_thresh:
    :param iou_thresh:
    :param keep_top_k:
    :return:
    '''
    if len(bboxes) == 0: return []

    conf_keep_idx = np.where(confidences > conf_thresh)[0]

    bboxes = bboxes[conf_keep_idx]
    confidences = confidences[conf_keep_idx]

    pick = []
    xmin = bboxes[:, 0]
    ymin = bboxes[:, 1]
    xmax = bboxes[:, 2]
    ymax = bboxes[:, 3]

    area = (xmax - xmin + 1e-3) * (ymax - ymin + 1e-3)
    idxs = np.argsort(confidences)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # keep top k
        if keep_top_k != -1:
            if len(pick) >= keep_top_k:
                break

        overlap_xmin = np.maximum(xmin[i], xmin[idxs[:last]])
        overlap_ymin = np.maximum(ymin[i], ymin[idxs[:last]])
        overlap_xmax = np.minimum(xmax[i], xmax[idxs[:last]])
        overlap_ymax = np.minimum(ymax[i], ymax[idxs[:last]])
        overlap_w = np.maximum(0, overlap_xmax - overlap_xmin)
        overlap_h = np.maximum(0, overlap_ymax - overlap_ymin)
        overlap_area = overlap_w * overlap_h
        overlap_ratio = overlap_area / (area[idxs[:last]] + area[i] - overlap_area)

        need_to_be_deleted_idx = np.concatenate(([last], np.where(overlap_ratio > iou_thresh)[0]))
        idxs = np.delete(idxs, need_to_be_deleted_idx)

    # if the number of final bboxes is less than keep_top_k, we need to pad it.
    # TODO
    return conf_keep_idx[pick]
def det_and_display(cv2_im, interpreter, labels, threshold):
    # cv2_im_float = cv2_im.astype(np.float32)
    # cv2_im_float = cv2_im
    img_orig_shape = cv2_im.shape
    cv2_im = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
    cv2_im_rgb = cv2.resize(cv2_im, (260,260))
    cv2_im_rgb = cv2_im_rgb/ 255.0
    cv2_im_rgb = np.expand_dims(cv2_im_rgb,0).astype(np.float32)
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    interpreter.set_tensor(input_details[0]['index'], cv2_im_rgb)
    #common.set_input(interpreter, cv2_im_rgb)

    interpreter.invoke()

    # return final
    feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
    anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
    anchor_ratios = [[1, 0.62, 0.42]] * 5
    # generate anchors
    anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)

    # for inference , the batch size is 1, the model output shape is [1, N, 4],
    # so we expand dim for anchors to [1, anchor_num, 4]
    anchors_exp = np.expand_dims(anchors, axis=0)
    output_data1 = interpreter.get_tensor(output_details[0]['index'])
    output_data2 = interpreter.get_tensor(output_details[1]['index'])
    #print(output_data1.shape)
    # print(output_data1[1],output_data2[1])
    y_bboxes = decode_bbox(anchors_exp, output_data2)[0]
    y_cls = output_data1[0]
    # To speed up, do single class NMS, not multiple classes NMS.
    bbox_max_scores = np.max(y_cls, axis=1)
    bbox_max_score_classes = np.argmax(y_cls, axis=1)

    # keep_idx is the alive bounding box after nms.
    keep_idxs = single_class_non_max_suppression(y_bboxes,
                                                 bbox_max_scores,
                                                 conf_thresh=threshold,
                                                 iou_thresh=0.6,
                                                 )
    boxes =[]
    for idx in keep_idxs:
        conf = float(bbox_max_scores[idx])
        class_id = bbox_max_score_classes[idx]
        bbox = y_bboxes[idx]
        # clip the coordinate, avoid the value exceed the image boundary.
        xmin = max(0, int(bbox[0] * img_orig_shape[1]))
        ymin = max(0, int(bbox[1] * img_orig_shape[0]))
        xmax = min(int(bbox[2] * img_orig_shape[1]), img_orig_shape[1])
        ymax = min(int(bbox[3] * img_orig_shape[0]), img_orig_shape[0])
        boxes.append((ymin,xmax,ymax,xmin))
        # compute area scale of bounding box with image
        width = img_orig_shape[1]
        height = img_orig_shape[0]
        bbox_area = ((xmax-xmin)/width) * ((ymax-ymin)/height)
        if bbox_area < 0.005:
            continue

        if class_id == 1:
            color = (255, 0, 0)
        else:
            color = (0, 0, 255)
        cv2.rectangle(cv2_im, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(cv2_im, "%s: %.2f" % (labels[class_id], conf), (xmin + 2, ymin - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color)

    #final = cv2.resize(cv2_im, (600,600))
    return cv2_im,boxes


print('Loading {} with {} labels.'.format('models/face_mask_detector_tpu.tflite', 'models/mask.txt'))
#interpreter = common.make_interpreter(args.model)
interpreter = tflite.Interpreter(model_path='models/face_mask_detector_tpu.tflite', experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
interpreter.allocate_tensors()
labels = load_labels('models/mask.txt')

cap = cv2.VideoCapture(0)
out = cv2.VideoWriter(str(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f'))+'.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (640,480))
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cv2_im = frame
    
    out.write(cv2_im) 
    start = time.perf_counter()
    cv2_im,boxes = det_and_display(cv2_im, interpreter, labels, 0.5)
    cv2_im = cv2.cvtColor(cv2_im, cv2.COLOR_RGB2BGR)
    print(boxes)
    
    encodings = face_recognition.face_encodings(cv2_im, boxes)

    names = []
    data = pickle.loads(open('encodings/facial_encodings.pkl', "rb").read())
    # loop over the facial embeddings
    for encoding in encodings:
        # attempt to match each face in the input image to our known encodings
        matches = face_recognition.compare_faces(data['encodings'], encoding, tolerance=0.55)
        name = "Unknown"

        # check to see if we have found any matches
        if True in matches:
            # find the indexes of all matched faces then initialize a dictionary to count
            # the total number of times each face was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # loop over the matched indexes and maintain a count for each recognized face face
            for i in matchedIdxs:
                name = data['names'][i]
                counts[name] = counts.get(name, 0) + 1

            # determine the recognized face with the largest number of votes: (notes: in the event of an unlikely
            # tie, Python will select first entry in the dictionary)
            name = max(counts, key=counts.get)
        names.append(name)
    inference_time = time.perf_counter() - start
    print('%.1fms' % (inference_time * 1000))
    # loop over the recognized faces
    for ((top, right, bottom, left), name) in zip(boxes, names):
        # rescale the face coordinates
        top = int(top)
        right = int(right)
        bottom = int(bottom)
        left = int(left)

        # draw the predicted face name on the image
        cv2.rectangle(cv2_im, (left, top), (right, bottom),
                    (0, 255, 0), 2)

        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(cv2_im, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 255, 0), 2)
    
    
    
    
    cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("frame",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cv2.imshow('frame', cv2_im)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
