#############################################
# Object detection - YOLO - OpenCV
# Author : Arun Ponnusamy   (July 16, 2018)
# Website : http://www.arunponnusamy.com
############################################

import cv2
import argparse
import logging
import numpy as np
import time
import pickle
from ctypes import *
import math
import random

from helmet_class import classifier 
from fastai.vision import *
from pathlib import Path
from fastai.metrics import error_rate
from pathlib import Path

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
np.random.seed(2)
ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', required=True,
                help = 'path to input video')
ap.add_argument('-c', '--config', required=True,
                help = 'path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help = 'path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help = 'path to text file containing class names')
ap.add_argument('--resize-out-ratio', type=float, default=4.0,
        help='if provided, resize heatmaps before they are post-processed. default=1.0')
ap.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
ap.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
ap.add_argument('--model', type=str, default='mobilenet_v2_large', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
ap.add_argument('--show-process', type=bool, default=False,
                    help='for debug purpose, if enabled, speed for inference is dropped.')
ap.add_argument('--showBG', type=bool, default=True, help='False to show skeleton only.')
ap.add_argument('--flip', type=bool, default=False, help='Flip.')
args = ap.parse_args()

frame_shift = 12






def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

    

#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = load_image(image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms)

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (round(b.x), round(b.y), round(b.w), round(b.h))))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res

def darknet_func(im_path, args):
    net = load_net(args.config.encode('UTF-8'), args.weights.encode('UTF-8'), 0)
    meta = load_meta(b"cfg/coco.data")
    r = detect(net, meta, im_path.encode('UTF-8'))
    return r


def yolo_func(args):
    cam = cv2.VideoCapture(args.video)
    length = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))

    fps_time = 0
    i = 0
    size = 0

    yolo_stuff = []

    while(True):

        ret_val, image = cam.read()

        if(args.flip==True):
            image = cv2.flip(image,0)
            image = cv2.flip(image,1)

        i+=frame_shift

        # if i%frame_shift!=0:
        #     continue

        if ret_val is False:
            print("Video File not opened!")
            break

        COLORS = np.random.uniform(0, 255, size=(30, 3))

        cam.set(1, i)

        cv2.imwrite("utilimg.jpg", image)
        im_path = "utilimg.jpg"
        

        outs  = darknet_func(im_path,args)
        
        for out in outs:
            class_name, confidence, box = out
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            cv2.rectangle(image, (x,y), (x+w,y+h), COLORS[round(np.random.uniform(0,30))],3)

        yolo_stuff.append(outs)

        cv2.imshow("object detection", image)
        if cv2.waitKey(1) == 27:
            break

    cv2.imwrite("object-detection.jpg", image)
    cv2.destroyAllWindows()
    return yolo_stuff, size

if __name__ == "__main__": 
    yolo_stuff, size = yolo_func(args)
    data_str = (size, yolo_stuff)
    data_file = open(r'/home/pragyan/cv_project/darknet/object-detection-opencv/stuff_data/data_stuff.pkl', 'wb')
    pickle.dump(data_str, data_file)