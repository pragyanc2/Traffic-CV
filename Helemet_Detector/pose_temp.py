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

from helmet_class import classifier 
from fastai.vision import *
from pathlib import Path
from fastai.metrics import error_rate
from pathlib import Path

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

np.random.seed(2)
ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', type=str, default = "/home/pragyan/meow2.mp4",
                help = 'path to input video')
ap.add_argument('-c', '--config', type = str, default = "/home/pragyan/cv_project/darknet/object-detection-opencv/yolo_reqs/yolov3.cfg",
                help = 'path to yolo config file')
ap.add_argument('-w', '--weights', type=str, default = "/home/pragyan/cv_project/darknet/object-detection-opencv/yolo_reqs/yolov3.weights",
                help = 'path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes' , type=str, default = "/home/pragyan/cv_project/darknet/object-detection-opencv/yolo_reqs/yolov3.txt",
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
ap.add_argument('--debug', type=bool, default=False, help='set true to print debug checkpoints')
ap.add_argument('--frames', type=int, default=3,help = "Define frame shift")
args = ap.parse_args()

frame_shift = args.frames

def isHumanSittingOnBike (human_rect, head_rect, bike_rect, THRESHOLD = 0.7, HEAD_THRESH = 0.7):
    x_max = max(min(human_rect[0][0],human_rect[1][0]),min(bike_rect[0][0],bike_rect[1][0]))
    x2_min = min(max(human_rect[0][0],human_rect[1][0]),max(bike_rect[0][0],bike_rect[1][0]))

    human_right = max(human_rect[0][0],human_rect[1][0])
    human_left = min(human_rect[0][0],human_rect[1][0])

    bike_mid = (bike_rect[0][0] + bike_rect[1][0])/2

    if (bike_mid > human_right or bike_mid < human_left):
        return 0
    
    if max(human_rect[0][1],human_rect[1][1]) < min(bike_rect[0][1],bike_rect[0][1] - 10):
        return 0

    l_intersection = x2_min - x_max
    l_human = max(human_rect[0][0],human_rect[1][0]) - min(human_rect[0][0],human_rect[1][0])
    inter = 0
    if (l_human != 0):
        inter = l_intersection/l_human
    
    if (inter < THRESHOLD):
        inter = 0

    x_max_head = max(min(head_rect[0][0],head_rect[1][0]),min(bike_rect[0][0],bike_rect[1][0]))
    x2_min_head = min(max(head_rect[0][0],head_rect[1][0]),max(bike_rect[0][0],bike_rect[1][0]))

    l_inter_head = x2_min_head - x_max_head
    l_head = max(head_rect[0][0],head_rect[1][0]) - min(head_rect[0][0],head_rect[1][0])

    inter_head = 0
    if(l_head!=0):
        inter_head = l_inter_head/l_head

    if(inter_head < HEAD_THRESH):
        inter = 0
    
    return inter


def pose_func(args,size,yolo_stuff):
    with open(args.classes, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
    pose_stuff = []
    cam = cv2.VideoCapture(args.video)
    fps_time = 0

    (grabbed, frame) = cam.read()
    fshape = frame.shape
    fheight = fshape[0]
    fwidth = fshape[1]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_video = cv2.VideoWriter('pose_demo.avi', fourcc, 10, (fwidth,fheight))

    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    print(size)

    indexi = -1
    ii=0
    globalRiders = []

    while indexi < size-1:
        ret_val, image = cam.read()

        if(args.flip==True):
            image = cv2.flip(image,0)
            image = cv2.flip(image,1)
            
        ii+=frame_shift

        if ret_val is False:
            print("GENTLE REMINDER!!!! Video file not opened!! Please open it manually!")
            break
        
        cam.set(1, ii)
        indexi+=1
        humans = e.inference(image,resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        if not args.showBG:
            image = np.zeros(image.shape)

        faceRectangles = TfPoseEstimator.detect_face(image, humans, imgcopy=False)
        humanRectangles = TfPoseEstimator.detect_body(image, humans, imgcopy=False)

        outs = yolo_stuff[indexi]
        bikes = []
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        for out in outs:
            label, confidence, box = out
            x = box[0]-box[2]/2
            y = box[1]-box[3]/2
            w = box[2]
            h = box[3]
            points = ((round(x),round(y)),(round(x+w),round(y+h)))
            if label.decode('UTF-8')=='motorbike':
                bikes.append((points, confidence))
        
        humans_on_bikes = []
        for i, face in enumerate(faceRectangles):
            human = humanRectangles[i]
            whichBike = 0
            bikeScore = 0
            for bike in bikes:
                if isHumanSittingOnBike(human, face, bike[0]) > bikeScore:
                    whichBike = bike
                    bikeScore = isHumanSittingOnBike(human, face, bike[0])
            if (bikeScore != 0):
                humans_on_bikes.append((human, face, whichBike))
                cv2.rectangle(image,face[0],face[1],(200,50,0),5)
                # cv2.rectangle(image,human[0],human[1],(255,0,255),3)
                cv2.rectangle(image,whichBike[0][0],whichBike[0][1],(200,50,0),5)
                           

        # THIS PIECE OF CODE IS TO FOR TESTING PURPOSES
        # for ed in humans_on_bikes:
        #     human, face, bike = ed
        #     imc = image.copy()
        #     draw_prediction(imc, 3, bike[1], bike[0][0][0], bike[0][0][1], bike[0][1][0], bike[0][1][1], classes, COLORS)
        #     cv2.rectangle(imc,face[0],face[1],(0,255,0),3)
        #     cv2.rectangle(imc,human[0],human[1],(0,255,0),3)                
        #     cv2.imshow("object detection", imc)
        #     a = cv2.waitKey(0) 
        #     if a == 27:
        #         break
        #     else:
        #         continue

        cv2.putText(image,
                        "Progress: %f" % (100*indexi/size),
                        (20, 20),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (9, 255, 9), 2)
        # cv2.imwrite("poster_face.png", image)
        globalRiders.append(humans_on_bikes)        
        if (args.debug):
            print("cry")
        cv2.imshow("object detection", image)
        out_video.write(image)
        if cv2.waitKey(1) == 27:
            break
        pose_stuff.append((faceRectangles, humanRectangles))

    #cv2.imwrite("object-detection.`jpg", image)
    cv2.destroyAllWindows()
    # print(len(pose_stuff))
    return indexi, pose_stuff, globalRiders 

if __name__ == "__main__":
    data_file = open(r'/home/pragyan/cv_project/darknet/object-detection-opencv/stuff_data/delete_later/data_stuff.pkl', 'rb')
    size, yolo_stuff =  pickle.load(data_file)

    size, pose_stuff, globalRiders = pose_func(args,size,yolo_stuff)
    # data_file = open(r'/home/pragyan/cv_project/darknet/object-detection-opencv/stuff_data/delete_later/pose_stuff.pkl', 'wb')
    pickle.dump((size,pose_stuff, globalRiders),data_file)