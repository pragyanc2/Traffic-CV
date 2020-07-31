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
ap.add_argument('--save', type=bool, default=False, help='Save.')
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
    
    # print(inter,inter_head)
    return inter


# cam1 = cv2.VideoCapture(args.video)

def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, label, confidence, x, y, x_plus_w, y_plus_h, classes, COLORS):

    # label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def helmet_func(args,pose_stuff,yolo_stuff,size, globalRiders):
    with open(args.classes, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    cam1 = cv2.VideoCapture(args.video)
    fps_time = 0
    path = Path('/home/pragyan/cv_project/darknet/object-detection-opencv/classifier-models')
    hel = load_learner(path, 'export18bs32.pkl')

    indexi = -1
    ii=0
    
    ii=-1
    (grabbed, frame) = cam1.read()
    fshape = frame.shape
    fheight = fshape[0]
    fwidth = fshape[1]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_video = cv2.VideoWriter('output.avi', fourcc, 10, (fwidth,fheight))

    #print(len(pose_stuff), len(yolo_stuffi), size)
    while(indexi<size-1):

        ret_val, image = cam1.read()
        ii+=frame_shift

        if ret_val is False:
            print("BLAH")
            break
        cam1.set(1, ii)
        indexi+=1

        if(args.flip==True):
            image = cv2.flip(image,0)
            image = cv2.flip(image,1)

        # outs = yolo_stuff[indexi] 
        # rectanges, bodyRects = pose_stuff[indexi]
        
        #for faceRect in rectanges:
            # cv2.rectangle(image,faceRect[0],faceRect[1],(255,0,0),3)


        # indices = yolo_stuff[0]
        # class_ids = yolo_stuff[1]
        # confidences = yolo_stuff[2]
        # boxes = yolo_stuff[3]

        # for out in outs:
        #     if(args.save==False):
        #         label, confidence, box = out
        #         x = box[0]
        #         y = box[1]
        #         w = box[2]
        #         h = box[3]
        #         points = ((round(x),round(y)),(round(x+w),round(y+h)))
        #         if (label.decode('UTF-8') == "motorcycle"):
        #             draw_prediction(image, "motorcycle", confidence, round(x), round(y), round(x+w), round(y+h), classes, COLORS)
        #     else:
        #         bike_img = image[x:x+w,y:y+h]
        #         cv2.imwrite("./dataset_classifier/bikes/{}_{}.jpg".format(i,ii),img2)

        # print(indexi,box)
        for index, riderBike in enumerate(globalRiders[indexi]):
            human, face, biket = riderBike
            bike = biket[0]
            
            # print("BLAH")
            if (isHumanSittingOnBike(human,face,bike)==0):
                continue
            # cv2.rectangle(image,face[0],face[1],(0,255,0),3)
            image2 = image[min(face[0][1],face[1][1]):max(face[0][1],face[1][1])+1,min(face[0][0],face[1][0]):max(face[0][0],face[1][0])+1]
            # img2 = image2.copy()
            img = Image(pil2tensor(image2, dtype = np.float32).div_(255))
            rvar = hel.predict(img)
            # print(rvar)
            narr = rvar[2].numpy()

            # print(bike)
            # print(narr)
            
            if (narr[0] > 0.97):
                if(args.save==False):
                    cv2.rectangle(image,face[0],face[1],(0,255,0),3)
                    cv2.putText(image,
                            "Prediction: Helmet{}".format(index),
                            bike[0],  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2)
                    cv2.rectangle(image,bike[0],bike[1],(0,255,0),3)
                    cv2.putText(image,
                            "Prediction: Helmet{}".format(index),
                            face[0],  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2)

                else:
                    cv2.imwrite("./dataset_classifier/pos/{}_{}.jpg".format(index,ii),img2)
            elif(narr[1] > 0.97):
                bike_image = image[bike[0][1]:bike[1][1],bike[0][0]:bike[1][0]]
                cv2.imwrite("./dataset_classifier/bike_neg/{}_{}.jpg".format(index,ii),bike_image)
                if(args.save==False):
                    cv2.rectangle(image,face[0],face[1],(0,0,255),3)
                    cv2.putText(image,
                            "Prediction: Without Helmet{}".format(index), 
                            bike[0],  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 2)
                    cv2.rectangle(image,bike[0],bike[1],(0,0,255),3)
                    cv2.putText(image,
                            "Prediction: Without Helmet{}".format(index), 
                            face[0],  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 2)
                else:
                    cv2.imwrite("./dataset_classifier/neg/{}_{}.jpg".format(index,ii),img2)

            # cv2.imshow("object detection", image)
            # if cv2.waitKey(0) == 27:
            #     break
            # else:
            #     continue

        cv2.putText(image,
                    "Progress: %f" % (100*indexi/size),
                    (20, 20),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (9, 255, 9), 2)
        out_video.write(image)
        cv2.imwrite("pose_helmet.jpg", image)
        cv2.imshow("object detection", image)
        # cv2.imshow("stuff", tempimage)
        if cv2.waitKey(1) == 27:
            break

    # cv2.imwrite("object-detection.jpg", image)
    cv2.destroyAllWindows()


if __name__ == "__main__": 
    # data_file = open(r'/home/pragyan/cv_project/darknet/object-detection-opencv/stuff_data/data_stuff.pkl', 'rb')
    # size, pose_stuff = pickle.load(data_file)
    
    # helmet_func(args,pose_stuff,size)
    yolo_file = open(r'/home/pragyan/cv_project/darknet/object-detection-opencv/stuff_data/delete_later/data_stuff.pkl', 'rb')
    pose_file = open(r'/home/pragyan/cv_project/darknet/object-detection-opencv/stuff_data/delete_later/pose_stuff.pkl', 'rb')
    size, yolo_stuff = pickle.load(yolo_file)
    size, pose_stuff, globalRiders = pickle.load(pose_file)
    # print(globalRiders)
    # print(yolo_stuff)
    helmet_func(args,pose_stuff,yolo_stuff,size, globalRiders)
    