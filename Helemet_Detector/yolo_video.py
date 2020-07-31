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
args = ap.parse_args()

frame_shift = 12

def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h, classes, COLORS):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def yolo_func(args):
    cam = cv2.VideoCapture(args.video)
    length = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))

    fps_time = 0
    i = 0
    size = 0

    # w, h = model_wh(args.resolution)
    # e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))


    # cap = cv2.VideoCapture(args.video)
    #hel = classifier(0)
    # path = Path('/home/pragyan/cv_project/pose_final/tf-pose-estimation')
    # hel = load_learner(path)

    yolo_stuff = []
    #pose_stuff = []

    while(True):

        ret_val, image = cam.read()
        image = cv2.flip(image,0)
        image = cv2.flip(image ,1)
        i+=frame_shift
        # if i%frame_shift!=0:
        #     continue

        if ret_val is False:
            print("Video File not opened!")
            break

        cam.set(1, i)
        
        size+=1
        Width = image.shape[1]
        Height = image.shape[0]
        scale = 0.00392

        classes = None

        with open(args.classes, 'r') as f:
            classes = [line.strip() for line in f.readlines()]

        COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

        net = cv2.dnn.readNet(args.weights, args.config)

        blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

        net.setInput(blob)

        outs = net.forward(get_output_layers(net))
        print(outs)
        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.7
        nms_threshold = 0.4
        # print ('progress: ', (i*100)/length, "%")
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        cv2.putText(image,
                        "FPS: %f" % (1.0 / (time.time() - fps_time)),
                        (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)
        cv2.putText(image,
                        "Progress: %f" % (100*i/length),
                        (200, 200),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)
        fps_time = time.time()
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        for j in indices:
            j = j[0]
            box = boxes[j]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            draw_prediction(image, class_ids[j], confidences[j], round(x), round(y), round(x+w), round(y+h), classes, COLORS)

        # print(size-1,box)
        yolo_stuff.append((indices,class_ids,confidences,boxes))

    
        # humans = e.inference(image,resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        # if not args.showBG:
        #     image = np.zeros(image.shape)
        # image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        # rectangles = TfPoseEstimator.detect_face(image, humans, imgcopy=False)
        # pose_stuff.append(rectangles)

        cv2.imshow("object detection", image)
        if cv2.waitKey(1) == 27:
            break

    cv2.imwrite("object-detection.jpg", image)
    cv2.destroyAllWindows()
    return yolo_stuff, size
'''
def pose_func(args,size,yolo_stuff):
    with open(args.classes, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
    pose_stuff = []
    cam = cv2.VideoCapture(args.video)
    fps_time = 0

    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    print(size)

    indexi = -1
    ii=0

    while indexi < size-1:
        ret_val, image = cam.read()
        ii+=1

        if ii%frame_shift!=0:
            continue

        if ret_val is False:
            print("BLAH")
            break

        indexi+=1
        humans = e.inference(image,resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        if not args.showBG:
            image = np.zeros(image.shape)
        image1 = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        rectangles = TfPoseEstimator.detect_face(image, humans, imgcopy=False)

        # yolo_stuff.append((indices,class_ids,confidences,boxes))
        indices, class_ids, confidences, boxes = yolo_stuff[indexi]
        for j in indices:
            j = j[0]
            box = boxes[j]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            draw_prediction(image, class_ids[j], confidences[j], round(x), round(y), round(x+w), round(y+h), classes, COLORS)

        cv2.imshow("object detection", image1)
        if cv2.waitKey(1) == 27:
            break
        pose_stuff.append((image1 ,image, rectangles, yolo_stuff[indexi]))

    #cv2.imwrite("object-detection.`jpg", image)
    cv2.destroyAllWindows()
    print(len(pose_stuff))
    return pose_stuff

def helmet_func(args,pose_stuff,yolo_stuff,size):
    path = Path('/home/pragyan/cv_project/pose_final/tf-pose-estimation')
    hel = load_learner(path)

    cam = cv2.VideoCapture(args.video)
    fps_time = 0
    i = 0
    indexesi = -1


    print(len(pose_stuff))
    while(indexesi<size-1):
        ret_val, image = cam.read()
        i+=1
        if i%frame_shift!=0:
            continue

        if ret_val is False:
            print("BLAH")
            break

        indexesi+=1
        #print(indexesi,size)
        indices = yolo_stuff[indexesi][0]
        class_ids = yolo_stuff[indexesi][1]
        confidences = yolo_stuff[indexesi][2]
        boxes = yolo_stuff[indexesi][3]

        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            # draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h), )
        
        rectangles = pose_stuff[indexesi]

        for index, rectangle in enumerate(rectangles):
            cv2.rectangle(image,rectangle[0],rectangle[1],(0,255,0),3)
            image2 = image[min(rectangle[0][1],rectangle[1][1]):max(rectangle[0][1],rectangle[1][1])+1,min(rectangle[0][0],rectangle[1][0]):max(rectangle[0][0],rectangle[1][0])+1]
            img = Image(pil2tensor(image2, dtype = np.float32).div_(255))
            rvar = hel.predict(img)
            print(rvar)
            # cv2.putText(image,
            #         "Prediction: %s" %rvar[0],
            #         rectangle[0],  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            #         (0, 255, 0), 2)

        cv2.imshow("object detection", image)
        if cv2.waitKey(1) == 27:
            break

    cv2.imwrite("object-detection.jpg", image)
    cv2.destroyAllWindows()
'''

if __name__ == "__main__": 
    yolo_stuff, size = yolo_func(args)
    #pose_stuff = pose_func(args,size,yolo_stuff)
    # pose_file = open(r'/home/pragyan/cv_project/darknet/object-detection-opencv/stuff_data/pose_stuff.pkl', 'wb')
    # pickle.dump(pose_stuff, pose_file)
    # yolo_file = open(r'/home/pragyan/cv_project/darknet/object-detection-opencv/stuff_data/yolo_stuff.pkl', 'wb')
    # pickle.dump(yolo_stuff, yolo_file) 
    data_str = (size, yolo_stuff)
    data_file = open(r'/home/pragyan/cv_project/darknet/object-detection-opencv/stuff_data/data_stuff.pkl', 'wb')
    pickle.dump(data_str, data_file)
    
    # helmet_func(args,pose_stuff,yolo_stuff,size)