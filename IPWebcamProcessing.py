
from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
#import numpy as np
#import cv2 
from util import *
from darknet import Darknet
from preprocess import prep_image, inp_to_image
import pandas as pd
import random 
import argparse
import pickle as pkl
#ABOVE YOLO imports

import urllib.request as urllib
import cv2
import numpy as np



class IPCamera():

    def __init__(self, name, url, cancelKey):
        self.name = name
        self.cancelKey = cancelKey
        self.stream_cancelled = False
        self.url = url
        print("CAMERA " + self.name + " STREAM INITIALIZED")

    def start(self):
        return urllib.urlopen(self.url)

    def isRunning(self):
        return not self.stream_cancelled
    
    def cancelStream(self):
              
              self.stream_cancelled = True
              print("CAMERA " + self.name + " STREAM TERMINATED")
              return False

    def run(self):
        check = True 
        while (check):
            try:
                imgNP = np.array(bytearray(self.start().read()),dtype=np.uint8)
                img = cv2.imdecode(imgNP, -1)
                cv2.imshow('test', img)
                if ord(self.cancelKey) == cv2.waitKey(10):
                    check = self.cancelStream()
                    exit(0)
            except Exception as e:
                print(e)
                print(self.cancelKey)
                print(self.url)
                check = self.cancelStream()
        return
'''
if __name__ == "__main__":
    url = 'http://10.65.67.187:8080/shot.jpg'
    cam1 = IPCamera('GalaxyS9', url,'q')
    #cam1.run()
    '''

    
def get_test_input(input_dim, CUDA):
    img = cv2.imread("imgs/messi.jpg")
    img = cv2.resize(img, (input_dim, input_dim)) 
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    
    if CUDA:
        img_ = img_.cuda()
    
    return img_

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = cv2.resize(orig_im, (inp_dim, inp_dim))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

def write(x, img):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img

def arg_parse():
    """
    Parse arguements to the detect module
    """
    
    
    parser = argparse.ArgumentParser(description='YOLO v3 Cam Demo')
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.25)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "160", type = str)
    return parser.parse_args()



if __name__ == '__main__':
    url = 'http://10.65.67.137:8080/shot.jpg'
    cam1 = IPCamera('GalaxyS9', url,'q')
    
    cfgfile = "cfg/yolov3.cfg"
    weightsfile = "yolov3.weights"
    num_classes = 80

    args = arg_parse()
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0
    CUDA = not torch.cuda.is_available() #cuda causing crashes on my personal. 
   

    num_classes = 80
    bbox_attrs = 5 + num_classes
    
    model = Darknet(cfgfile)
    model.load_weights(weightsfile)
    
    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    
    assert inp_dim % 32 == 0 
    assert inp_dim > 32

    if CUDA:
        model.cuda()
            
    model.eval()

    capCam1 = cv2.VideoCapture(cam1.url)

    assert capCam1.isOpened(),'not capture source'
    
    frames = 0
    start = time.time()    
    while capCam1.isOpened():
        capCam1 = cv2.VideoCapture(cam1.url)

        ret, frame = capCam1.read()
        if ret:
        
            img, orig_im, dim = prep_image(frame, inp_dim)
            
#          im_dim = torch.FloatTensor(dim).repeat(1,2)                        
            
            
            if CUDA:
                im_dim = im_dim.cuda()
                img = img.cuda()
            
            
            output = model(Variable(img), CUDA)
            output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)

            if type(output) == int:
                #frames += 1
                #print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
                cv2.imshow("frame", cv2.resize(orig_im,1000,1000))
                key = cv2.waitKey(1)
                if key == ord(cam1.cancelKey):
                    cam1.cancelStream()
                    break
                continue
            

        
            output[:,1:5] = torch.clamp(output[:,1:5], 0.0, float(inp_dim))/inp_dim
            
#            im_dim = im_dim.repeat(output.size(0), 1)
            output[:,[1,3]] *= frame.shape[1]
            output[:,[2,4]] *= frame.shape[0]

            
            classes = load_classes('data/coco.names')
            colors = pkl.load(open("pallete", "rb"))
            
            list(map(lambda x: write(x, orig_im), output))
            
            
            cv2.imshow("frame", cv2.resize(orig_im,(1000,1000)))
            key = cv2.waitKey(1)
            if key & 0xFF == ord(cam1.cancelKey):
                cam1.cancelStream()
                break
            frames += 1
            print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))

            
        else:
            break
