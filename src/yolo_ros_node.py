#!/usr/bin/env python

from __future__ import print_function
import os
import time
import sys
sys.path.append('../')

import rospy
import rospkg
from vision_msgs.msg import BoundingBox2D
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import cv2
import pycuda.autoinit  # For initializing CUDA driver
import pycuda.driver as cuda

from utils.yolo_classes import get_cls_dict
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from yolo_trt import Yolo_TRT

from yolo_ros.msg import Detector2DArray
from yolo_ros.msg import Detector2D

from utils.helpers import *

class YoloNode(object):
    def __init__(self):
        """Constructor"""
        self.cv_bridge = CvBridge()
        self.init_params()
        self.init_yolo()
        self.cuda_ctx = cuda.Device(0).make_context()
        self.yolo = Yolo_TRT((self.model_path + self.model), (self.h, self.w), self.category_num)
        print("[YOLO-Node] Ros Node Initialization done")

    def __del__(self):
        """Destructor"""
        self.cuda_ctx.pop()
        del self.yolo
        del self.cuda_ctx

    def init_params(self):
        
        """ Initializes ros parameters """
        rospack = rospkg.RosPack()
        package_path = rospack.get_path("yolo_ros")
        self.video_topic = rospy.get_param("/video_topic", "/dummy_image_topic")
        self.model = rospy.get_param("/model", "yolov4-480")
        self.model_path = rospy.get_param(
            "/model_path", package_path + "/models/")
        self.category_num = rospy.get_param("/category_number", 2)
        self.input_shape = rospy.get_param("/input_shape", "480")
        self.conf_th = rospy.get_param("/confidence_threshold", 0.5)
        self.show_img = rospy.get_param("/show_image", False)
        self.namesfile = rospy.get_param("/namesfile_path", package_path+ "/cfg/obj.names")
        
        """ Setup Subscribers"""
        self.image_sub = rospy.Subscriber(
            self.video_topic, Image, self.img_callback, queue_size=10, buff_size=1920*1080*3)

        """ Setup Publishers"""
        self.detection_pub = rospy.Publisher(
            "/detections", Detector2DArray, queue_size=1)
        self.overlay_pub = rospy.Publisher(
            "/result/overlay", Image, queue_size=1)

    def init_yolo(self):
        """ Initialises yolo parameters required for the TensorRT engine """

        if self.model.find('-') == -1:
            self.model = self.model + "-" + self.input_shape
            
        yolo_dim = self.model.split('-')[-1] # yolo_dim = input size = 480

        self.h = self.w = int(yolo_dim)  # h = w = 480
        if self.h % 32 != 0 or self.w % 32 != 0:
            raise SystemExit('ERROR: bad yolo_dim (%s)!' % yolo_dim)

        cls_dict = get_cls_dict(self.category_num)   # cls_dict = {0: 'drone', 1: 'human'}

        self.vis = BBoxVisualization(cls_dict)


    def img_callback(self, ros_img):
        """Continuously capture images from camera and do object detection """
        tic = time.time()

        # Convert ROS Image msg into CV Image (BGR encoding)
        try:
            cv_img = self.cv_bridge.imgmsg_to_cv2(ros_img, desired_encoding="bgr8")
            rospy.logdebug("ROS Image converted for processing")
        except CvBridgeError as e:
            rospy.logerr("Failed to convert image %s", str(e))

        # Detection via img callback
        if cv_img is not None:
            #boxes, confs, clss = self.yolo.detect(cv_img, self.conf_th)
            detections = self.yolo.detect(cv_img, self.conf_th)
            #cv_img = self.vis.draw_bboxes(cv_img, boxes, confs, clss)
            class_names = load_class_names(self.namesfile)
            cv_img, boxes, clss, confs = plot_boxes_cv2(cv_img, detections, class_names=class_names)
            toc = time.time()
            fps = 1.0 / (toc - tic)
            
            # Publish the bounding box center coordinate, confidence score, and Class of object
            self.publisher(boxes, confs, clss)

            if self.show_img:
                cv_img = show_fps(cv_img, fps)
                cv2.imshow("YOLOv4 DETECTION RESULTS", cv_img)
                cv2.waitKey(1)

        # Convert CV Image back to ROS Image msg for publishing the overlay images
        try:
            overlay_img = self.cv_bridge.cv2_to_imgmsg(cv_img, encoding="bgr8")
            rospy.logdebug("CV Image converted for publishing")
            self.overlay_pub.publish(overlay_img)
        except CvBridgeError as e:
            rospy.loginfo("Failed to convert image %s", str(e))

    def publisher(self, boxes, confs, clss):
        """ Publishes to detector_msgs

        Parameters:
        boxes (List(List(int))) : Bounding boxes of all objects (centers<x,y>, sizes<x,y>)
        confs (List(double))	: Probability/Confident scores of all objects
        clss  (List(int))	: Class ID of all classes {0/1}
        """
        detections_msg = Detector2DArray()
        detections_msg.header.stamp = rospy.Time.now()
        detections_msg.header.frame_id = "camera" # change accordingly
        time_stamp = rospy.Time.now()
        for i in range(len(boxes)):
            # boxes : xmin, ymin, xmax, ymax
            #for _ in boxes:
            detection = Detector2D()
            detection.header.stamp = time_stamp
            detection.header.frame_id = "camera" # change accordingly
            detection.results.id = clss[i]
            detection.results.score = confs[i]
            detection.bbox.center.x = boxes[i][0] + (boxes[i][2] - boxes[i][0])/2
            detection.bbox.center.y = boxes[i][1] + (boxes[i][3] - boxes[i][1])/2
            detection.bbox.center.theta = 0.0  # change if required

            detection.bbox.size_x = abs(boxes[i][0] - boxes[i][2])
            detection.bbox.size_y = abs(boxes[i][1] - boxes[i][3])

            detections_msg.detections.append(detection)
        
        rospy.logdebug("Number of detections in the list: {}".format(len(detections_msg.detections)))
        
        self.detection_pub.publish(detections_msg)

def main():
    node = YoloNode()
    rospy.init_node('yolo_ros_node', anonymous=True, log_level=rospy.INFO)
    try:

        # spin() simply keeps python from exiting until this node is stopped
        rospy.spin()
    
    except KeyboardInterrupt:
        rospy.on_shutdown(yolo.clean_up())
        print("Shutting down")


if __name__ == '__main__':
    main()



