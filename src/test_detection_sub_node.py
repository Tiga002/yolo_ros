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

from yolo_ros.msg import Detector2DArray
from yolo_ros.msg import Detector2D


def callback(msg):
    detections_list = msg.detections
    count = 0
    for i in range(len(detections_list)):
        count = count + 1
        center_x = detections_list[i].bbox.center.x
        center_y = detections_list[i].bbox.center.y
        class_id = detections_list[i].results.id
        conf_score = detections_list[i].results.score
        print("===== Detections #{} =====".format(count))
        print("BBox Center = <{}, {}>".format(center_x, center_y))
        print("Class ID = {}".format(class_id))
        print("Conf Score = {}".format(conf_score))
    print('---------------------------------------------')

def listener():
    rospy.init_node("dectections_sub_node")
    rospy.Subscriber("/detections", Detector2DArray, callback)
    rospy.spin()


if __name__ == "__main__":
    listener()


        