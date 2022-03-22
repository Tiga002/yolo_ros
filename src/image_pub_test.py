#!/usr/bin/env python3

from __future__ import print_function

import sys
import rospkg
import rospy
import cv2
from std_msgs.msg import String

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class image_converter:

    def __init__(self):
        
        self.image_pub = rospy.Publisher("/dummy_image_topic", Image, queue_size=10)
        self.cv_bridge = CvBridge()
        self.publish_rate = rospy.get_param("~publish_rate", 100)
        rospack = rospkg.RosPack()
        self.package_path = rospack.get_path("yolo_ros")
        self.test_image_path = rospy.get_param("~test_img_path", self.package_path+"/data/20.jpg")

    def run(self):
        rate = rospy.Rate(self.publish_rate)
        
        while not rospy.is_shutdown():
            img = cv2.imread(self.test_image_path)

            img_msg = self.cv_bridge.cv2_to_imgmsg(img, "bgr8")

            try:
                self.image_pub.publish(img_msg)
            except CvBridgeError as e:
                print(e)
            
            rate.sleep()

def main():
    rospy.init_node("image_pub_node")
    ic = image_converter()
    ic.run()

if __name__ == "__main__":
    main()






