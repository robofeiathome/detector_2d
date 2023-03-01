#!/usr/bin/env python3
# -- coding: utf-8 --

import rospy
import numpy as np
import tf

import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, PointCloud2
from hera_face.srv import face_list
import torch
from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor

import processing as pr

class Detector:

    def __init__(self):
        image_topic = '~/zed2/zed_node/left_raw/image_raw_color'

        self._global_frame = 'camera'
        self._frame = 'camera_depth_frame'
        self._tf_listener = tf.TransformListener()
        # create detector
        self._bridge = CvBridge()

        # image and point cloud subscribers
        # and variables that will hold their values
        self._image_sub = rospy.Subscriber(image_topic, Image, self.image_callback)

        self._current_image = None
        self._current_pc = None

        # publisher for frames with detected objects
        self._imagepub = rospy.Publisher('~objects_label', Image, queue_size=10)

        self._tfpub = tf.TransformBroadcaster()
        rospy.loginfo('Ready to detect!')

    def image_callback(self, image):
        """Image callback"""
        # Store value on a private attribute
        self._current_image = image

    def run(self):
        # run while ROS runs
        while not rospy.is_shutdown():
            # only run if there's an image present
            if self._current_image is not None:
                try:
                    #Search image
                    small_frame = self._bridge.imgmsg_to_cv2(self._current_image, desired_encoding='bgr8')
                    
                    #Load model
                    yolo = YOLO("yolov8n.pt")
                    results = yolo.predict(source=small_frame, conf=0.8)
                    boxes = results[0].boxes
                    
                    #Plot bbox 
                    small_frame = pr.plot_bboxes(small_frame, boxes.boxes, conf=0.5)
                    
                    #Publisher
                    self._imagepub.publish(self._bridge.cv2_to_imgmsg(small_frame, 'rgb8'))

                except CvBridgeError as e:
                    print(e)

if __name__ == '__main__':
    rospy.init_node('detector_2d', log_level=rospy.INFO)

    try:
        Detector().run()
    except KeyboardInterrupt:
        rospy.loginfo('Shutting down')