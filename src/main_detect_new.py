#!/usr/bin/env python3
# -- coding: utf-8 --

import rospy
import numpy as np
import tf
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from ultralytics import YOLO
import torch
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import Image, PointCloud2
from detector_2d.msg import DicBoxes, CoordBoxes
import processing as pr
import time
import traceback

class Detector:
    def __init__(self):
        
        # get topics from launch file
        image_topic = rospy.get_param('~image_topic')
        point_cloud_topic = rospy.get_param('~point_cloud_topic', None)
        self._global_frame = rospy.get_param('~global_frame', None)
        self._tf_prefix = rospy.get_param('~tf_prefix', rospy.get_name())

        # load Yolo model
        self.yolo = YOLO("yolov8n.pt")

        # start tf listener and set topic image and point cloud to none
        self._tf_listener = tf.TransformListener()
        self._current_image = None
        self._current_pc = None

        # set torch to use gpu
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # create detector
        self._bridge = CvBridge()

        # image and point cloud subscribers
        # and variables that will hold their values
        self._image_sub = rospy.Subscriber(image_topic, Image, self.image_callback)

        # publisher for frames with detected objects
        self._imagepub = rospy.Publisher('~objects_label', Image, queue_size=5)
        self._boxespub = rospy.Publisher('~boxes_coordinates', DicBoxes, queue_size=5)

        if point_cloud_topic is not None:
            rospy.Subscriber(point_cloud_topic, PointCloud2, self.pc_callback)
        else:
            rospy.loginfo(
            'No point cloud information available. Objects will not be placed in the scene.')

        self._tfpub = tf.TransformBroadcaster()
        rospy.loginfo('Ready to detect!')

    def image_callback(self, image):
        """Image callback"""
        # Store value on a private attribute
        self._current_image = image

    def pc_callback(self, pc):
        """Point cloud callback"""
        # Store value on a private attribute
        self._current_pc = pc

    def run(self):
        frame_rate = 5
        prev = 0
        while not rospy.is_shutdown():
            time_elapsed = time.time() - prev
            if time_elapsed > 1./frame_rate:
                prev = time.time()

                if self._current_image is not None:
                    try:
                        small_frame = self._bridge.imgmsg_to_cv2(self._current_image, desired_encoding='bgr8')
                        if self._global_frame is not None:
                            (trans, _) = self._tf_listener.lookupTransform('/' + self._global_frame, '/zed2_camera_center', rospy.Time(0))
                        results = self.yolo.predict(source=small_frame, conf=0.8, device=0)
