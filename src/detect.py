#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import tf

from calendar import c
from pyexpat import model

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import Image, PointCloud2
import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
from hera_face.srv import face_list
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor

import processing as pr
#from PIL import Image

class Detector:

    def __init__(self):
        image_topic = '~/zed2/zed_node/left_raw/image_raw_color'
        point_cloud_topic = '/zed2/zed_node/point_cloud/cloud_registered'

        self._global_frame = 'camera'
        self._frame = 'camera_depth_frame'
        self._tf_listener = tf.TransformListener()
        # create detector
        self._bridge = CvBridge()

        # image and point cloud subscribers
        # and variables that will hold their values
        rospy.Subscriber(image_topic, Image, self.image_callback)

        if point_cloud_topic is not None:
            rospy.Subscriber(point_cloud_topic, PointCloud2, self.pc_callback)
        else:
            rospy.loginfo(
                'No point cloud information available. Objects will not be placed in the scene.')

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

    def pc_callback(self, pc):
        """Point cloud callback"""
        # Store value on a private attribute
        self._current_pc = pc

    def run(self):
        # run while ROS runs
        while not rospy.is_shutdown():
            # only run if there's an image present
            if self._current_image is not None:
                try:
                    #Busca imagem
                    small_frame = self._bridge.imgmsg_to_cv2(self._current_image, desired_encoding='bgr8')
                    
                    #Carrega o modelo
                    yolo = YOLO("yolov8n.pt")
                    results = yolo.predict(source=small_frame, conf=0.8)
                    boxes = results[0].boxes
                    
                    #Coloca as bboxes
                    small_frame = pr.plot_bboxes(small_frame, boxes.boxes, conf=0.5)
                    #small_frame = results[0].plot() <-- outra opção de colocar bbox

                    #publica o topico
                    self._imagepub.publish(self._bridge.cv2_to_imgmsg(small_frame, 'rgb8'))

                except CvBridgeError as e:
                    print(e)

if __name__ == '__main__':
    rospy.init_node('detector_2d', log_level=rospy.INFO)

    try:
        Detector().run()
    except KeyboardInterrupt:
        rospy.loginfo('Shutting down')