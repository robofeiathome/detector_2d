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
        self._global_frame = 'zed2i_left_camera_frame'

        image_topic = rospy.get_param('~image_topic')
        point_cloud_topic = rospy.get_param('~point_cloud_topic', None)

        self._global_frame = rospy.get_param('~global_frame', None)
        self._tf_prefix = rospy.get_param('~tf_prefix', rospy.get_name())
        
        self.yolo = YOLO("/home/robofei/Workspace/catkin_ws/src/3rd_party/vision_System/detector_2d/src/best.pt")

        self._tf_listener = tf.TransformListener()
        self._current_image = None
        self._current_pc = None

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # create detector
        self._bridge = CvBridge()

        # image and point cloud subscribers
        # and variables that will hold their values
        self._image_sub = rospy.Subscriber(image_topic, Image, self.image_callback)


        # publisher for frames with detected objects
        self._imagepub = rospy.Publisher('~objects_label', Image, queue_size=10)
        self._boxespub = rospy.Publisher('~boxes_coordinates', DicBoxes, queue_size=10)
        
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
        # run while ROS runs

        frame_rate = 5
        prev = 0
        while not rospy.is_shutdown():
            time_elapsed = time.time() - prev
            if time_elapsed > 1./frame_rate:
                prev = time.time()
                # only run if there's an image present
                if self._current_image is not None:
                    try:
                        #Search image
                        small_frame = self._bridge.imgmsg_to_cv2(self._current_image, desired_encoding='bgr8')

                        if self._global_frame is not None:
                            (trans, _) = self._tf_listener.lookupTransform('/' + self._global_frame, 'zed2i_left_camera_frame', rospy.Time(0))
                        
                        detected_object = DicBoxes()
                        
                        #Load model
                        results = self.yolo.predict(source=small_frame, conf=0.7, device=0)
                        
                        #Boxes to msg
                        boxes = results[0].boxes
                        for i, c in enumerate(boxes.cls):
                            arr = boxes[i].xywh[0]
                            aux = CoordBoxes()
                            obj_class = self.yolo.names[int(c)]
                            aux.type.data = self.yolo.names[int(c)]
                            aux.image_x.data = int(arr[0].item())
                            aux.image_y.data = int(arr[1].item())
                            aux.image_width.data = int(arr[2].item()) 
                            aux.image_height.data = int(arr[3].item())

                            #Publish tf
                            publish_tf = False
                            if self._current_pc is None:
                                rospy.loginfo('No point cloud')
                            else:
                                # y_center = round(arr[1].item() - ((arr[1].item() - arr[3].item()) / 2))
                                # x_center = round(arr[0].item() - ((arr[0].item() - arr[2].item()) / 2))
                                # this function gives us a generator of points.
                                # we ask for a single point in the center of our object.
                                x_center = int(boxes[i].xywh[0][0])
                                y_center = int(boxes[i].xywh[0][1])

                                pc_list = list(
                                    pc2.read_points(self._current_pc,
                                                skip_nans=True,
                                                field_names=('x', 'y', 'z'),
                                                uvs=[(x_center, y_center)]))

                                if len(pc_list) > 0:
                                    publish_tf = True
                                    # this is the location of our object in space
                                    tf_id = obj_class + '_' + str(c)
                                    point_z, point_x, point_y = pc_list[0]

                                # if the user passes a tf prefix, we append it to the object tf name here
                                if self._tf_prefix is not None:
                                    tf_id = self._tf_prefix + '/' + str(self.yolo.names[int(c)]) + str(i)
                                    aux.tf_id.data = tf_id

                            # print(aux)
                            detected_object.detected_objects.append(aux)
                            
                            if publish_tf:
                                # object tf (x, y, z) must be
                                # passed as (z,-x,-y)
                                object_tf = [point_z, point_x + 0.0, point_y]
                                # print(object_tf)
                                frame = '/zed2i_left_camera_frame'

                                # translate the tf in regard to the fixed frame
                                if self._global_frame is not None:
                                    object_tf = np.array(trans) + object_tf
                                    frame = self._global_frame

                                if object_tf is not None and point_x != float("-inf") and point_y != float("-inf") and point_z != float("-inf"):
                                    try:
                                        self._tfpub.sendTransform((object_tf),
                                                                    tf.transformations.quaternion_from_euler(0, 0, 0),
                                                                    rospy.Time.now(),
                                                                    tf_id,
                                                                    frame)
                                    except:
                                        pass

                        #Plot bbox 
                        small_frame = pr.plot_bboxes(small_frame, results[0].boxes.boxes, self.yolo.names, conf=0.5)
                        
                        #Publisher
                        self._imagepub.publish(self._bridge.cv2_to_imgmsg(small_frame, 'rgb8'))

                        self._boxespub.publish(detected_object)
                        

                    except Exception:
                        traceback.print_exc()        

if __name__ == '__main__':
    rospy.init_node('detector_2d', log_level=rospy.INFO)
    try:
        Detector().run()
    except KeyboardInterrupt:
        rospy.loginfo('Shutting down')