#!/usr/bin/env python3
# -- coding: utf-8 --

import rospy
import numpy as np
import rospkg
import datetime
import tf
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, PointCloud2
from ultralytics import YOLO
import torch
from PIL import Image as img
from detector_2d.msg import DicBoxes, CoordBoxes
from detector_2d.srv import Log
import processing as pr
import time
import traceback
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from hera_objects.srv import FindObject
import sensor_msgs.point_cloud2 as pc2

class Object:
    def __init__(self):
        self.xywh = None
        self.obj_class = None

class Detector:
    def __init__(self):
        self.objects = rospy.ServiceProxy('/objects', FindObject)
        model_name = rospy.get_param('~model_name')
        rospy.loginfo(f"Usando modelo: {model_name}")
        image_topic = rospy.get_param('~camera_topic')
        point_cloud_topic = rospy.get_param('~point_cloud_topic', None)

        rospack = rospkg.RosPack()
        self.path_to_package = rospack.get_path('detector_2d')
        self._global_frame = rospy.get_param('~global_frame', None)
        self._tf_prefix = rospy.get_param('~tf_prefix', rospy.get_name())

        self.yolo = YOLO(f'{self.path_to_package}/models/{model_name}')
        self._tf_listener = tf.TransformListener()
        self._tfpub = tf.TransformBroadcaster()
        self._bridge = CvBridge()

        self._current_image = None
        self._current_pc = None
        self._det_image = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._image_sub = rospy.Subscriber(image_topic, Image, self.image_callback)
        self._detectsub = rospy.Subscriber("/detector_2d_node/objects_label", Image, self.detect_callback)
        self._boxespub = rospy.Publisher('~boxes_coordinates', DicBoxes, queue_size=10)
        self._imagepub = rospy.Publisher('~objects_label', Image, queue_size=10)

        if point_cloud_topic:
            rospy.Subscriber(point_cloud_topic, PointCloud2, self.pc_callback)
        else:
            rospy.loginfo('No point cloud information available. Objects will not be placed in the scene.')

        rospy.Service('detector_log', Log, self.log)
        rospy.loginfo('Ready to detect!')

    def image_callback(self, image):
        self._current_image = image

    def pc_callback(self, pc):
        self._current_pc = pc

    def detect_callback(self, det):
        self._det_image = det

    def log(self, req):
        try:
            if self._current_image:
                ct = datetime.datetime.now()
                rospy.loginfo('Writing log')
                small_frame = self._bridge.imgmsg_to_cv2(self._det_image, desired_encoding='bgr8')
                small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                log_image_path = f'{self.path_to_package}/src/log {ct}.jpg'
                cv2.imwrite(log_image_path, small_frame)
                rospy.loginfo(f'Log written on {self.path_to_package}')

                resp = self.objects("all", "", 0, 0, [])
                taken_object = resp.taken_object

                canv = canvas.Canvas(f'{self.path_to_package}/src/log {ct}.pdf', pagesize=letter)
                objects_image = img.fromarray(np.uint8(small_frame)).convert('RGB')
                canv.drawInlineImage(objects_image, 0, 0)
                sx = 700
                for obj in taken_object:
                    canv.drawString(100, sx, str(obj))
                    sx -= 10
                canv.save()
                return True
            else:
                rospy.loginfo('No image to write log')
                return False
        except Exception as e:
            rospy.loginfo(f'Could not write log: {e}')
            return False

    def publish_bookcase_tall(self):
        try:
            trans, rot = self._tf_listener.lookupTransform('map', 'bookcase', rospy.Time(0))
            theta = tf.transformations.euler_from_quaternion(rot)
            dist = 0.55
            trans[0] += dist * np.cos(theta[2]) + 0.05
            trans[1] += dist * np.sin(theta[2])
            trans[2] += 0.92
            self._tfpub.sendTransform(trans, tf.transformations.quaternion_from_euler(0, 0, 0), rospy.Time.now(), "bookcase_tall", "map")
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn(f'Failed to publish bookcase tall: {e}')

    def run(self):
        frame_rate = 12
        prev = 0
        while not rospy.is_shutdown():
            time_elapsed = time.time() - prev
            if time_elapsed > 1.0 / frame_rate:
                prev = time.time()
                if self._current_image:
                    self.process_frame()

    def process_frame(self):
        try:
            small_frame = self._bridge.imgmsg_to_cv2(self._current_image, desired_encoding='bgr8')
            results = self.yolo.predict(source=small_frame, conf=0.7, device=0, verbose=False)
            boxes = results[0].boxes
            objects = [Object() for obj in boxes]
            for obj, detection in zip(objects, boxes):
                obj.xywh = detection.xywh.tolist()[0]
                obj.obj_class = str(self.yolo.names[int(detection.cls)])

            objects.sort(key=lambda x: x.xywh[0])

            detected_object = DicBoxes()
            for obj in objects:
                aux = self.create_coord_box(obj, objects)
                detected_object.detected_objects.append(aux)
                if self._current_pc:
                    self.publish_object_tf(obj, aux)

            small_frame = pr.plot_bboxes(small_frame, results[0].boxes.data, self.yolo.names, conf=0.7)
            self._imagepub.publish(self._bridge.cv2_to_imgmsg(small_frame, 'rgb8'))
            self._boxespub.publish(detected_object)
            self.publish_bookcase_tall()
        except Exception:
            traceback.print_exc()

    def create_coord_box(self, obj, objects):
        aux = CoordBoxes()
        aux.type.data = obj.obj_class
        aux.image_x.data = int(obj.xywh[0])
        aux.image_y.data = int(obj.xywh[1])
        aux.image_width.data = int(obj.xywh[2])
        aux.image_height.data = int(obj.xywh[3])
        aux.tf_id.data = f'{self._tf_prefix}/{obj.obj_class}{sum(1 for prev in objects[:objects.index(obj)] if prev.obj_class == obj.obj_class)}'
        return aux

    def publish_object_tf(self, obj, aux):
        points = self.get_points_for_pc(obj)
        try:
            pc_list = list(pc2.read_points(self._current_pc, skip_nans=True, field_names=('x', 'y', 'z'), uvs=points))
            if pc_list:
                publish_point = self.get_publish_point(pc_list)
                if publish_point:
                    point_x, point_y, point_z = publish_point
                    if self._global_frame:
                        self._tfpub.sendTransform(
                            [point_y, -point_z, -point_x],
                            tf.transformations.quaternion_from_euler(0, 0, 0),
                            rospy.Time.now(),
                            aux.tf_id.data,
                            self._global_frame
                        )
        except Exception as e:
            rospy.logwarn(f'Failed to publish object tf: {e}')

    def get_points_for_pc(self, obj):
        x_center, y_center, width, height = map(int, obj.xywh)
        points = [(x_center, y_center + 40)]
        points.extend([(x, y) for x in range(x_center - width // 2, x_center + width // 2 + 1, 15)
                       for y in range(y_center - height // 2, y_center + height // 2 + 1, 15)])
        return points

    def get_publish_point(self, pc_list):
        first = min(pc_list, key=lambda item: item[2])
        suitable_points = [(x, y) for x, y, z in pc_list if z <= first[2] + 0.1]
        if suitable_points:
            pc_list_x = np.mean([x for x, _ in suitable_points])
            pc_list_y = np.mean([y for _, y in suitable_points])
            return pc_list_x, pc_list_y, first[2]
        return None

if __name__ == '__main__':
    rospy.init_node('detector_2d', log_level=rospy.INFO)
    try:
        Detector().run()
    except KeyboardInterrupt:
        rospy.loginfo('Shutting down')
