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
from detector_2d.msg import DicBoxes, CoordBoxes
from detector_2d.srv import Log
from PIL import Image as img
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from hera_objects.srv import FindObject
import processing as pr
import time
import traceback
from sensor_msgs import point_cloud2 as pc2

class Object:
    def __init__(self, xywh=None, obj_class=None):
        self.xywh = xywh
        self.obj_class = obj_class

class Detector:

    def __init__(self):
        self._initialize_params()
        self._initialize_subscribers_and_publishers()
        self._initialize_detector_model()
        self._initialize_tf_listener()

        rospy.Service('detector_log', Log, self.log)
        rospy.loginfo('Ready to detect!')

    def _initialize_params(self):
        """ Initialize ROS parameters and essential variables """
        self.objects = rospy.ServiceProxy('/objects', FindObject)
        rospack = rospkg.RosPack()
        self.path_to_package = rospack.get_path('detector_2d')
        self._global_frame = rospy.get_param('~global_frame', None)
        self._tf_prefix = rospy.get_param('~tf_prefix', rospy.get_name())
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._current_image = None
        self._current_pc = None
        self._det_image = None
        self._bridge = CvBridge()

    def _initialize_subscribers_and_publishers(self):
        """ Set up ROS subscribers and publishers """
        image_topic = rospy.get_param('~camera_topic')
        point_cloud_topic = rospy.get_param('~point_cloud_topic', None)
        self._image_sub = rospy.Subscriber(image_topic, Image, self.image_callback)
        self._imagepub = rospy.Publisher('~objects_label', Image, queue_size=10)
        self._boxespub = rospy.Publisher('~boxes_coordinates', DicBoxes, queue_size=10)
        self._detectsub = rospy.Subscriber("/detector_2d_node/objects_label", Image, self.detect_callback)

        if point_cloud_topic:
            rospy.Subscriber(point_cloud_topic, PointCloud2, self.pc_callback)
        else:
            rospy.loginfo('No point cloud information available.')

        self._tfpub = tf.TransformBroadcaster()

    def _initialize_detector_model(self):
        """ Load the YOLO model """
        model_name = rospy.get_param('~model_name')
        rospy.loginfo(f"Using model: {model_name}")
        self.yolo = YOLO(f'{self.path_to_package}/models/{model_name}')

    def _initialize_tf_listener(self):
        """ Initialize the TF listener """
        self._tf_listener = tf.TransformListener()

    def image_callback(self, image):
        self._current_image = image

    def pc_callback(self, pc):
        self._current_pc = pc

    def detect_callback(self, det):
        self._det_image = det

    def log(self, req):
        try:
            if self._current_image is None:
                rospy.loginfo('No image to write log')
                return False

            # Save image and log PDF
            self._save_log_image_and_pdf()
            return True

        except Exception as e:
            rospy.loginfo(f"Error logging: {e}")
            return False

    def _save_log_image_and_pdf(self):
        """ Helper method to save log image and create a PDF report """
        ct = datetime.datetime.now()
        rospy.loginfo('Writing log')
        small_frame = self._bridge.imgmsg_to_cv2(self._det_image, desired_encoding='bgr8')
        cv2.imwrite(f'{self.path_to_package}/log/log_{ct}.jpg', small_frame)
        
        # PDF generation
        self._create_log_pdf(small_frame, ct)

    def _create_log_pdf(self, small_frame, ct):
        """ Create a PDF report with detected objects """
        resp = self.objects("all", "", 0, 0)
        taken_object = resp.taken_object
        canv = canvas.Canvas(f'{self.path_to_package}/src/log_{ct}.pdf', pagesize=letter)
        objects_image = img.fromarray(np.uint8(small_frame)).convert('RGB')
        canv.drawInlineImage(image=objects_image, x=0, y=0)
        sx = 700

        for obj in taken_object:
            canv.drawString(100, sx, str(obj))
            sx -= 10

        canv.save()

    def publish_bookcase_tall(self):
        try:
            trans, a = self._tf_listener.lookupTransform('map', 'bookcase', rospy.Time(0))
            theta = tf.transformations.euler_from_quaternion(a)
            dist = 0.55
            trans[0] += dist * np.cos(theta[2]) + 0.05
            trans[1] += dist * np.sin(theta[2])
            trans[2] += 0.92
            self._tfpub.sendTransform(trans, tf.transformations.quaternion_from_euler(0, 0, 0), rospy.Time.now(), "bookcase_tall", "map")
        except Exception as e:
            rospy.loginfo(f"Error publishing bookcase tall: {e}")

    def run(self):
        """ Main loop """
        frame_rate = 12
        prev = 0
        while not rospy.is_shutdown():
            time_elapsed = time.time() - prev
            if time_elapsed > 1./frame_rate and self._current_image:
                prev = time.time()
                self._process_frame()

    def _process_frame(self):
        """ Process a single frame and publish results """
        try:
            small_frame = self._bridge.imgmsg_to_cv2(self._current_image, desired_encoding='bgr8')
            small_frame = cv2.resize(small_frame, (1280, 720))
            results = self.yolo.predict(source=small_frame, conf=0.6, device=self.device, verbose=False)

            detected_object = DicBoxes()
            objects = self._extract_detected_objects(results)

            self._publish_tf_and_boxes(objects, detected_object)
            self._publish_image_with_boxes(small_frame, results)

            self.publish_bookcase_tall()

        except Exception:
            traceback.print_exc()

    def _extract_detected_objects(self, results):
        """ Extract detected objects from YOLO results """
        objects = []
        for obj in results[0].boxes:
            objects.append(Object(xywh=obj.xywh.tolist()[0], obj_class=str(self.yolo.names[int(obj.cls)])))
        return sorted(objects, key=lambda x: x.xywh[0])

    def _publish_tf_and_boxes(self, objects, detected_object):
        """ Publish detected object coordinates and transform frames """
        for obj in objects:
            aux = CoordBoxes()
            aux.type.data = obj.obj_class
            aux.image_x.data = int(obj.xywh[0])
            aux.image_y.data = int(obj.xywh[1])
            aux.image_width.data = int(obj.xywh[2])
            aux.image_height.data = int(obj.xywh[3])

            tf_id, publish_tf, publish_point = self._process_point_cloud(obj)

            detected_object.detected_objects.append(aux)

            if publish_tf:
                self._publish_tf(tf_id, publish_point)

        self._boxespub.publish(detected_object)

    def _process_point_cloud(self, obj):
        """ Process point cloud to calculate object position """
        tf_id = f"{self._tf_prefix}/{obj.obj_class}"
        publish_tf = False
        publish_point = None

        if self._current_pc is not None:
            points = self._generate_points_around_object(obj)
            pc_list = self._read_point_cloud(points)

            if pc_list:
                publish_point = self._calculate_publish_point(pc_list)
                publish_tf = True

        return tf_id, publish_tf, publish_point

    def _generate_points_around_object(self, obj):
        """ Generate a list of points around the detected object """
        x_center = int(obj.xywh[0])
        y_center = int(obj.xywh[1])
        width = int(obj.xywh[2])
        height = int(obj.xywh[3])

        points = [(x_center, y_center + 40)]
        for x in range(x_center - width // 2, x_center + width // 2 + 1, 15):
            for y in range(y_center - height // 2, y_center + height // 2 + 1, 15):
                points.append((x, y))
        return points

    def _read_point_cloud(self, points):
        """ Read points from the point cloud """
        try:
            return list(pc2.read_points(self._current_pc, skip_nans=True, field_names=('x', 'y', 'z'), uvs=points))
        except:
            return []

    def _calculate_publish_point(self, pc_list):
        """ Calculate the point in the point cloud to publish """
        first = min(pc_list, key=lambda item: item[2])
        suitable_x = [item[0] for item in pc_list if item[2] <= first[2] + 0.1]
        suitable_y = [item[1] for item in pc_list if item[2] <= first[2] + 0.1]

        return [(min(suitable_x) + max(suitable_x)) / 2, (min(suitable_y) + max(suitable_y)) / 2, first[2]]

    def _publish_tf(self, tf_id, publish_point):
        """ Publish the transform frame for the detected object """
        object_tf = [publish_point[1], -publish_point[2], -publish_point[0]]
        frame = self._global_frame or "map"
        self._tfpub.sendTransform(object_tf, tf.transformations.quaternion_from_euler(0, 0, 0), rospy.Time.now(), tf_id, frame)

    def _publish_image_with_boxes(self, small_frame, results):
        """ Publish the image with drawn bounding boxes """
        small_frame = pr.plot_bboxes(small_frame, results[0].boxes.data, self.yolo.names, conf=0.6)
        self._imagepub.publish(self._bridge.cv2_to_imgmsg(small_frame, 'rgb8'))


if __name__ == '__main__':
    rospy.init_node('detector_2d', log_level=rospy.INFO)
    try:
        Detector().run()
    except KeyboardInterrupt:
        rospy.loginfo('Shutting down')
