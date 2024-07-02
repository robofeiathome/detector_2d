#!/usr/bin/env python3
# -- coding: utf-8 --

import rospy
import numpy as np
import rospkg
import datetime
import tf
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from ultralytics import YOLO
import torch
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import Image, PointCloud2
from PIL import Image as img
from detector_2d.msg import DicBoxes, CoordBoxes
from detector_2d.srv import Log
import processing as pr
import time
import traceback
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from hera_objects.srv import FindObject, FindSpecificObject

class Object:

    def __init__(self):
        self.xywh = None
        self.obj_class = None

class Detector:

    def __init__(self):

        self.objects = rospy.ServiceProxy('/objects', FindObject)
        model_name = rospy.get_param('~model_name')
        print("Usando modelo: ", model_name)
        image_topic = rospy.get_param('~camera_topic')
        point_cloud_topic = rospy.get_param('~point_cloud_topic', None)

        rospack = rospkg.RosPack()
        self.path_to_package = rospack.get_path('detector_2d')
        # self._global_frame = "camera_bottom_screw_frame"
        self._global_frame = rospy.get_param('~global_frame', None)
        self._tf_prefix = rospy.get_param('~tf_prefix', rospy.get_name())

        self.yolo = YOLO(f'{self.path_to_package}/models/{model_name}')

        self._tf_listener = tf.TransformListener()
        self._current_image = None
        self._current_pc = None
        self._det_image = None

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # create detector
        self._bridge = CvBridge()

        # image and point cloud subscribers
        # and variables that will hold their values
        self._image_sub = rospy.Subscriber(image_topic, Image, self.image_callback)


        # publisher for frames with detected objects
        self._imagepub = rospy.Publisher('~objects_label', Image, queue_size=10)
        self._detectsub = rospy.Subscriber("/detector_2d_node/objects_label", Image, self.detect_callback)
        self._boxespub = rospy.Publisher('~boxes_coordinates', DicBoxes, queue_size=10)
        
        if point_cloud_topic is not None:
            rospy.Subscriber(point_cloud_topic, PointCloud2, self.pc_callback)
        else:
            rospy.loginfo(
            'No point cloud information available. Objects will not be placed in the scene.')

        self._tfpub = tf.TransformBroadcaster()

        rospy.Service('detector_log', Log, self.log)

        rospy.loginfo('Ready to detect!')

    def image_callback(self, image):
        """Image callback"""
        # Store value on a private attribute
        self._current_image = image

    def pc_callback(self, pc):
        """Point cloud callback"""
        # Store value on a private attribute
        self._current_pc = pc

    def detect_callback(self, det):
        self._det_image = det

    def log(self, req):
        try:
            if self._current_image is not None:
                    ct = datetime.datetime.now()
                    rospy.loginfo('Writing log')
                    small_frame = self._bridge.imgmsg_to_cv2(self._det_image, desired_encoding='bgr8')
                    # small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(f'{self.path_to_package}/src/log {ct}.jpg', small_frame)
                    rospy.loginfo('Log written on '+self.path_to_package)

                    resp = self.objects("all", "", 0, 0)
                    taken_object = resp.taken_object

                    # pdf
                    canv = canvas.Canvas(f'{self.path_to_package}/src/log {ct}.pdf', pagesize=letter)
                    objects_image = img.fromarray(np.uint8(small_frame)).convert('RGB')
                    canv.drawInlineImage(image=objects_image, x=0, y=0)
                    sx = 700
                    for obj in taken_object:
                        canv.drawString(100,sx,str(obj))
                        sx -= 10

                    canv.save()

                    return True
                # except Exception as e:
                #     print(e)
                #     rospy.loginfo('Could not write Log')
                #     return False
            else:
                rospy.loginfo('No image to write log')
                return False
        except Exception as e:
            print(e)
            return False

    def publish_bookcase_tall(self):
        trans, a = self._tf_listener.lookupTransform('map', 'bookcase', rospy.Time(0))
        theta = tf.transformations.euler_from_quaternion(a)
        dist = 0.55
        trans[0] = trans[0] + dist*np.cos(theta[2]) + 0.05
        trans[1] = trans[1] + dist*np.sin(theta[2])
        trans[2] = trans[2] + 0.92
        self._tfpub.sendTransform((trans), tf.transformations.quaternion_from_euler(0, 0, 0), rospy.Time.now(), "bookcase_tall", "map")


    def run(self):
        # run while ROS runs

        frame_rate = 12
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
                        small_frame = cv2.resize(small_frame, (1280, 720))
                        detected_object = DicBoxes()
                        
                        #Load model
                        results = self.yolo.predict(source=small_frame, conf=0.6, device=0, verbose=False)
                        
                        #Boxes to msg
                        boxes = results[0].boxes
                        objects = []

                        for obj in boxes:
                            object = Object()
                            object.xywh = obj.xywh.tolist()[0]
                            object.obj_class = str(self.yolo.names[int(obj.cls)])
                            objects.append(object)

                        objects.sort(key=lambda x: x.xywh[0], reverse=False)

                        for obj in objects:

                            same_object_counter = 0

                            arr = obj.xywh
                            aux = CoordBoxes()
                            aux.type.data = obj.obj_class
                            aux.image_x.data = int(arr[0])
                            aux.image_y.data = int(arr[1])
                            aux.image_width.data = int(arr[2])
                            aux.image_height.data = int(arr[3])

                            for previous in objects[0:objects.index(obj)]:
                                if previous.obj_class == obj.obj_class:
                                    same_object_counter += 1

                            #Publish tf
                            publish_tf = False
                            if self._current_pc is None:
                                # rospy.loginfo('No point cloud')
                                pass
                            else:
                                # y_center = round(arr[1].item() - ((arr[1].item() - arr[3].item()) / 2))
                                # x_center = round(arr[0].item() - ((arr[0].item() - arr[2].item()) / 2))
                                # this function gives us a generator of points.
                                # we ask for a single point in the center of our object.
                                
                                x_center = int(obj.xywh[0])
                                y_center = int(obj.xywh[1])
                                width = int(obj.xywh[2])
                                height = int(obj.xywh[3])
                                
                                # print(x_center, y_center)

                                # print(self._current_pc)
                                points = [(x_center, y_center + 40)]

                                for x in range((x_center - int(width/2)), (x_center + int(width/2) + 1), 15):
                                    for y in range((y_center - int(height/2)), (y_center + int(height/2) + 1), 15):
                                        points.append(tuple([x, y]))

                                try:
                                    pc_list = list(
                                        pc2.read_points(self._current_pc,
                                                    skip_nans=True,
                                                    field_names=('x', 'y', 'z'),
                                                    uvs=points))
                                except:
                                    pc_list = []
                                # pc_list_sum = np.sum(pc_list,axis=0)

                                # if obj_class == 'English_Sauce':
                                #     for k in pc_list:
                                #         print(k[2])
                                
                                if len(pc_list) > 0:
                                    first = pc_list[0]
                                    for item in pc_list:
                                        if item[2] < first[2]:
                                            first = item

                                    suitable_x = []
                                    suitable_y = []

                                    for item in pc_list:
                                        if item[2] <= first[2] + 0.1:
                                            suitable_x.append(item[0])
                                            suitable_y.append(item[1])

                                    pc_list_x = (min(suitable_x) + max(suitable_x))/2
                                    pc_list_y = (min(suitable_y) + max(suitable_y))/2
                                    pc_list_z = first[2]
                                    publish_point = [pc_list_x, pc_list_y, pc_list_z]

                                    publish_tf = True
                                    # this is the location of our object in space
                                    # point_z, point_x, point_y = (pc_list_sum[0]/len(pc_list)), (pc_list_sum[1]/len(pc_list)), (pc_list_sum[2]/len(pc_list))
                                    point_z, point_x, point_y = publish_point
                                # if the user passes a tf prefix, we append it to the object tf name here
                                if self._tf_prefix is not None:
                                    
                                    tf_id = self._tf_prefix + '/' + str(obj.obj_class) + str(same_object_counter)
                                    aux.tf_id.data = tf_id

                            # print(aux)
                            detected_object.detected_objects.append(aux)
                            
                            if publish_tf:
                                # object tf (x, y, z) must be
                                # passed as (z,-x,-y)
                                object_tf = [point_y, -point_z + 0.0, -point_x]
                                # print(object_tf)

                                # translate the tf in regard to the fixed frame
                                if self._global_frame is not None:
                                    frame = self._global_frame

                                if object_tf is not None and point_x != float("-inf") and point_x != float("inf") and point_y != float("-inf") and point_y != float("inf") and point_z != float("-inf") and point_z != float("inf"):
                                    try:
                                        self._tfpub.sendTransform((object_tf),
                                                                    tf.transformations.quaternion_from_euler(0, 0, 0),
                                                                    rospy.Time.now(),
                                                                    tf_id,
                                                                    frame)
                                    except:
                                        pass

                        #Plot bbox 
                        small_frame = pr.plot_bboxes(small_frame, results[0].boxes.data, self.yolo.names, conf=0.6)
                        
                        #Publisher
                        self._imagepub.publish(self._bridge.cv2_to_imgmsg(small_frame, 'rgb8'))

                        self._boxespub.publish(detected_object)

                        self.publish_bookcase_tall()
                        

                    except Exception:
                        traceback.print_exc()        

if __name__ == '__main__':
    rospy.init_node('detector_2d', log_level=rospy.INFO)
    try:
        Detector().run()
    except KeyboardInterrupt:
        rospy.loginfo('Shutting down')
