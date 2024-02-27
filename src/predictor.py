#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from detector_2d.srv import Predictor
from detector_2d.msg import Detection, DetectionArray, BoundingBox
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO
import cv2

class Predict:
    def __init__(self):
        self.bridge_object = CvBridge()
        self.topic = rospy.get_param('~camera_topic')
        self.yolo = YOLO('yolov8m.pt')
        self.image_sub = rospy.Subscriber(self.topic, Image, self.camera_callback)
        self.service = rospy.Service('predictor', Predictor, self.handler)
        rospy.loginfo("Finished Predictor Init process, ready to predict")

    def camera_callback(self, data):
        """Callback function that receives camera images."""
        self.cam_image = data

    def save_image(self, img, path):
        cv2.imwrite(path, img)

    def predict(self, img, threshold=0.7, classes=None):
        try:
            cv_image = self.bridge_object.imgmsg_to_cv2(img, desired_encoding="bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return False

        result = self.yolo.predict(source=cv_image, conf=threshold, classes=classes)
        result = result[0].boxes
        
        if result:
            response = DetectionArray()
            response.detections = []

            for i, box in enumerate(result):
                det = box.xyxy[0]
                bbox_msg = BoundingBox()
                bbox_msg.x_min = det[0]
                bbox_msg.y_min = det[1]
                bbox_msg.x_max = det[2]
                bbox_msg.y_max = det[3]

                detection_msg = Detection()
                detection_msg.bbox = bbox_msg
                detection_msg.class_id = int(result.cls[i])

                response.detections.append(detection_msg)
            self.save_image(cv_image, 'test.jpg')
            return response
        return False

    def handler(self, request):
        if self.cam_image is None:
            rospy.logwarn("No image available from camera.")
            return False

        result = self.predict(self.cam_image, request.threshold, request.classes)
        return result

if __name__ == '__main__':
    rospy.init_node('predictor_node', log_level=rospy.INFO)
    predict = Predict()

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
