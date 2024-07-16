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
        self._imagepub = rospy.Publisher('~objects_label', Image, queue_size=10)
        self.image_sub = rospy.Subscriber(self.topic, Image, self.camera_callback)
        self.service = rospy.Service('~predictor', Predictor, self.handler)
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
        result = result[0].boxes.data
        
        response = DetectionArray()
        response.detections = []

        for i, box in enumerate(result):
            print(box)
            #det = box.xyxy[0]
            bbox_msg = BoundingBox()
            bbox_msg.x_min = box[0].item()
            bbox_msg.y_min = box[1].item()
            bbox_msg.x_max = box[2].item()
            bbox_msg.y_max = box[3].item()

            detection_msg = Detection()
            detection_msg.bbox = bbox_msg
            detection_msg.class_id = self.yolo.names[result[i][5].item()]

            response.detections.append(detection_msg)
        #self.save_image(cv_image, 'test.jpg')
        self._imagepub.publish(self.bridge_object.cv2_to_imgmsg(cv_image, 'rgb8'))
        return response

    def handler(self, request):
        if self.cam_image is None:
            rospy.logwarn("No image available from camera.")
            return False

        result = self.predict(self.cam_image,
                              request.threshold, 
                              request.classes if len(request.classes) > 0 else None
                             )
        return result

if __name__ == '__main__':
    rospy.init_node('predictor_node', log_level=rospy.INFO)
    predict = Predict()

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
