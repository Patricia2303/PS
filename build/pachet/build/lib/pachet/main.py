import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
from cv_bridge import CvBridge
import numpy as np


class LaneFollowing(Node):
    def __init__(self):
        super().__init__('lanefollowing')

        # Abonare la imaginea de la camera
        self.img_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.subs_callback,
            10
        )

        # Publicarea comenzilor de viteza
        self.cmd_vel_pub_ = self.create_publisher(Twist, 'cmd_vel', 10)

        # Timer pentru actualizarea comenzilor
        self.update_timer_ = self.create_timer(0.01, self.update_callback)


        # Inițializare variabile
        self.bridge = CvBridge()
        self.prevpt1 = (0, 0)
        self.prevpt2 = (0, 0)
        self.fpt = (0, 0)
        self.error = 0

    def subs_callback(self, msg):
        # Conversia imaginii din ROS la OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        gray = self.bridge.imgmsg_to_cv2(msg, "mono8")

        # Preprocesare
        gray = gray + 100 - np.mean(gray)
        _, gray = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)

        dst = gray[gray.shape[0] // 3 * 2:]

        # Detectarea componentelor conectate
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
        
        mindistance1 = []
        mindistance2 = []
        if num_labels > 1:
            for i in range(1, num_labels):
                p = centroids[i]
                ptdistance1 = abs(p[0] - self.prevpt1[0])
                ptdistance2 = abs(p[0] - self.prevpt2[0])
                mindistance1.append(ptdistance1)
                mindistance2.append(ptdistance2)

            threshdistance1 = min(mindistance1)
            threshdistance2 = min(mindistance2)

            minlb1 = mindistance1.index(threshdistance1)
            minlb2 = mindistance2.index(threshdistance2)

            cpt1 = (centroids[minlb1, 0], centroids[minlb1, 1])
            cpt2 = (centroids[minlb2, 0], centroids[minlb2, 1])

            if threshdistance1 > 100:
                cpt1 = self.prevpt1
            if threshdistance2 > 100:
                cpt2 = self.prevpt2

            self.prevpt1 = cpt1
            self.prevpt2 = cpt2
        else:
            cpt1 = self.prevpt1
            cpt2 = self.prevpt2

        # Calcularea erorii de direcție
        self.fpt = ((cpt1[0] + cpt2[0]) / 2, (cpt1[1] + cpt2[1]) / 2 + gray.shape[0] // 3 * 2)
        self.error = gray.shape[1] / 2 - self.fpt[0]

        # Desenarea liniilor pe imagine
        cv2.circle(frame, (int(self.fpt[0]), int(self.fpt[1])), 2, (0, 0, 255), 2)
        cv2.circle(dst, (int(cpt1[0]), int(cpt1[1])), 2, (0, 0, 255), 2)
        cv2.circle(dst, (int(cpt2[0]), int(cpt2[1])), 2, (255, 0, 0), 2)

        # Afișarea imaginilor
        cv2.imshow("camera", frame)
        cv2.imshow("gray", dst)
        cv2.waitKey(1)

    def update_callback(self):
        # Publicarea comenzilor de mișcare pe baza erorii de direcție
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.5
        cmd_vel.angular.z = (self.error * 90.0 / 400) / 15
        self.cmd_vel_pub_.publish(cmd_vel)


def main(args=None):
    rclpy.init(args=args)
    lane_following = LaneFollowing()
    rclpy.spin(lane_following)
    lane_following.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
