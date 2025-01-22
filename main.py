import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
from cv_bridge import CvBridge
import numpy as np
import time

class LaneFollowing(Node):
    def __init__(self):
        super().__init__('lanefollowing')
        
        self.img_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.subs_callback,
            10)
        
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
       
        self.bridge = CvBridge()
        self.error = 0
        self.prevpt1 = (0, 0)
        self.prevpt2 = (0, 0)
        self.stop_detected = False
        self.stop_time = None
        self.stop_once = False  # Flag pentru a permite o singură oprire la STOP

        # Inițializare SIFT și BFMatcher
        self.sift = cv2.SIFT_create()
        self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        # Încărcăm imaginea de referință cu semnul STOP
        self.stop_template = cv2.imread("/home/patricia23/ros2_ws/src/pachet/pachet/stop_sign.png", cv2.IMREAD_GRAYSCALE)
        if self.stop_template is None:
            self.get_logger().error("Imaginea stop_sign.png nu a fost găsită!")
            exit(1)
        self.kp_template, self.des_template = self.sift.detectAndCompute(self.stop_template, None)

    def subs_callback(self, msg):
        try:
            # Convertim imaginea de la ROS la format OpenCV
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
           
            # Detectăm semnul STOP doar dacă nu s-a oprit deja
            if not self.stop_once:
                self.detect_stop_sign_sift(gray, frame)

            # Afișăm imaginea procesată
            canny = cv2.Canny(frame, 100, 200)
            self.freespace(canny, frame)
            cv2.imshow("Camera", frame)

            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def detect_stop_sign_sift(self, gray_frame, frame):
        """Detectează semnul STOP folosind SIFT."""
        kp_frame, des_frame = self.sift.detectAndCompute(gray_frame, None)
        
        if des_frame is not None and len(des_frame) > 0:
            # Căutăm potriviri între caracteristicile șablonului și cele din cadru
            matches = self.bf.match(self.des_template, des_frame)
            matches = sorted(matches, key=lambda x: x.distance)

            # Dacă avem suficiente potriviri, detectăm semnul STOP
            if len(matches) > 10:  # Prag pentru potriviri
                self.get_logger().info("Semn STOP detectat!")
                cv2.putText(frame, "STOP DETECTED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Oprim robotul timp de 5 secunde dacă nu s-a oprit deja
                if not self.stop_detected:
                    self.stop_detected = True
                    self.stop_time = time.time()
                return

    def freespace(self, canny_frame, img):
        # Logica pentru oprirea la STOP
        if self.stop_detected and not self.stop_once:
            msg = Twist()
            msg.linear.x = 0.0
            msg.angular.z = 0.0
            self.cmd_vel_pub.publish(msg)
            if time.time() - self.stop_time >= 5:  # După 5 secunde, continuăm drumul
                self.stop_detected = False
                self.stop_once = True  # Marcăm faptul că am oprit deja
                self.get_logger().info("Am oprit la semnul STOP și continui drumul.")
            return

        # Continuă logica de urmărire a benzii
        # Continuă logica de urmărire a benzii dacă semnul STOP a fost deja tratat
        height, width = canny_frame.shape
        DreaptaLim = width // 2
        StangaLim = width // 2

        mask = np.zeros((height, width), dtype=np.uint8)
        contour = []

        # Setăm limita dreapta
        for i in range(width // 2, width - 1):
            if canny_frame[height - 10, i]:
                DreaptaLim = i
                break

        # Setăm limita stânga
        for i in range(width // 2):
            if canny_frame[height - 10, width // 2 - i]:
                StangaLim = width // 2 - i
                break

        # Ajustăm limitele
        if StangaLim == width // 2:
            StangaLim = 1
        if DreaptaLim == width // 2:
            DreaptaLim = width
        contour.append((StangaLim, height - 10))
        cv2.circle(img, (StangaLim, height - 10), 5, (255), -1)
        cv2.circle(img, (DreaptaLim, height - 10), 5, (255), -1)

        for j in range(StangaLim, DreaptaLim + 1, 10):
            for i in range(height - 10, 9, -1):
                if canny_frame[i, j]:
                    cv2.line(img, (j, height - 10), (j, i), (255), 2)
                    contour.append((j, i))
                    break
                if i == 10:
                    contour.append((j, i))
                    cv2.line(img, (j, height - 10), (j, i), (255), 2)
        contour.append((DreaptaLim, height - 10))
        contours = [np.array(contour)]
        cv2.drawContours(mask, contours, 0, (255), cv2.FILLED)
        cv2.imshow("mask", mask)

        # Calculăm centroidul
        M = cv2.moments(contours[0])
        if M["m00"] != 0:
            centroid_x = int(M["m10"] / M["m00"])
            centroid_y = int(M["m01"] / M["m00"])
        else:
            centroid_x, centroid_y = 0, 0

        # Trasează o săgeată către centroid
        cv2.arrowedLine(img, (width // 2, height - 10), (centroid_x, centroid_y), (60, 90, 255), 4)

        error_x = centroid_x - (width // 2)  # Diferența pe axa X
        error_y = height - 10 - centroid_y  # Diferența pe axa Y
        unghi = np.arctan2(error_x, error_y)

        msg = Twist()
        msg.linear.x = 0.3
        msg.angular.z = - unghi * 1.2
        self.cmd_vel_pub.publish(msg)
        

def main(args=None):
    rclpy.init(args=args)
    lane_following = LaneFollowing()
    rclpy.spin(lane_following)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
