import rclpy
from rclpy.node import Node
import numpy as np

from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from apriltag_msgs.msg import AprilTagDetectionArray
from sensor_msgs.msg import Imu

from tf2_ros import TransformBroadcaster

class AROV_EKF_Global(Node):
    '''
    Node to run global position estimate for the AROV using fixed, known AprilTag locations.
    '''
    def __init__(self):
        super().__init__('arov_ekf_global')
        self.declare_parameters(namespace='',parameters=[
            ('vehicle_name', 'arov'),                       # Used for the topics and services that this node subscribes to and provides
        ])

        self.arov = self.get_parameter('vehicle_name').value

        self.arov_pose_sub = self.create_subscription(
            Odometry,
            f'{self.arov}/odom',
            self.arov_pose_callback,
            10)
        self.arov_pose_sub  # prevent unused variable warning

        self.arov_imu_sub = self.create_subscription(
            Imu,
            f'{self.arov}/imu',
            self.arov_imu_callback,
            10)
        self.arov_imu_sub  # prevent unused variable warning

        self.arov_apriltag_detect_sub = self.create_subscription(
            AprilTagDetectionArray,
            '/detections',
            self.arov_apriltag_detect_callback,
            10)
        self.arov_apriltag_detect_sub  # prevent unused variable warning

        self.tf_broadcaster = TransformBroadcaster(self)

        self.odom = Odometry()
        self.imu = Imu()

        self.transform = TransformStamped()
        self.transform.header.frame_id = 'map'
        self.transform.child_frame_id = f'{self.arov}/odom'

        self.time_km1 = self.get_clock().now().nanoseconds
        self.state = np.zeros((6))                              # State estimate: x, y, z, roll, pitch, yaw
        self.cov = np.diag([1, 1, 1, 1, 1, 1])                  # Covariance estimate

        self.pub_timer = self.create_timer(0.02, self.publish_transform)

    def publish_transform(self):
        self.transform.header.stamp = self.get_clock().now().to_msg()

        self.tf_broadcaster.sendTransform(self.transform)

    def arov_pose_callback(self, msg: Odometry):
        self.odom = msg

    def arov_imu_callback(self, msg: Imu):
        self.imu = msg

    def arov_apriltag_detect_callback(self, msg: AprilTagDetectionArray):
        pass

    def predict(self):
        '''
        Propogate forward the position estimate and convariance.

        Needs:
            - Linear velocity
            - Angular velocity
            - Orientation
            - Delta time
        '''
        # Vel is ENU in the odom frame
        vel = np.array([self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z,
                        self.odom.twist.twist.angular.x, self.odom.twist.twist.angular.y, self.odom.twist.twist.angular.z])
        
        dt =  (self.get_clock().now().nanoseconds - self.time_km1) / 10.0**9            # Time since last prediction
        self.time_km1 = self.get_clock().now().nanoseconds

        self.state += vel * dt

    def correct(self):
        # Proabably just handle this in the callback functions
        pass

def main():    
    rclpy.init()

    arov_ekf_global = AROV_EKF_Global()

    rclpy.spin(arov_ekf_global)

    arov_ekf_global.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()