import rclpy
from rclpy.node import Node
import numpy as np

from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from apriltag_msgs.msg import AprilTagDetectionArray

from tf2_ros import TransformBroadcaster, TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

class AROV_EKF_Global(Node):
    '''
    Node to run global position estimate for the AROV using fixed, known AprilTag locations.
    '''
    def __init__(self):
        super().__init__('arov_ekf_global')
        self.declare_parameters(namespace='',parameters=[
            ('~vehicle_name', 'arov'),                              # Used for the topics and services that this node subscribes to and provides
            ('~initial_cov', [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),       # 
            ('~predict_noise', [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),     # Diagonals of the covariance matrix for prediction noise (if there should
                                                                    # be non-zero covariance, change where this parameter is used from diag to array)
            ('~depth_noise', [0, 0, 1.0, 0, 0, 0]),                 # Sensor noise values.  Should be the same dimensions as the states
            ('~compass_noise', [0, 0, 0, 0, 0, 1.0]),
            ('~roll_pitch_noise', [0, 0, 0, 1.0, 1.0, 0]),
            ('~apriltag_noise', [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        ])

        self.arov = self.get_parameter('~vehicle_name').value

        self.arov_pose_sub = self.create_subscription(
            Odometry,
            f'{self.arov}/odom',
            self.arov_pose_callback,
            10)
        self.arov_pose_sub  # prevent unused variable warning

        self.arov_apriltag_detect_sub = self.create_subscription(
            AprilTagDetectionArray,
            '/detections',
            self.arov_apriltag_detect_callback,
            10)
        self.arov_apriltag_detect_sub  # prevent unused variable warning

        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.odom = None

        self.transform = TransformStamped()
        self.transform.header.frame_id = 'map'
        self.transform.child_frame_id = f'{self.arov}/odom'

        self.time_km1 = None
        self.state = None                                                                       # State estimate: x, y, z, roll, pitch, yaw
        self.cov = np.diag(self.get_parameter('~initial_cov').value)                            # Covariance estimate
        self.predict_noise = np.diag(self.get_parameter('~predict_noise').value)
        self. correct_noise = {'depth': np.array(self.get_parameter('~depth_noise').value),
                               'compass': np.array(self.get_parameter('~compass_noise').value),
                               'pitch_roll': np.array(self.get_parameter('~roll_pitch_noise').value),
                               'apriltag': np.array(self.get_parameter('~apriltag').value)}

        self.pub_timer = self.create_timer(0.02, self.publish_transform)

    def publish_transform(self):
        self.transform.header.stamp = self.get_clock().now().to_msg()

        try:
            map_to_odom = self.tf_buffer.lookup_transform(
                f'{self.arov}/odom',
                'map',
                rclpy.time.Time())
            
            self.transform.transform.translation.x = self.state[0] - map_to_odom.transform.translation.x
            self.transform.transform.translation.y = self.state[1] - map_to_odom.transform.translation.y
            self.transform.transform.translation.z = self.state[2] - map_to_odom.transform.translation.z
            self.transform.transform.rotation.x = self.state[3] - map_to_odom.transform.rotation.x
            self.transform.transform.rotation.y = self.state[4] - map_to_odom.transform.rotation.y
            self.transform.transform.rotation.z = self.state[5] - map_to_odom.transform.rotation.z

        except TransformException as ex:
            self.get_logger().info(
                f'Could not transform map to odom: {ex}')
            return

        self.tf_broadcaster.sendTransform(self.transform)

    def arov_pose_callback(self, msg: Odometry):
        self.odom = msg
        
        self.correct('depth', [False, False, True, False, False, False], np.array([0, 0, msg.pose.pose.position.z, 0, 0, 0]), np.array([0, 0, 1, 0, 0, 0]))
        self.correct('compass', [False, False, False, False, False, True], np.array([0, 0, 0, 0, 0, msg.pose.pose.orientation.z]),
                     np.array([0, 0, 0, 0, 0, 1]))
        self.correct('pitch_roll', [False, False, False, True, True, False], np.array([0, 0, 0, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, 0]),
                     np.array([0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 1]))

    def arov_apriltag_detect_callback(self, msg: AprilTagDetectionArray):
        # TODO Add correct for orientation from AprilTags
        if msg.detections:
            for tag in msg.detections:
                tag_frame = f'{tag.family}:{tag.id}'
                
                try:
                    base_link_to_tag = self.tf_buffer.lookup_transform(
                        tag_frame,
                        f'{self.arov}/base_link',
                        rclpy.time.Time())
                    
                    map_to_tag = self.tf_buffer.lookup_transform(
                        tag_frame,
                        'map',
                        rclpy.time.Time())

                    observation = np.array([map_to_tag.transform.translation.x - base_link_to_tag.transform.translation.x,
                                            map_to_tag.transform.translation.y - base_link_to_tag.transform.translation.y,
                                            map_to_tag.transform.translation.z - base_link_to_tag.transform.translation.z,
                                            0, 0, 0])
                    
                    h_jacobian = np.array([[1, 0, 0, 0, 0, 0],
                                           [0, 1, 0, 0, 0, 0],
                                           [0, 0, 1, 0, 0, 0]])

                    self.correct('apriltag', [True, True, True, False, False, False], observation, h_jacobian)

                except TransformException as ex:
                    self.get_logger().info(
                        f'Could not transform {f'{self.arov}/base_link'} or map to {tag_frame}: {ex}')
                    return

    def predict(self):
        '''
        Propogate forward the position estimate and convariance.

        Needs:
            - Linear velocity
            - Angular velocity
            - Orientation
            - Delta time
        '''
        if self.state == None:
            if self.odom != None:
                self.state = np.array([self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z,
                                       self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y, self.odom.pose.pose.orientation.z])
                self.time_km1 = self.get_clock().now().nanoseconds
            return
        
        # Vel is ENU in the odom frame TODO May need to rotate this into the map frame if orientation is not close enough between map and odom
        vel = np.array([self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z,
                        self.odom.twist.twist.angular.x, self.odom.twist.twist.angular.y, self.odom.twist.twist.angular.z])
        
        time_k = self.get_clock().now().nanoseconds
        dt =  (time_k - self.time_km1) / 10.0**9                                # Time since last prediction
        self.time_km1 = time_k

        self.state += vel * dt

        F = np.diag([1, 1, 1, 1, 1, 1])                                         # TODO if a rotation is added in vel, account for it here
        self.cov = F @ self.cov @ F.transpose() + (dt * self.predict_noise)

    def correct(self, observation_name: str, state_mask: list[bool], observation: np.ndarray, h_jacobian: np.ndarray):
        '''
        Generic correction for a state to run whenever an observation comes in.

        Args:
            observation_name (str): Name that is used for the state or states being corrected.
            state_mask (list[bool]): Mask for which states are being observed.  Same length as the number of states being estimated.
            observation (ndarray): The observation being used to correct the state estimate.  Dimensions should match the states being estimated.
            h_jacobian (ndarray): The jacobian of the observation function.
        '''
        err = np.array(state_mask, dtype=int) * (observation - self.state)
        err_cov = h_jacobian @ self.cov @ h_jacobian.transpose() + self.correct_noise[observation_name]
        K_gain = (self.cov @ h_jacobian.transpose()) @ err_cov.inv()

        self.state += (K_gain * err)
        self.cov = (np.eye(np.shape(self.cov)[0]) - K_gain @ h_jacobian) @ self.cov


def main():    
    rclpy.init()

    arov_ekf_global = AROV_EKF_Global()

    rclpy.spin(arov_ekf_global)

    arov_ekf_global.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()