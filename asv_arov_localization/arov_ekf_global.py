import rclpy
from rclpy.node import Node
import numpy as np

from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from apriltag_msgs.msg import AprilTagDetectionArray

from tf2_ros import TransformBroadcaster, TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from scipy.spatial.transform import Rotation

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
            ('~depth_noise', [1.0]),                                # Sensor noise values. TODO Should it be the same dimensions as the states?
            ('~compass_noise', [1.0]),
            ('~roll_pitch_noise', [1.0, 1.0]),
            ('~apriltag_noise', [0.1, 0.1, 0.1, 0.01, 0.01, 0.01])
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
        self. correct_noise = {'depth': np.diag(self.get_parameter('~depth_noise').value),
                               'compass': np.diag(self.get_parameter('~compass_noise').value),
                               'pitch_roll': np.diag(self.get_parameter('~roll_pitch_noise').value),
                               'apriltag': np.diag(self.get_parameter('~apriltag_noise').value)}

        self.predict_timer = self.create_timer(0.04, self.predict)
        self.pub_timer = self.create_timer(0.02, self.publish_transform)

    def publish_transform(self):
        if self.state is None: return
        
        self.transform.header.stamp = self.get_clock().now().to_msg()

        try:
            odom_to_base_link = self.tf_buffer.lookup_transform(
                f'{self.arov}/odom',
                f'{self.arov}/base_link',
                rclpy.time.Time())
            
            orientation = Rotation.from_euler('xyz', self.state[3:]) * Rotation.from_quat([odom_to_base_link.transform.rotation.w,
                                                                                           odom_to_base_link.transform.rotation.x,
                                                                                           odom_to_base_link.transform.rotation.y,
                                                                                           odom_to_base_link.transform.rotation.z]).inv()
            
            translation = self.state[:3] - orientation.inv().apply(np.array([odom_to_base_link.transform.translation.x,
                                                                             odom_to_base_link.transform.translation.y,
                                                                             odom_to_base_link.transform.translation.z]))

            orientation = orientation.as_quat()

            self.transform.transform.translation.x = translation[0]
            self.transform.transform.translation.y = translation[1]
            self.transform.transform.translation.z = translation[2]
            self.transform.transform.rotation.w = orientation[0]
            self.transform.transform.rotation.x = orientation[1]
            self.transform.transform.rotation.y = orientation[2]
            self.transform.transform.rotation.z = orientation[3]

        except TransformException as ex:
            self.get_logger().info(
                f'Could not transform odom to base_link: {ex}')
            return

        self.tf_broadcaster.sendTransform(self.transform)

    def arov_pose_callback(self, msg: Odometry):
        self.odom = msg
        
        # self.correct('depth', [False, False, True, False, False, False], np.array([0, 0, msg.pose.pose.position.z, 0, 0, 0]), np.array([0, 0, 1, 0, 0, 0]))
        # self.correct('compass', [False, False, False, False, False, True], np.array([0, 0, 0, 0, 0, msg.pose.pose.orientation.z]),
        #              np.array([0, 0, 0, 0, 0, 1]))
        # self.correct('pitch_roll', [False, False, False, True, True, False], np.array([0, 0, 0, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, 0]),
        #              np.array([[0, 0, 0, 0, 1, 0],
        #                        [0, 0, 0, 0, 0, 1]]))

    def arov_apriltag_detect_callback(self, msg: AprilTagDetectionArray):
        '''
        Runs full state correction using AprilTag detections whenever one or more AprilTags are observed.  Assumes a map of true AprilTag transforms
        is provided (can be static or dynamic).
        '''
        if msg.detections:
            for tag in msg.detections:
                tag_frame = f'{tag.family}:{tag.id}'
                
                try:
                    base_link_to_tag = self.tf_buffer.lookup_transform(
                        f'{self.arov}/base_link',
                        tag_frame,
                        rclpy.time.Time())
                    
                    tag_to_base_link = self.tf_buffer.lookup_transform(
                        tag_frame,
                        f'{self.arov}/base_link',
                        rclpy.time.Time())

                    map_to_tag = self.tf_buffer.lookup_transform(
                        'map',
                        f'{tag_frame}_true',
                        rclpy.time.Time())

                    base_link_to_map = self.tf_buffer.lookup_transform(
                        f'{self.arov}/base_link',
                        'map',
                        rclpy.time.Time())
                    
                    observed_orientation = Rotation.from_quat([map_to_tag.transform.rotation.x, map_to_tag.transform.rotation.y,
                                                                map_to_tag.transform.rotation.z, map_to_tag.transform.rotation.w]) * \
                                            Rotation.from_quat([base_link_to_tag.transform.rotation.x, base_link_to_tag.transform.rotation.y,
                                                                base_link_to_tag.transform.rotation.z, base_link_to_tag.transform.rotation.w]).inv()
                    
                    tag_in_map = observed_orientation.apply(np.array([base_link_to_tag.transform.translation.x, base_link_to_tag.transform.translation.y, 
                                                                     base_link_to_tag.transform.translation.z]), True)

                    observation = np.array([map_to_tag.transform.translation.x - tag_in_map[0],
                                            map_to_tag.transform.translation.y - tag_in_map[1],
                                            map_to_tag.transform.translation.z - tag_in_map[2],
                                            *observed_orientation.inv().as_euler('xyz')])

                    # observed_orientation = (Rotation.from_quat([map_to_tag.transform.rotation.x, map_to_tag.transform.rotation.y,
                    #                                             map_to_tag.transform.rotation.z, map_to_tag.transform.rotation.w]) * \
                    #                         Rotation.from_quat([base_link_to_tag.transform.rotation.x, base_link_to_tag.transform.rotation.y,
                    #                                             base_link_to_tag.transform.rotation.z, base_link_to_tag.transform.rotation.w]).inv())\
                    #                                             .as_euler('xyz')
                    
                    # tag_in_map = Rotation.from_quat([base_link_to_map.transform.rotation.x, base_link_to_map.transform.rotation.y,
                    #                                  base_link_to_map.transform.rotation.z, base_link_to_map.transform.rotation.w])\
                    #                                 .apply(np.array([base_link_to_tag.transform.translation.x, base_link_to_tag.transform.translation.y, 
                    #                                                  base_link_to_tag.transform.translation.z]), True)

                    # observation = np.array([map_to_tag.transform.translation.x - tag_in_map[0],
                    #                         map_to_tag.transform.translation.y - tag_in_map[1],
                    #                         map_to_tag.transform.translation.z - tag_in_map[2],
                    #                         *observed_orientation])
                    
                    h_jacobian = np.array([[1, 0, 0, 0, 0, 0],
                                           [0, 1, 0, 0, 0, 0],
                                           [0, 0, 1, 0, 0, 0],
                                           [0, 0, 0, 1, 0, 0],
                                           [0, 0, 0, 0, 1, 0],
                                           [0, 0, 0, 0, 0, 1]])

                    self.correct('apriltag', [True, True, True, True, True, True], observation, h_jacobian)

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
        if self.state is None:
            if self.odom is not None:
                orientation = Rotation.from_quat([self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y,
                                                  self.odom.pose.pose.orientation.z, self.odom.pose.pose.orientation.w]).as_euler('xyz')
                self.state = np.array([self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z,
                                       *orientation])
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
        if self.state is None: return

        err = np.atleast_2d(np.extract(state_mask, observation - self.state)).transpose()
        err_cov = (h_jacobian @ self.cov @ h_jacobian.transpose()) + self.correct_noise[observation_name]
        K_gain = (self.cov @ np.atleast_2d(h_jacobian).transpose()) @ np.linalg.inv(err_cov)

        state_correction = np.zeros_like(self.state)
        np.place(state_correction, state_mask, K_gain @ err)

        self.state += state_correction
        self.cov = (np.eye(np.shape(self.cov)[0]) - K_gain @ np.atleast_2d(h_jacobian)) @ self.cov

        # Normalize rotations between -pi to pi
        self.state[3:] = np.arctan2(np.sin(self.state[3:]), np.cos(self.state[3:]))


def main():    
    rclpy.init()

    arov_ekf_global = AROV_EKF_Global()

    rclpy.spin(arov_ekf_global)

    arov_ekf_global.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()