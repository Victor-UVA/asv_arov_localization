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
            ('~ros_bag', True),                                         # Toggle for using bagged data and switching to sending test transforms
            ('~initial_cov', [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            ('~predict_noise', [0.5, 0.5, 0.5, 0.75, 0.75, 0.75]),   # Diagonals of the covariance matrix for prediction noise (if there should
                                                                        # be non-zero covariance, change where this parameter is used from diag to array)
            ('~depth_noise', [0.75]),                                    # Sensor noise values.
            ('~compass_noise', [0.5]),
            ('~roll_pitch_noise', [0.125, 0.125]),
            ('~apriltag_noise', [0.75, 0.75, 0.75, 0.25, 0.25, 0.25])
        ])

        self.ros_bag = self.get_parameter('~ros_bag').value
        self.arov = self.get_namespace().strip('/')

        self.arov_pose_sub = self.create_subscription(
            Odometry,
            f'{self.get_namespace()}/odom',
            self.arov_pose_callback,
            10)
        self.arov_pose_sub  # prevent unused variable warning

        self.arov_apriltag_detect_sub = self.create_subscription(
            AprilTagDetectionArray,
            f'{self.get_namespace()}/detections',
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
        self.predict_noise = np.power(np.diag(self.get_parameter('~predict_noise').value), 2)
        self.correct_noise = {'depth': np.diag(self.get_parameter('~depth_noise').value),
                              'compass': np.diag(self.get_parameter('~compass_noise').value),
                              'roll_pitch': np.diag(self.get_parameter('~roll_pitch_noise').value),
                              'apriltag': np.diag(self.get_parameter('~apriltag_noise').value)}
        
        for noise in self.correct_noise:
            self.correct_noise[noise] = np.power(self.correct_noise[noise], 2)

        self.predict_timer = self.create_timer(0.01, self.predict)
        self.pub_timer = self.create_timer(0.02, self.publish_transform)

    def publish_transform(self):
        if self.state is None: return
        
        self.transform.header.stamp = self.get_clock().now().to_msg()

        try:
            odom_to_base_link = self.tf_buffer.lookup_transform(
                f'{self.arov}/odom',
                f'{self.arov}/base_link',
                rclpy.time.Time())
            
            orientation = Rotation.from_euler('xyz', self.state[3:]) * Rotation.from_quat([odom_to_base_link.transform.rotation.x,
                                                                                           odom_to_base_link.transform.rotation.y,
                                                                                           odom_to_base_link.transform.rotation.z,
                                                                                           odom_to_base_link.transform.rotation.w]).inv()
            
            translation = self.state[:3] - orientation.apply(np.array([odom_to_base_link.transform.translation.x,
                                                                       odom_to_base_link.transform.translation.y,
                                                                       odom_to_base_link.transform.translation.z]), False)

            orientation = orientation.as_quat()

            self.transform.transform.translation.x = translation[0]
            self.transform.transform.translation.y = translation[1]
            self.transform.transform.translation.z = translation[2]
            self.transform.transform.rotation.x = orientation[0]
            self.transform.transform.rotation.y = orientation[1]
            self.transform.transform.rotation.z = orientation[2]
            self.transform.transform.rotation.w = orientation[3]

            if not self.ros_bag:
                self.tf_broadcaster.sendTransform(self.transform)
            else:
                self.bag_testing(odom_to_base_link)

        except TransformException as ex:
            self.get_logger().info(
                f'Could not transform odom to base_link: {ex}')
            return

    def arov_pose_callback(self, msg: Odometry):
        self.odom = msg

        if self.state is None:
            return

        try:
            odom_to_base_link = self.tf_buffer.lookup_transform(
                f'{self.arov}/odom',
                f'{self.arov}/base_link',
                rclpy.time.Time())
            
            msg_orientation = Rotation.from_quat([msg.pose.pose.orientation.x,
                                                  msg.pose.pose.orientation.y,
                                                  msg.pose.pose.orientation.z,
                                                  msg.pose.pose.orientation.w])

            odom_to_global_rot = Rotation.from_euler('xyz', self.state[3:]).inv() * Rotation.from_quat([odom_to_base_link.transform.rotation.x,
                                                                                                        odom_to_base_link.transform.rotation.y,
                                                                                                        odom_to_base_link.transform.rotation.z,
                                                                                                        odom_to_base_link.transform.rotation.w])

            odom_position = np.array([msg.pose.pose.position.x,
                                      msg.pose.pose.position.y,
                                      msg.pose.pose.position.z])
            
            global_depth = odom_to_global_rot.apply(odom_position)[2]
            global_rot = (odom_to_global_rot * msg_orientation).as_euler('xyz')

            self.correct('depth', [False, False, True, False, False, False], np.array([0, 0, global_depth, 0, 0, 0]),
                         np.array([0, 0, 1, 0, 0, 0]))
            
            self.correct('compass', [False, False, False, False, False, True],
                         np.array([0, 0, 0, 0, 0, global_rot[2]]),
                         np.array([0, 0, 0, 0, 0, 1]))
            
            self.correct('roll_pitch', [False, False, False, True, True, False],
                         np.array([0, 0, 0, global_rot[0], global_rot[1], 0]),
                         np.array([[0, 0, 0, 1, 0, 0],
                                   [0, 0, 0, 0, 1, 0]]))
        
        except TransformException as ex:
            self.get_logger().info(
                f'Could not transform odom to base_link: {ex}')
            return

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

                    map_to_tag = self.tf_buffer.lookup_transform(
                        'map',
                        f'{tag_frame}_true',
                        rclpy.time.Time())
                    
                    observed_orientation = Rotation.from_quat([map_to_tag.transform.rotation.x, map_to_tag.transform.rotation.y,
                                                                map_to_tag.transform.rotation.z, map_to_tag.transform.rotation.w]) * \
                                           Rotation.from_quat([base_link_to_tag.transform.rotation.x, base_link_to_tag.transform.rotation.y,
                                                                base_link_to_tag.transform.rotation.z, base_link_to_tag.transform.rotation.w]).inv()
                    
                    tag_in_map = observed_orientation.apply(np.array([base_link_to_tag.transform.translation.x,
                                                                      base_link_to_tag.transform.translation.y, 
                                                                      base_link_to_tag.transform.translation.z]), False)

                    observation = np.array([map_to_tag.transform.translation.x - tag_in_map[0],
                                            map_to_tag.transform.translation.y - tag_in_map[1],
                                            map_to_tag.transform.translation.z - tag_in_map[2],
                                            *observed_orientation.as_euler('xyz')])
                    
                    h_jacobian = np.array([[1, 0, 0, 0, 0, 0],
                                           [0, 1, 0, 0, 0, 0],
                                           [0, 0, 1, 0, 0, 0],
                                           [0, 0, 0, 1, 0, 0],
                                           [0, 0, 0, 0, 1, 0],
                                           [0, 0, 0, 0, 0, 1]])

                    self.correct('apriltag', [True, True, True, True, True, True], observation, h_jacobian)

                    if self.ros_bag:
                        base_link_to_tag.header.frame_id = f'{self.arov}_bag/base_link'
                        base_link_to_tag.child_frame_id = f'{tag_frame}_bag'

                        base_link_to_tag.header.stamp = self.get_clock().now().to_msg()

                        self.tf_broadcaster.sendTransform(base_link_to_tag)

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
        

        try:
            odom_to_base_link = self.tf_buffer.lookup_transform(
                f'{self.arov}/odom',
                f'{self.arov}/base_link',
                rclpy.time.Time())
            
            time_k = self.get_clock().now().nanoseconds
            dt =  (time_k - self.time_km1) / 10.0**9                                # Time since last prediction
            self.time_km1 = time_k

            orientation = Rotation.from_euler('xyz', self.state[3:]) * Rotation.from_quat([odom_to_base_link.transform.rotation.x,
                                                                                           odom_to_base_link.transform.rotation.y,
                                                                                           odom_to_base_link.transform.rotation.z,
                                                                                           odom_to_base_link.transform.rotation.w]).inv()

            linear_vel = np.array([self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z])
            angular_vel = np.array([self.odom.twist.twist.angular.x, self.odom.twist.twist.angular.y, self.odom.twist.twist.angular.z])

            self.state[:3] += orientation.apply(linear_vel) * dt
            self.state[3:] += Rotation.from_euler('xyz', angular_vel * dt).as_euler('xyz')

            # Normalize rotations between -pi to pi
            self.state[3:] = np.arctan2(np.sin(self.state[3:]), np.cos(self.state[3:]))

            F = np.diag([1, 1, 1, 1, 1, 1])
            self.cov = F @ self.cov @ F.transpose() + (dt * self.predict_noise)
        
        except TransformException as ex:
            self.get_logger().info(
                f'Could not transform odom to base_link: {ex}')
            return

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
        K_gain = self.cov @ np.atleast_2d(h_jacobian).transpose() @ np.linalg.inv(err_cov)

        state_correction = np.zeros_like(self.state)
        np.copyto(state_correction, (K_gain @ err).transpose()) # , where=state_mask

        self.state += state_correction
        self.cov = (np.eye(np.shape(self.cov)[0]) - K_gain @ np.atleast_2d(h_jacobian)) @ self.cov

        # Normalize rotations between -pi to pi
        self.state[3:] = np.arctan2(np.sin(self.state[3:]), np.cos(self.state[3:]))

    def bag_testing(self, odom_to_base_link: TransformStamped):
        bag_odom = self.transform
        bag_odom.child_frame_id = f'{self.arov}_bag/odom'

        odom_to_base_link.header.frame_id = f'{self.arov}_bag/odom'
        odom_to_base_link.child_frame_id = f'{self.arov}_bag/base_link'

        odom_to_base_link.header.stamp = self.get_clock().now().to_msg()
        
        self.tf_broadcaster.sendTransform(bag_odom)
        self.tf_broadcaster.sendTransform(odom_to_base_link)


def main():    
    rclpy.init()

    arov_ekf_global = AROV_EKF_Global()

    rclpy.spin(arov_ekf_global)

    arov_ekf_global.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()