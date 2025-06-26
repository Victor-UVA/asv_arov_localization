import rclpy
from rclpy.node import Node
import numpy as np
import math
from collections import deque

from geometry_msgs.msg import TransformStamped, Quaternion
from nav_msgs.msg import Odometry
from apriltag_msgs.msg import AprilTagDetectionArray

import rclpy.time
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
        self.declare_parameters(namespace='', parameters=[
            ('~ros_bag', True),  # Toggle for using bagged data and switching to sending test transforms
            ('~initial_cov', [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            ('~predict_noise', [0.05, 0.05, 0.05]),                                 # Diagonals of the covariance matrix for the linear portion of prediction noise
            ('~gyro_noise', [0.025, 0.025, 0.025]),                                 # Used to compute the angular portion of prediction noise
            ('~apriltag_noise', [0.5, 0.5, 0.5, 0.025, 0.025, 0.025, 0.025]),
            ('~linear_vel_scale', 2.0),                                             # Scaling factors for AprilTag noise
            ('~angular_vel_scale', 4.0),
            ('~tag_dist_scale', 3.0)
        ])

        self.ros_bag = self.get_parameter('~ros_bag').value
        self.arov = self.get_namespace().strip('/')
        if self.ros_bag:
            self.odom_frame = f'{self.arov}_bag/odom'
            self.base_link_frame = f'{self.arov}_bag/base_link'
        else:
            self.odom_frame = f'{self.arov}/odom'
            self.base_link_frame = f'{self.arov}/base_link'

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
        self.odom_to_base_link_km1 = None

        self.transform = TransformStamped()
        self.transform.header.frame_id = 'map'
        self.transform.child_frame_id = f'{self.arov}/odom'

        self.time_km1 = None
        self.state = None  # State estimate: x, y, z, q(w, x, y, z)
        self.cov = np.diag(self.get_parameter('~initial_cov').value)  # Covariance estimate

        self.predict_timer = self.create_timer(1.0 / 50.0, self.predict)
        self.pub_timer = self.create_timer(1.0 / 50.0, self.publish_transform)

        self.tag_data = {}  # id / translation / orientation / filtered translation / filtered orientation
        pose_f0 = 0.1
        Fs = 25
        pose_alpha = 1 - math.exp(-2 * math.pi * pose_f0/ Fs)
        self.pose_coeffs = [[1.2*pose_alpha, -0.2*pose_alpha], [1.0, pose_alpha-1]]
        self.base_quat_alpha = 0.1
        self.quat_sens = 0.5

    def publish_transform(self):
        if self.state is None: return

        odom_to_base_link = None

        try:
            odom_to_base_link = self.tf_buffer.lookup_transform(
                f'{self.arov}/odom',
                f'{self.arov}/base_link',
                rclpy.time.Time())

        except TransformException as ex:
            return

        if odom_to_base_link is not None:
            orientation = Rotation.from_quat([*self.state[4:], self.state[3]]) * Rotation.from_quat(
                [odom_to_base_link.transform.rotation.x,
                 odom_to_base_link.transform.rotation.y,
                 odom_to_base_link.transform.rotation.z,
                 odom_to_base_link.transform.rotation.w]).inv()

            translation = self.state[:3] - orientation.apply(np.array([odom_to_base_link.transform.translation.x,
                                                                       odom_to_base_link.transform.translation.y,
                                                                       odom_to_base_link.transform.translation.z]),
                                                             False)

            orientation = orientation.as_quat()

            self.transform.header.stamp = self.get_clock().now().to_msg()
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

    def arov_pose_callback(self, msg: Odometry):
        self.odom = msg

    def arov_apriltag_detect_callback(self, msg: AprilTagDetectionArray):
        '''
        Runs full state correction using AprilTag detections whenever one or more AprilTags are observed.  Assumes a map of true AprilTag transforms
        is provided (can be static or dynamic).
        '''
        if msg.detections:
            for tag in msg.detections:
                if tag.decision_margin < 20.0:
                    continue

                tag_frame = f'{tag.family}:{tag.id}'

                base_link_to_tag = None
                map_to_tag = None

                try:
                    base_link_to_tag = self.tf_buffer.lookup_transform(
                        f'{self.arov}/base_link',
                        tag_frame,
                        rclpy.time.Time())
                except TransformException as ex:
                    continue

                try:
                    map_to_tag = self.tf_buffer.lookup_transform(
                        'map',
                        f'{tag_frame}_true',
                        rclpy.time.Time())
                except TransformException as ex:
                    continue

                if base_link_to_tag is not None and map_to_tag is not None:
                    base_link_to_tag = self.lowpass_filter(tag.id, base_link_to_tag)
                    observation = np.array([[base_link_to_tag.transform.translation.x],
                                            [base_link_to_tag.transform.translation.y],
                                            [base_link_to_tag.transform.translation.z],
                                            [base_link_to_tag.transform.rotation.w],
                                            [base_link_to_tag.transform.rotation.x],
                                            [base_link_to_tag.transform.rotation.y],
                                            [base_link_to_tag.transform.rotation.z]])

                    diff = np.array([map_to_tag.transform.translation.x - self.state[0],
                                     map_to_tag.transform.translation.y - self.state[1],
                                     map_to_tag.transform.translation.z - self.state[2]])

                    h_expected = np.array([[diff[0] * (
                                self.state[3] ** 2 + self.state[4] ** 2 - self.state[5] ** 2 - self.state[
                            6] ** 2) + 2 * (-self.state[3] * self.state[5] * diff[2] + self.state[3] * self.state[6] *
                                            diff[1] + self.state[4] * self.state[5] * diff[1] + self.state[4] *
                                            self.state[6] * diff[2])],
                                           [diff[1] * (self.state[3] ** 2 - self.state[4] ** 2 + self.state[5] ** 2 -
                                                       self.state[6] ** 2) + 2 * (
                                                        -self.state[3] * self.state[6] * diff[0] + self.state[3] *
                                                        self.state[4] * diff[2] + self.state[5] * self.state[4] * diff[
                                                            0] + self.state[5] * self.state[6] * diff[2])],
                                           [diff[2] * (self.state[3] ** 2 - self.state[4] ** 2 - self.state[5] ** 2 +
                                                       self.state[6] ** 2) + 2 * (
                                                        -self.state[3] * self.state[4] * diff[1] + self.state[3] *
                                                        self.state[5] * diff[0] + self.state[6] * self.state[4] * diff[
                                                            0] + self.state[6] * self.state[5] * diff[1])],
                                           [self.state[3] * map_to_tag.transform.rotation.w + self.state[
                                               4] * map_to_tag.transform.rotation.x + self.state[
                                                5] * map_to_tag.transform.rotation.y + self.state[
                                                6] * map_to_tag.transform.rotation.z],
                                           [-self.state[4] * map_to_tag.transform.rotation.w + self.state[
                                               3] * map_to_tag.transform.rotation.x + self.state[
                                                6] * map_to_tag.transform.rotation.y - self.state[
                                                5] * map_to_tag.transform.rotation.z],
                                           [-self.state[5] * map_to_tag.transform.rotation.w - self.state[
                                               6] * map_to_tag.transform.rotation.x + self.state[
                                                3] * map_to_tag.transform.rotation.y + self.state[
                                                4] * map_to_tag.transform.rotation.z],
                                           [-self.state[6] * map_to_tag.transform.rotation.w + self.state[
                                               5] * map_to_tag.transform.rotation.x - self.state[
                                                4] * map_to_tag.transform.rotation.y + self.state[
                                                3] * map_to_tag.transform.rotation.z]])

                    H_jacobian = np.array([[-(
                                self.state[3] ** 2 + self.state[4] ** 2 - self.state[5] ** 2 - self.state[6] ** 2),
                                            -2 * (self.state[3] * self.state[6] + self.state[4] * self.state[5]),
                                            2 * (self.state[3] * self.state[5] - self.state[4] * self.state[6]), 2 * (
                                                        diff[0] * self.state[3] - diff[2] * self.state[5] + diff[1] *
                                                        self.state[6]), 2 * (
                                                        diff[0] * self.state[4] + diff[1] * self.state[5] + diff[2] *
                                                        self.state[6]), 2 * (
                                                        -diff[0] * self.state[5] - diff[2] * self.state[3] + diff[1] *
                                                        self.state[4]), 2 * (
                                                        -diff[0] * self.state[6] + diff[1] * self.state[3] + diff[2] *
                                                        self.state[4])],
                                           [2 * (self.state[3] * self.state[6] - self.state[5] * self.state[4]), -(
                                                       self.state[3] ** 2 - self.state[4] ** 2 + self.state[5] ** 2 -
                                                       self.state[6] ** 2),
                                            -2 * (self.state[3] * self.state[4] + self.state[5] * self.state[6]), 2 * (
                                                        diff[1] * self.state[3] - diff[0] * self.state[6] + diff[2] *
                                                        self.state[4]), 2 * (
                                                        -diff[1] * self.state[4] + diff[2] * self.state[3] + diff[0] *
                                                        self.state[5]), 2 * (
                                                        diff[1] * self.state[5] + diff[0] * self.state[4] + diff[2] *
                                                        self.state[6]), 2 * (
                                                        -diff[1] * self.state[6] - diff[0] * self.state[3] + diff[2] *
                                                        self.state[5])],
                                           [-2 * (self.state[3] * self.state[5] + self.state[6] * self.state[4]),
                                            2 * (self.state[3] * self.state[4] - self.state[6] * self.state[5]), -(
                                                       self.state[3] ** 2 - self.state[4] ** 2 - self.state[5] ** 2 +
                                                       self.state[6] ** 2), 2 * (
                                                        diff[2] * self.state[3] - diff[1] * self.state[4] + diff[0] *
                                                        self.state[5]), 2 * (
                                                        -diff[2] * self.state[4] - diff[1] * self.state[3] + diff[0] *
                                                        self.state[6]), 2 * (
                                                        -diff[2] * self.state[5] + diff[0] * self.state[3] + diff[1] *
                                                        self.state[6]), 2 * (
                                                        diff[2] * self.state[6] + diff[0] * self.state[4] + diff[1] *
                                                        self.state[5])],
                                           [0, 0, 0, map_to_tag.transform.rotation.w, map_to_tag.transform.rotation.x,
                                            map_to_tag.transform.rotation.y, map_to_tag.transform.rotation.z],
                                           [0, 0, 0, map_to_tag.transform.rotation.x, -map_to_tag.transform.rotation.w,
                                            -map_to_tag.transform.rotation.z, map_to_tag.transform.rotation.y],
                                           [0, 0, 0, map_to_tag.transform.rotation.y, map_to_tag.transform.rotation.z,
                                            -map_to_tag.transform.rotation.w, -map_to_tag.transform.rotation.x],
                                           [0, 0, 0, map_to_tag.transform.rotation.z, -map_to_tag.transform.rotation.y,
                                            map_to_tag.transform.rotation.x, -map_to_tag.transform.rotation.w]])

                    observation_noise = np.power(np.array([[0.1, 0, 0, 0, 0, 0, 0],
                                                           [0, 0.1, 0, 0, 0, 0, 0],
                                                           [0, 0, 0.1, 0, 0, 0, 0],
                                                           [0, 0, 0, 0.5, 0.0, 0.0, 0.0],
                                                           [0, 0, 0, 0.0, 0.5, 0.0, 0.0],
                                                           [0, 0, 0, 0.0, 0.0, 0.5, 0.0],
                                                           [0, 0, 0, 0.0, 0.0, 0.0, 0.5]]), 2)

                    # linear_vel = np.linalg.norm([self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z])
                    # angular_vel = np.linalg.norm([self.odom.twist.twist.angular.x, self.odom.twist.twist.angular.y, self.odom.twist.twist.angular.z])
                    # tag_dist = np.linalg.norm([base_link_to_tag.transform.translation.x, base_link_to_tag.transform.translation.y, base_link_to_tag.transform.translation.z])

                    # observation_noise *= self.get_parameter('~linear_vel_scale').value * linear_vel *\
                    #                      angular_vel ** self.get_parameter('~angular_vel_scale').value *\
                    #                      tag_dist ** self.get_parameter('~tag_dist_scale').value

                    self.correct(observation, observation_noise, h_expected, H_jacobian)

                    # self.get_logger().info(f'{self.state}')

                    if self.ros_bag:
                        observed_orientation = Rotation.from_quat([map_to_tag.transform.rotation.x,
                                                                   map_to_tag.transform.rotation.y,
                                                                   map_to_tag.transform.rotation.z,
                                                                   map_to_tag.transform.rotation.w]) * \
                                               Rotation.from_quat([base_link_to_tag.transform.rotation.x,
                                                                   base_link_to_tag.transform.rotation.y,
                                                                   base_link_to_tag.transform.rotation.z,
                                                                   base_link_to_tag.transform.rotation.w]).inv()

                        tag_in_map = observed_orientation.apply(np.array([base_link_to_tag.transform.translation.x,
                                                                          base_link_to_tag.transform.translation.y,
                                                                          base_link_to_tag.transform.translation.z]))

                        observed_orientation = observed_orientation.as_quat()

                        # observation = np.array([map_to_tag.transform.translation.x - tag_in_map[0],
                        #                         map_to_tag.transform.translation.y - tag_in_map[1],
                        #                         map_to_tag.transform.translation.z - tag_in_map[2],
                        #                         observed_orientation[3],
                        #                         observed_orientation[0],
                        #                         observed_orientation[1],
                        #                         observed_orientation[2]])

                        observation = h_expected

                        arov_in_map = TransformStamped()
                        arov_in_map.header.frame_id = f'{self.arov}_bag/base_link'
                        arov_in_map.header.stamp = self.get_clock().now().to_msg()
                        arov_in_map.child_frame_id = f'{self.arov}/base_link_obs_apriltag:{tag.id}'

                        arov_in_map.transform.translation.x = observation[0][0]
                        arov_in_map.transform.translation.y = observation[1][0]
                        arov_in_map.transform.translation.z = observation[2][0]
                        arov_in_map.transform.rotation.x = observation[4][0]
                        arov_in_map.transform.rotation.y = observation[5][0]
                        arov_in_map.transform.rotation.z = observation[6][0]
                        arov_in_map.transform.rotation.w = observation[3][0]

                        arov_in_map = base_link_to_tag
                        arov_in_map.header.frame_id = f'{self.arov}_bag/base_link'
                        arov_in_map.header.stamp = self.get_clock().now().to_msg()
                        arov_in_map.child_frame_id = f'{self.arov}/base_link_obs_apriltag:{tag.id}'

                        self.tf_broadcaster.sendTransform(arov_in_map)

                        try:
                            base_link_to_tag = self.tf_buffer.lookup_transform(
                                f'{self.arov}/base_link',
                                tag_frame,
                                rclpy.time.Time())

                        except TransformException as ex:
                            continue

                        base_link_to_tag.header.frame_id = f'{self.arov}_bag/base_link'
                        base_link_to_tag.child_frame_id = f'{tag_frame}_bag'

                        base_link_to_tag.header.stamp = self.get_clock().now().to_msg()

                        self.tf_broadcaster.sendTransform(base_link_to_tag)

    def predict(self):
        '''
        Propogate forward the position estimate and convariance.
        '''
        if self.state is None or self.odom is None:
            odom_to_base_link = None

            try:
                odom_to_base_link = self.tf_buffer.lookup_transform(
                    f'{self.arov}/odom',
                    f'{self.arov}/base_link',
                    rclpy.time.Time())

            except TransformException as ex:
                return

            if odom_to_base_link is not None:
                self.state = np.array([odom_to_base_link.transform.translation.x,
                                       odom_to_base_link.transform.translation.y,
                                       odom_to_base_link.transform.translation.z,
                                       odom_to_base_link.transform.rotation.w,
                                       odom_to_base_link.transform.rotation.x,
                                       odom_to_base_link.transform.rotation.y,
                                       odom_to_base_link.transform.rotation.z])

                self.time_km1 = self.get_clock().now()
                self.odom_to_base_link_km1 = odom_to_base_link
            return

        base_link_to_map = None
        odom_to_base_link = None

        try:
            base_link_to_map = self.tf_buffer.lookup_transform(
                self.base_link_frame,
                'map',
                rclpy.time.Time())

        except TransformException as ex:
            return

        try:
            odom_to_base_link = self.tf_buffer.lookup_transform(
                self.odom_frame,
                self.base_link_frame,
                rclpy.time.Time())

        except TransformException as ex:
            return

        if base_link_to_map is not None and odom_to_base_link is not None:
            time_k = self.get_clock().now()
            dt = (time_k - self.time_km1).nanoseconds / 10.0 ** 9  # Time since last prediction
            self.time_km1 = time_k

            # Linear prediction
            odom_to_map = Rotation.from_quat([base_link_to_map.transform.rotation.x,
                                              base_link_to_map.transform.rotation.y,
                                              base_link_to_map.transform.rotation.z,
                                              base_link_to_map.transform.rotation.w]) * \
                          Rotation.from_quat([odom_to_base_link.transform.rotation.x,
                                              odom_to_base_link.transform.rotation.y,
                                              odom_to_base_link.transform.rotation.z,
                                              odom_to_base_link.transform.rotation.w])

            linear_diff = np.array([odom_to_base_link.transform.translation.x,
                                    odom_to_base_link.transform.translation.y,
                                    odom_to_base_link.transform.translation.z]) - \
                          np.array([self.odom_to_base_link_km1.transform.translation.x,
                                    self.odom_to_base_link_km1.transform.translation.y,
                                    self.odom_to_base_link_km1.transform.translation.z])

            # Angular prediction
            angular_diff = Rotation.from_quat([odom_to_base_link.transform.rotation.x,
                                               odom_to_base_link.transform.rotation.y,
                                               odom_to_base_link.transform.rotation.z,
                                               odom_to_base_link.transform.rotation.w]) * \
                           Rotation.from_quat([self.odom_to_base_link_km1.transform.rotation.x,
                                               self.odom_to_base_link_km1.transform.rotation.y,
                                               self.odom_to_base_link_km1.transform.rotation.z,
                                               self.odom_to_base_link_km1.transform.rotation.w]).inv()

            angular_prediction = (angular_diff * Rotation.from_quat([*self.state[4:], self.state[3]])).as_quat()

            self.state[:3] += odom_to_map.inv().apply(linear_diff)
            self.state[3:] = [angular_prediction[3], *angular_prediction[:3]]

            # Normalize quaternion
            self.state[3:] = self.state[3:] / np.linalg.norm(self.state[3:])

            self.odom_to_base_link_km1 = odom_to_base_link

            F = np.diag([1, 1, 1, 1, 1, 1, 1])

            linear_noise = dt * np.power(np.diag(self.get_parameter('~predict_noise').value), 2)

            W_k = (dt / 2) * np.array([[-self.state[4], -self.state[5], -self.state[6]],
                                       [self.state[3], -self.state[6], self.state[5]],
                                       [self.state[6], self.state[3], -self.state[4]],
                                       [-self.state[5], self.state[4], self.state[3]]])
            angular_noise = W_k @ np.power(np.diag(self.get_parameter('~gyro_noise').value), 2) @ W_k.transpose()

            predict_noise = np.zeros((7, 7))
            predict_noise[:3, :3] = linear_noise
            predict_noise[3:, 3:] = angular_noise

            self.cov = F @ self.cov @ F.transpose() + (predict_noise)

    def correct(self, observation: np.ndarray, observation_noise: np.ndarray, h_expected: np.ndarray,
                H_jacobian: np.ndarray):
        '''
        Generic correction for a state to run whenever an observation comes in.

        Args:
            observation (ndarray): The observation being used to correct the state estimate.
            observation_noise (ndarray): The noise associated with the observation at the current timestep.
            h_expected (ndarray): The expected observation based on the current state.
            h_jacobian (ndarray): The jacobian of the observation function.
        '''
        if self.state is None: return

        err = observation - h_expected
        err_cov = (H_jacobian @ self.cov @ H_jacobian.transpose()) + observation_noise
        K_gain = self.cov @ H_jacobian.transpose() @ np.linalg.inv(err_cov)

        self.state += (K_gain @ err).transpose()[0]
        self.cov = (np.eye(np.shape(self.cov)[0]) - K_gain @ H_jacobian) @ self.cov

        # Normalize quaternion
        self.state[3:] = self.state[3:] / np.linalg.norm(self.state[3:])

    def bag_testing(self, odom_to_base_link: TransformStamped):
        bag_odom = self.transform
        bag_odom.child_frame_id = f'{self.arov}_bag/odom'

        odom_to_base_link.header.frame_id = f'{self.arov}_bag/odom'
        odom_to_base_link.child_frame_id = f'{self.arov}_bag/base_link'

        odom_to_base_link.header.stamp = self.get_clock().now().to_msg()

        self.tf_broadcaster.sendTransform(bag_odom)
        self.tf_broadcaster.sendTransform(odom_to_base_link)

        estimate = TransformStamped()
        estimate.header.stamp = self.get_clock().now().to_msg()

        estimate.header.frame_id = 'map'
        estimate.child_frame_id = f'{self.arov}_est/base_link'

        estimate.transform.translation.x = self.state[0]
        estimate.transform.translation.y = self.state[1]
        estimate.transform.translation.z = self.state[2]

        estimate.transform.rotation.w = self.state[3]
        estimate.transform.rotation.x = self.state[4]
        estimate.transform.rotation.y = self.state[5]
        estimate.transform.rotation.z = self.state[6]

        # self.tf_broadcaster.sendTransform(estimate)

    def lowpass_filter(self, id, transform):
        translation = np.array([transform.transform.translation.x,
                                transform.transform.translation.y,
                                transform.transform.translation.z], dtype=np.float64)
        orientation = np.array([transform.transform.rotation.x,
                                transform.transform.rotation.y,
                                transform.transform.rotation.z,
                                transform.transform.rotation.w], dtype=np.float64)

        if id not in self.tag_data:  # create dictionary entry and fill it
            self.tag_data[id] = {'pose': deque([translation.copy()], maxlen=2), # populate with current translation once, append second on next go
                                 'quat': deque([orientation.copy()], maxlen=2), # populate with current orientation once, append second on next go
                                 'flt_pose': deque([translation.copy()], maxlen=2), # same, append with filtered data on next go
                                 'flt_quat': deque([orientation.copy()], maxlen=2)} # same, append with filtered data on next go
            return transform

        flt_data = self.tag_data[id]
        flt_data['pose'].append(translation)
        if np.dot(orientation, flt_data['quat'][0]) < 0.0:  # test for quaternion flip
            orientation *= -1
        flt_data['quat'].append(orientation)

        # Difference eq LPF for pose
        flt_pose = self.pose_coeffs[0][0] * flt_data['pose'][1] + self.pose_coeffs[0][1] * flt_data['pose'][0] - self.pose_coeffs[1][1] * flt_data['flt_pose'][0]
        flt_data['flt_pose'].append(flt_pose)

        # Adaptive LPF for quaternion
        quat_alpha = self.base_quat_alpha * (1 + self.quat_sens * 2 * np.arccos(np.clip(np.abs(np.dot(flt_data['quat'][1], flt_data['quat'][0])), -1, 1)))
        quat_alpha = np.clip(quat_alpha, 0.05, 0.5)
        flt_quat = quat_alpha * flt_data['quat'][1] + (1-quat_alpha) * flt_data['flt_quat'][0]
        flt_quat /= np.linalg.norm(flt_quat)
        flt_data['flt_quat'].append(flt_quat)

        filtered_transform = TransformStamped()
        filtered_transform.header = transform.header
        filtered_transform.transform.translation.x = flt_pose[0]
        filtered_transform.transform.translation.y = flt_pose[1]
        filtered_transform.transform.translation.z = flt_pose[2]
        filtered_transform.transform.rotation.x = flt_quat[0]
        filtered_transform.transform.rotation.y = flt_quat[1]
        filtered_transform.transform.rotation.z = flt_quat[2]
        filtered_transform.transform.rotation.w = flt_quat[3]

        return filtered_transform

def main():
    rclpy.init()

    arov_ekf_global = AROV_EKF_Global()

    rclpy.spin(arov_ekf_global)

    arov_ekf_global.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()