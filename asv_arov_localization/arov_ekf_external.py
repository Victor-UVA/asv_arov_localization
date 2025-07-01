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


class AROV_EKF_External(Node):
    '''
    Node to run global position estimate for the AROV using fixed, known AprilTag locations.
    '''

    def __init__(self):
        super().__init__('arov_ekf_external')
        self.declare_parameters(namespace='', parameters=[
            ('~ros_bag', True),                                                         # Toggle for using bagged data and switching to sending test transforms
            ('~initial_cov', [5.0, 5.0, 5.0, 2.0, 2.0, 2.0, 2.0]),
            ('~predict_noise', [0.5, 0.5, 0.5, 0.025, 0.025, 0.025, 0.025]),            # Diagonals of the covariance matrix for the linear portion of prediction noise
            ('~apriltag_noise', [0.01, 0.01, 0.01, 0.025, 0.025, 0.025, 0.025]),
            ('~camera_namespace', 'cam1')
        ])

        self.ros_bag = self.get_parameter('~ros_bag').value

        self.arov_pose_sub = self.create_subscription(
            Odometry,
            f'{self.get_namespace()}/odom',
            self.arov_pose_callback,
            10)
        self.arov_pose_sub  # prevent unused variable warning

        self.arov_apriltag_detect_sub = self.create_subscription(
            AprilTagDetectionArray,
            f'{self.get_parameter('~camera_namespace').value}/detections',
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
        self.transform.child_frame_id = f'{self.get_namespace().strip('/')}/odom'

        self.time_km1 = None
        self.state = None  # State estimate: x, y, z, q(w, x, y, z)
        self.cov = np.diag(self.get_parameter('~initial_cov').value)  # Covariance estimate

        self.predict_noise = np.power(np.diag(self.get_parameter('~predict_noise').value), 2)
        self.apriltag_noise = np.power(np.diag(self.get_parameter('~apriltag_noise').value), 2)

        self.predict_timer = self.create_timer(1.0 / 50.0, self.predict)

        self.tag_data = {}  # id / translation / orientation / filtered translation / filtered orientation
        pose_f0 = 0.1
        Fs = 25
        pose_alpha = 1 - math.exp(-2 * math.pi * pose_f0/ Fs)
        self.pose_coeffs = [[1.2*pose_alpha, -0.2*pose_alpha], [1.0, pose_alpha-1]]
        self.base_quat_alpha = 0.1
        self.quat_sens = 0.5

    def publish_transform(self) :
        if self.state is None : return

        odom_to_base_link = None

        try :
            odom_to_base_link = self.tf_buffer.lookup_transform(
                f'{self.get_namespace().strip('/')}/odom',
                f'{self.get_namespace().strip('/')}/base_link',
                rclpy.time.Time())

        except TransformException as ex :
            return

        if odom_to_base_link is not None :
            orientation = Rotation.from_quat([*self.state[4:], self.state[3]]) * Rotation.from_quat(
                [odom_to_base_link.transform.rotation.x,
                 odom_to_base_link.transform.rotation.y,
                 odom_to_base_link.transform.rotation.z,
                 odom_to_base_link.transform.rotation.w]).inv()

            translation = self.state[:3] - orientation.apply(np.array([odom_to_base_link.transform.translation.x,
                                                                       odom_to_base_link.transform.translation.y,
                                                                       odom_to_base_link.transform.translation.z]))

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
                        f'{self.get_namespace().strip('/')}/base_link',
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

                    observation_noise = self.apriltag_noise

                    # self.correct(observation, observation_noise, h_expected, H_jacobian)

    def predict(self):
        '''
        Propogate forward the position estimate and convariance.
        '''
        odom_to_base_link = None

        try:
            odom_to_base_link = self.tf_buffer.lookup_transform(
                f'{self.get_namespace().strip('/')}/odom',
                f'{self.get_namespace().strip('/')}/base_link',
                rclpy.time.Time())

        except TransformException as ex:
            return
        
        if self.state is None or self.odom is None:
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

        if odom_to_base_link is not None:
            time_k = self.get_clock().now()
            dt = (time_k - self.time_km1).nanoseconds / 10.0 ** 9  # Time since last prediction
            self.time_km1 = time_k

            # Linear prediction
            linear_diff = Rotation.from_quat([
                self.odom_to_base_link_km1.transform.rotation.x,
                self.odom_to_base_link_km1.transform.rotation.y,
                self.odom_to_base_link_km1.transform.rotation.z,
                self.odom_to_base_link_km1.transform.rotation.w
            ]).apply(
                np.array([
                    odom_to_base_link.transform.translation.x - self.odom_to_base_link_km1.transform.translation.x,
                    odom_to_base_link.transform.translation.y - self.odom_to_base_link_km1.transform.translation.y,
                    odom_to_base_link.transform.translation.z - self.odom_to_base_link_km1.transform.translation.z
                ])
            )
            
            dx = linear_diff[0]
            dy = linear_diff[1]
            dz = linear_diff[2]

            # Angular prediction
            angular_diff = (Rotation.from_quat([odom_to_base_link.transform.rotation.x,
                                                odom_to_base_link.transform.rotation.y,
                                                odom_to_base_link.transform.rotation.z,
                                                odom_to_base_link.transform.rotation.w]) * \
                            Rotation.from_quat([self.odom_to_base_link_km1.transform.rotation.x,
                                                self.odom_to_base_link_km1.transform.rotation.y,
                                                self.odom_to_base_link_km1.transform.rotation.z,
                                                self.odom_to_base_link_km1.transform.rotation.w]).inv()).as_quat()

            dwq = angular_diff[3]
            dxq = angular_diff[0]
            dyq = angular_diff[1]
            dzq = angular_diff[2]

            xr = self.state[0]
            yr = self.state[1]
            zr = self.state[2]
            wqr = self.state[3]
            xqr = self.state[4]
            yqr = self.state[5]
            zqr = self.state[6]

            self.state = np.array([
                xr + dx*(wqr**2 + xqr**2 - yqr**2 - zqr**2) + 2*(wqr*-yqr*dz + wqr*zqr*dy + yqr*dy*xqr + zqr*dz*xqr),
                yr + dy*(wqr**2 - xqr**2 + yqr**2 - zqr**2) + 2*(wqr*-zqr*dx + wqr*xqr*dz + xqr*dx*yqr + zqr*dz*yqr),
                zr + dz*(wqr**2 - xqr**2 - yqr**2 + zqr**2) + 2*(wqr*-xqr*dy + wqr*yqr*dx + xqr*dx*zqr + yqr*dy*zqr),
                wqr * dwq - xqr * dxq - yqr * dyq - zqr * dzq,
                wqr * dxq + xqr * dwq - yqr * dzq + zqr * dyq,
                wqr * dyq + xqr * dzq + yqr * dwq - zqr * dxq,
                wqr * dzq - xqr * dyq + yqr * dxq + zqr * dwq
            ])

            # Normalize quaternion
            self.state[3:] = self.state[3:] / np.linalg.norm(self.state[3:])

            self.odom_to_base_link_km1 = odom_to_base_link

            F = np.array([
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, dwq, -dxq, -dyq, -dzq],
                [0, 0, 0, dxq, dwq, -dzq, dyq],
                [0, 0, 0, dyq, dzq, dwq, -dxq],
                [0, 0, 0, dzq, -dyq, dxq, dwq]
            ])

            self.cov = F @ self.cov @ F.transpose() + (self.predict_noise)
            
            self.publish_transform()

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
        bag_odom.child_frame_id = f'{self.get_namespace().strip('/')}_bag/odom'

        odom_to_base_link.header.frame_id = f'{self.get_namespace().strip('/')}_bag/odom'
        odom_to_base_link.child_frame_id = f'{self.get_namespace().strip('/')}_bag/base_link'

        odom_to_base_link.header.stamp = self.get_clock().now().to_msg()

        self.tf_broadcaster.sendTransform(bag_odom)
        self.tf_broadcaster.sendTransform(odom_to_base_link)

        estimate = TransformStamped()
        estimate.header.stamp = self.get_clock().now().to_msg()

        estimate.header.frame_id = 'map'
        estimate.child_frame_id = f'{self.get_namespace().strip('/')}_est/base_link'

        estimate.transform.translation.x = self.state[0]
        estimate.transform.translation.y = self.state[1]
        estimate.transform.translation.z = self.state[2]

        estimate.transform.rotation.w = self.state[3]
        estimate.transform.rotation.x = self.state[4]
        estimate.transform.rotation.y = self.state[5]
        estimate.transform.rotation.z = self.state[6]

        # self.tf_broadcaster.sendTransform(estimate)

    def lowpass_filter(self, id, transform: TransformStamped):
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

    arov_ekf_external = AROV_EKF_External()

    rclpy.spin(arov_ekf_external)

    arov_ekf_external.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()