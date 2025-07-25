import rclpy
from rclpy.node import Node
import numpy as np
import math
from collections import deque

from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from apriltag_msgs.msg import AprilTagDetectionArray
from sensor_msgs.msg import Imu

import rclpy.time
from tf2_ros import TransformBroadcaster, TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from scipy.spatial.transform import Rotation


class AROV_EKF_Onboard(Node):
    '''
    Node to run global position estimate for the AROV using fixed, known AprilTag locations.
    '''

    def __init__(self):
        super().__init__('arov_ekf_onboard')
        self.declare_parameters(namespace='', parameters=[
            ('~ros_bag', True),                                                         # Toggle for using bagged data and switching to sending test transforms
            ('~use_gyro', False),                                                       # Whether to use an external gyro for predict or the PixHawk estimate
            ('~initial_cov', [5.0, 5.0, 5.0, 2.0, 2.0, 2.0, 2.0]),
            ('~predict_noise', [0.75, 0.75, 0.75, 0.025, 0.025, 0.025, 0.025]),         # Diagonals of the covariance matrix for the linear portion of prediction noise
            ('~gyro_noise', [0.25, 0.25, 0.25]),
            ('~apriltag_noise', [0.025, 0.025, 0.025, 0.0125, 0.0125, 0.0125, 0.0125]),
            ('~camera_namespaces', ['/arov']),
            ('~arov_tag_ids', [7, 8, 9])
        ])

        self.ros_bag = self.get_parameter('~ros_bag').value

        self.arov_pose_sub = self.create_subscription(
            Odometry,
            f'{self.get_namespace()}/odom',
            self.arov_pose_callback,
            10)

        self.apriltag_subs = []
        for namespace in self.get_parameter('~camera_namespaces').value :
            self.apriltag_subs.append(
                self.create_subscription(
                    AprilTagDetectionArray,
                    f'{namespace}/detections',
                    self.apriltag_detect_callback,
                    10
                )
            )

        if self.get_parameter('~use_gyro').value :
            self.gyro_sub = self.create_subscription(
                                Imu,
                                f'{self.get_namespace()}/bno055/gyro',
                                self.gyro_callback,
                                10
                            )
            
            self.gyro = Imu()

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

        self.observations = deque([], maxlen=5)

        self.predict_timer = self.create_timer(1.0 / 50.0, self.predict)

        self.tag_data = {}  # id / translation / orientation / filtered translation / filtered orientation
        self.base_trns_alpha = 0.245
        self.trns_sens = 0.75
        self.base_quat_alpha = 0.245
        self.quat_sens = 0.25

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

    def arov_pose_callback(self, msg: Odometry) :
        self.odom = msg

    def gyro_callback(self, msg: Imu) :
        self.gyro = msg

    def apriltag_detect_callback(self, msg: AprilTagDetectionArray) :
        '''
        Runs full state correction using AprilTag detections whenever one or more AprilTags are observed.
        '''
        if self.state is None : return

        if msg.detections:
            namespace = msg.header.frame_id[:-7]

            if namespace in self.get_parameter('~camera_namespaces').value :
                for tag in msg.detections:
                    if tag.decision_margin < 10.0 : continue

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

                        observed_orientation = Rotation.from_quat([
                            map_to_tag.transform.rotation.x,
                            map_to_tag.transform.rotation.y,
                            map_to_tag.transform.rotation.z,
                            map_to_tag.transform.rotation.w
                            ]) * Rotation.from_quat([
                            base_link_to_tag.transform.rotation.x,
                            base_link_to_tag.transform.rotation.y,
                            base_link_to_tag.transform.rotation.z,
                            base_link_to_tag.transform.rotation.w
                            ]).inv()
                        
                        tag_in_map = observed_orientation.apply(np.array([
                            base_link_to_tag.transform.translation.x,
                            base_link_to_tag.transform.translation.y,
                            base_link_to_tag.transform.translation.z
                            ]))
                        
                        observed_orientation = observed_orientation.as_quat()

                        observation = np.array([
                            [map_to_tag.transform.translation.x - tag_in_map[0]],
                            [map_to_tag.transform.translation.y - tag_in_map[1]],
                            [map_to_tag.transform.translation.z - tag_in_map[2]],
                            [observed_orientation[3]],
                            [observed_orientation[0]],
                            [observed_orientation[1]],
                            [observed_orientation[2]]
                            ])
                                            
                        H_jacobian = np.array([
                            [1, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0, 1]
                            ])
                        
                        h_expected = np.array([
                            [self.state[0]],
                            [self.state[1]],
                            [self.state[2]],
                            [self.state[3]],
                            [self.state[4]],
                            [self.state[5]],
                            [self.state[6]]
                        ])

                        # tag_rot = Rotation.from_quat([
                        #     map_to_tag.transform.rotation.x,
                        #     map_to_tag.transform.rotation.y,
                        #     map_to_tag.transform.rotation.z,
                        #     map_to_tag.transform.rotation.w
                        # ])

                        # tag_translation = tag_rot.apply(
                        #     np.array([
                        #         base_link_to_tag.transform.translation.x,
                        #         base_link_to_tag.transform.translation.y,
                        #         base_link_to_tag.transform.translation.z
                        #     ])
                        # )

                        # base_link_rot = (tag_rot * Rotation.from_quat([
                        #     base_link_to_tag.transform.rotation.x,
                        #     base_link_to_tag.transform.rotation.y,
                        #     base_link_to_tag.transform.rotation.z,
                        #     base_link_to_tag.transform.rotation.w
                        # ])).as_quat()
                        
                        # observation = np.array([
                        #     [map_to_tag.transform.translation.x + tag_translation[0]],
                        #     [map_to_tag.transform.translation.y + tag_translation[1]],
                        #     [map_to_tag.transform.translation.z + tag_translation[2]],
                        #     [base_link_rot[3]],
                        #     [base_link_rot[0]],
                        #     [base_link_rot[1]],
                        #     [base_link_rot[2]]
                        # ])

                        # h_expected = np.array([
                        #     [self.state[0]],
                        #     [self.state[1]],
                        #     [self.state[2]],
                        #     [self.state[3]],
                        #     [self.state[4]],
                        #     [self.state[5]],
                        #     [self.state[6]]
                        # ])

                        # H_jacobian = np.array([
                        #     [1, 0, 0, 0, 0, 0, 0],
                        #     [0, 1, 0, 0, 0, 0, 0],
                        #     [0, 0, 1, 0, 0, 0, 0],
                        #     [0, 0, 0, 1, 0, 0, 0],
                        #     [0, 0, 0, 0, 1, 0, 0],
                        #     [0, 0, 0, 0, 0, 1, 0],
                        #     [0, 0, 0, 0, 0, 0, 1]
                        # ])

                        observation_noise = self.apriltag_noise * np.linalg.norm(np.array([
                            base_link_to_tag.transform.translation.x,
                            base_link_to_tag.transform.translation.y,
                            base_link_to_tag.transform.translation.z
                        ])) ** 2

                        self.correct(observation, observation_noise, h_expected, H_jacobian)

                        self.observations.append(observation.transpose()[0])

                        # self.state = np.mean(self.observations, axis=0)

                        # Normalize quaternion
                        self.state[3:] = self.state[3:] / np.linalg.norm(self.state[3:])

                        observed_arov = TransformStamped()
                        observed_arov.child_frame_id = f'arov_observed_{tag_frame}'
                        observed_arov.header.frame_id = 'map'
                        observed_arov.header.stamp = self.get_clock().now().to_msg()

                        observed_arov.transform.translation.x = observation[0][0]
                        observed_arov.transform.translation.y = observation[1][0]
                        observed_arov.transform.translation.z = observation[2][0]

                        observed_arov.transform.rotation.w = observation[3][0]
                        observed_arov.transform.rotation.x = observation[4][0]
                        observed_arov.transform.rotation.y = observation[5][0]
                        observed_arov.transform.rotation.z = observation[6][0]

                        self.tf_broadcaster.sendTransform(observed_arov)

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
        
        if self.state is None:
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

                self.publish_transform()
            return

        if odom_to_base_link is not None:
            time_k = self.get_clock().now()
            dt = (time_k - self.time_km1).nanoseconds / 10.0 ** 9  # Time since last prediction
            self.time_km1 = time_k

            xr = self.state[0]
            yr = self.state[1]
            zr = self.state[2]
            wqr = self.state[3]
            xqr = self.state[4]
            yqr = self.state[5]
            zqr = self.state[6]

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

            if not self.get_parameter('~use_gyro').value :
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

                self.state = np.array([
                    xr + dx*(wqr**2 + xqr**2 - yqr**2 - zqr**2) + 2*(wqr*-yqr*dz + wqr*zqr*dy + yqr*dy*xqr + zqr*dz*xqr),
                    yr + dy*(wqr**2 - xqr**2 + yqr**2 - zqr**2) + 2*(wqr*-zqr*dx + wqr*xqr*dz + xqr*dx*yqr + zqr*dz*yqr),
                    zr + dz*(wqr**2 - xqr**2 - yqr**2 + zqr**2) + 2*(wqr*-xqr*dy + wqr*yqr*dx + xqr*dx*zqr + yqr*dy*zqr),
                    wqr * dwq - xqr * dxq - yqr * dyq - zqr * dzq,
                    wqr * dxq + xqr * dwq - yqr * dzq + zqr * dyq,
                    wqr * dyq + xqr * dzq + yqr * dwq - zqr * dxq,
                    wqr * dzq - xqr * dyq + yqr * dxq + zqr * dwq
                ])


                F = np.array([
                    [1, 0, 0, 2*(dx*wqr - yqr*dz + zqr*dy), 2*(dx*xqr + yqr*dy + zqr*dz), 2*(dx*-yqr - wqr*dz + dy*xqr), 2*(dx*-zqr + wqr*dy + dz*xqr)],
                    [0, 1, 0, 2*(dy*wqr - zqr*dx + xqr*dz), 2*(dy*-xqr + wqr*dz + dx*yqr), 2*(dy*yqr + xqr*dx + zqr*dz), 2*(dy*-zqr - wqr*dx + dz*yqr)],
                    [0, 0, 1, 2*(dz*wqr - xqr*dy + yqr*dx), 2*(dz*-xqr - wqr*dy + dx*zqr), 2*(dz*-yqr + wqr*dx + dy*zqr), 2*(dz*zqr + xqr*dx + yqr*dy)],
                    [0, 0, 0, dwq, -dxq, -dyq, -dzq],
                    [0, 0, 0, dxq, dwq, -dzq, dyq],
                    [0, 0, 0, dyq, dzq, dwq, -dxq],
                    [0, 0, 0, dzq, -dyq, dxq, dwq]
                ])
            
            else :
                # Angular prediction (w 3, x 4, y 5, z 6) witih gyro
                # Gyro rates
                gxdot = self.gyro.angular_velocity.x
                gydot = self.gyro.angular_velocity.y
                gzdot = self.gyro.angular_velocity.z

                self.state = np.array([
                    xr + dx*(wqr**2 + xqr**2 - yqr**2 - zqr**2) + 2*(wqr*-yqr*dz + wqr*zqr*dy + yqr*dy*xqr + zqr*dz*xqr),
                    yr + dy*(wqr**2 - xqr**2 + yqr**2 - zqr**2) + 2*(wqr*-zqr*dx + wqr*xqr*dz + xqr*dx*yqr + zqr*dz*yqr),
                    zr + dz*(wqr**2 - xqr**2 - yqr**2 + zqr**2) + 2*(wqr*-xqr*dy + wqr*yqr*dx + xqr*dx*zqr + yqr*dy*zqr),
                    wqr - (dt/2) * gxdot * xqr - (dt/2) * gydot * yqr - (dt/2) * gzdot * zqr,
                    xqr + (dt/2) * gxdot * wqr - (dt/2) * gydot * zqr + (dt/2) * gzdot * yqr,
                    yqr + (dt/2) * gxdot * zqr + (dt/2) * gydot * wqr - (dt/2) * gzdot * xqr,
                    zqr - (dt/2) * gxdot * yqr + (dt/2) * gydot * xqr + (dt/2) * gzdot * wqr
                ])

                F = np.array([
                    [1, 0, 0, 2*(dx*wqr - yqr*dz + zqr*dy), 2*(dx*xqr + yqr*dy + zqr*dz), 2*(dx*-yqr - wqr*dz + dy*xqr), 2*(dx*-zqr + wqr*dy + dz*xqr)],
                    [0, 1, 0, 2*(dy*wqr - zqr*dx + xqr*dz), 2*(dy*-xqr + wqr*dz + dx*yqr), 2*(dy*yqr + xqr*dx + zqr*dz), 2*(dy*-zqr - wqr*dx + dz*yqr)],
                    [0, 0, 1, 2*(dz*wqr - xqr*dy + yqr*dx), 2*(dz*-xqr - wqr*dy + dx*zqr), 2*(dz*-yqr + wqr*dx + dy*zqr), 2*(dz*zqr + xqr*dx + yqr*dy)],
                    [0, 0, 0, 1, -(dt/2) * gxdot, -(dt/2) * gydot, -(dt/2) * gzdot],
                    [0, 0, 0, (dt/2) * gxdot, 1, (dt/2) * gzdot, -(dt/2) * gydot],
                    [0, 0, 0, (dt/2) * gydot, -(dt/2) * gzdot, 1, (dt/2) * gxdot],
                    [0, 0, 0, (dt/2) * gzdot, (dt/2) * gydot, -(dt/2) * gxdot, 1]
                    ])
                
                linear_noise = dt * np.power(np.diag(self.get_parameter('~predict_noise').value[:3]), 2)
                
                W_k = (dt/2) * np.array([[-xqr, -yqr, -zqr],
                                        [wqr, -zqr, yqr],
                                        [zqr, wqr, -xqr],
                                        [-yqr, xqr, wqr]])
                angular_noise = W_k @ np.power(np.diag(self.get_parameter('~gyro_noise').value), 2) @ W_k.transpose()

                predict_noise = np.zeros((7, 7))
                predict_noise[:3, :3] = linear_noise
                predict_noise[3:, 3:] = angular_noise

            self.cov = F @ self.cov @ F.transpose() + (self.predict_noise)

            # Normalize quaternion
            self.state[3:] = self.state[3:] / np.linalg.norm(self.state[3:])

            self.odom_to_base_link_km1 = odom_to_base_link

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
            self.tag_data[id] = {'trns': deque([translation.copy()], maxlen=2), # populate with current translation once, append second on next go
                                 'quat': deque([orientation.copy()], maxlen=2), # populate with current orientation once, append second on next go
                                 'flt_trns': deque([translation.copy()], maxlen=2), # same, append with filtered data on next go
                                 'flt_quat': deque([orientation.copy()], maxlen=2)} # same, append with filtered data on next go
            return transform

        flt_data = self.tag_data[id]
        flt_data['trns'].append(translation)
        if np.dot(orientation, flt_data['quat'][0]) < 0.0:  # test for quaternion flip
            orientation *= -1
        flt_data['quat'].append(orientation)

        # Adaptive LPF for translation
        trns_alpha = self.base_trns_alpha * (1 + self.trns_sens * np.linalg.norm(flt_data['trns'][1] - flt_data['trns'][0]))
        trns_alpha = np.clip(trns_alpha, 0.05, 0.5)
        flt_trns = trns_alpha * flt_data['trns'][1] + (1 - trns_alpha) * flt_data['flt_trns'][0]
        flt_data['flt_trns'].append(flt_trns)

        # Adaptive LPF for quaternion
        quat_alpha = self.base_quat_alpha * (1 + self.quat_sens * 2 * np.arccos(np.clip(np.abs(np.dot(flt_data['quat'][1], flt_data['quat'][0])), -1, 1)))
        quat_alpha = np.clip(quat_alpha, 0.05, 0.5)
        flt_quat = quat_alpha * flt_data['quat'][1] + (1-quat_alpha) * flt_data['flt_quat'][0]
        flt_quat /= np.linalg.norm(flt_quat)
        flt_data['flt_quat'].append(flt_quat)

        filtered_transform = TransformStamped()
        filtered_transform.header = transform.header
        filtered_transform.transform.translation.x = flt_trns[0]
        filtered_transform.transform.translation.y = flt_trns[1]
        filtered_transform.transform.translation.z = flt_trns[2]
        # filtered_transform.transform.translation = transform.transform.translation
        filtered_transform.transform.rotation.x = flt_quat[0]
        filtered_transform.transform.rotation.y = flt_quat[1]
        filtered_transform.transform.rotation.z = flt_quat[2]
        filtered_transform.transform.rotation.w = flt_quat[3]

        return filtered_transform

def main():
    rclpy.init()

    arov_ekf_onboard = AROV_EKF_Onboard()

    rclpy.spin(arov_ekf_onboard)

    arov_ekf_onboard.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
