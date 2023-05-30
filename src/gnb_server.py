#!/usr/bin/env python

import rospy
import ros_numpy

from utils.grasp import Grasp, GraspGroup

from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose, Vector3
from std_msgs.msg import ColorRGBA

import time
import torch
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from models.graspnet import GraspNet, pred_decode
from utils.collision_detector import ModelFreeCollisionDetector


class GNBServer:
    def __init__(self):
        self.get_ros_param()
        self.net = GraspNet(
            input_feature_dim=0,
            num_view=300,
            num_angle=12,
            num_depth=4,
            cylinder_radius=0.05,
            hmin=-0.02,
            hmax_list=[0.01, 0.02, 0.03, 0.04],
            is_training=False,
        )
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        checkpoint = torch.load(self.model_path)
        self.net.load_state_dict(checkpoint["model_state_dict"])
        self.net.eval()

        rospy.Subscriber(self.pointcloud_topic, PointCloud2, self.predict_grasps)

        self.grasp_pub = rospy.Publisher(self.grasp_topic, Pose, queue_size=1)

        self.marker_pub = rospy.Publisher(self.marker_topic, MarkerArray, queue_size=1)
        self.draw_grasp_id = 0

    def get_ros_param(self):
        self.model_path = rospy.get_param("~model_path", "../models/checkpoint-rs.tar")
        self.max_point_num = rospy.get_param("~max_point_num", 20000)
        self.collision_thresh = rospy.get_param("~collision_thresh", 0.01)
        self.voxel_size = rospy.get_param("~voxel_size", 0.01)
        self.max_grasps_num = rospy.get_param("~max_grasps_num", 20)

        self.pointcloud_topic = rospy.get_param("~pointcloud_topic", "pointcloud")
        self.grasp_topic = rospy.get_param("~grasp_topic", "grasps")
        self.marker_topic = rospy.get_param("~marker_topic", "marker")

    def get_points(
        self, data
    ) -> tuple[dict[str, torch.Tensor], o3d.utility.Vector3dVector]:
        pc = ros_numpy.numpify(data)
        pc_np = np.concatenate(
            (pc["x"].reshape(-1, 1), pc["y"].reshape(-1, 1), pc["z"].reshape(-1, 1)),
            axis=1,
        ).astype(
            np.float32
        )  # (N, 3)
        if len(pc_np) > self.max_point_num:
            rospy.logdebug(f"点云数量{len(pc_np)}，随机采样{self.max_point_num}个点")
            idxs = np.random.choice(len(pc_np), self.max_point_num, replace=False)
        else:
            rospy.logdebug(f"点云数量{len(pc_np)}")
            idxs = np.arange(len(pc_np))
        pc_np_sampled = pc_np[idxs]
        points_torch = torch.from_numpy(pc_np_sampled[np.newaxis]).to(
            self.device
        )  # (1, N, 3)
        end_points = {"point_clouds": points_torch}
        points_o3d = o3d.utility.Vector3dVector(pc_np)
        return end_points, points_o3d

    def get_grasps(self, end_points):
        with torch.no_grad():
            end_points = self.net(end_points)
            grasp_preds = pred_decode(end_points)
        gg_array = grasp_preds[0].detach().cpu().numpy()
        gg = GraspGroup(gg_array)
        return gg

    def collision_detection(self, gg, cloud) -> GraspGroup:
        mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=self.voxel_size)
        collision_mask = mfcdetector.detect(
            gg, approach_dist=0.05, collision_thresh=self.collision_thresh
        )
        gg = gg[~collision_mask]
        rospy.loginfo(f"碰撞检测后剩余{len(gg)}个")
        return gg

    def get_top_n(self, gg: GraspGroup, n):
        gg = gg.nms()
        rospy.loginfo(f"非极大值抑制后剩余{len(gg)}个")
        gg.sort_by_score()
        if len(gg) > n:
            rospy.loginfo(f"发布前{n}个")
        rospy.loginfo(f"最高分{gg[0].score:.3f}")
        return gg[:n]

    def to_pose_msg(self, gg):
        # 将 https://github.com/graspnet/graspnetAPI 中定义的夹爪轴方向转换为沿x轴开合、沿z轴接近
        rot_mat_z = Rotation.from_euler(
            "z", 90, degrees=True
        ).as_matrix()  # [[0,-1,0],[1,0,0],[0,0,1]]
        rot_mat_y = Rotation.from_euler(
            "y", 90, degrees=True
        ).as_matrix()  # [[0,0,1],[0,1,0],[-1,0,0]]
        rot_mat = rot_mat_y @ rot_mat_z  # [[0,0,1],[1,0,0],[0,1,0]]
        msgs = []
        q = []
        for i in range(len(gg)):
            grasp: Grasp = gg[i]
            q.append(grasp.score)
            pose_mat = np.eye(4)
            pose_mat[:3, 3] = grasp.translation
            pose_mat[:3, :3] = rot_mat @ grasp.rotation_matrix
            pose_mat = forward_pose(pose_mat, grasp.depth)
            # https://github.com/graspnet/graspnet-baseline/issues/23#issuecomment-893119187
            pose = matrix_to_pose_msg(pose_mat)
            msgs.append(pose)
        return msgs, q

    def vis_grasps(self, poses, f, qualities):
        clear_markers(self.marker_pub)
        cm = lambda s: tuple([float(1 - s), float(s), float(0), float(1)])
        vmin = 0
        vmax = 1
        self.draw_grasp_id = 0
        for pose, q in zip(poses, qualities):
            color = cm((q - vmin) / (vmax - vmin))
            markers = create_grasp_markers(f, pose, color, "grasp", self.draw_grasp_id)
            self.draw_grasp_id += 4
            self.marker_pub.publish(MarkerArray(markers=markers))
            time.sleep(0.05)

    def pub_grasp(self, poses):
        for pose in poses:
            self.grasp_pub.publish(pose)

    def predict_grasps(self, data):
        end_points, points_o3d = self.get_points(data)
        gg = self.get_grasps(end_points)
        if self.collision_thresh > 0:
            gg = self.collision_detection(gg, np.array(points_o3d))
        gg = self.get_top_n(gg, self.max_grasps_num)
        grasps, q = self.to_pose_msg(gg)
        self.vis_grasps(grasps, data.header.frame_id, q)
        self.pub_grasp(grasps)


def clear_markers(pub):
    delete = [Marker(action=Marker.DELETEALL)]
    pub.publish(MarkerArray(delete))


def create_grasp_markers(frame, pose: Pose, color, ns, id=0):
    # 抓取点位于指尖
    pose_mat = pose_msg_to_matrix(pose)
    w, d, radius = 0.075, 0.05, 0.005

    left_point = np.dot(pose_mat, np.array([-w / 2, 0, -d / 2, 1]))
    left_pose = pose_mat.copy()
    left_pose[:3, 3] = left_point[:3]
    scale = [radius, radius, d]
    left = create_marker(Marker.CYLINDER, frame, left_pose, scale, color, ns, id)

    right_point = np.dot(pose_mat, np.array([w / 2, 0, -d / 2, 1]))
    right_pose = pose_mat.copy()
    right_pose[:3, 3] = right_point[:3]
    scale = [radius, radius, d]
    right = create_marker(Marker.CYLINDER, frame, right_pose, scale, color, ns, id + 1)

    wrist_point = np.dot(pose_mat, np.array([0.0, 0.0, -d * 5 / 4, 1]))
    wrist_pose = pose_mat.copy()
    wrist_pose[:3, 3] = wrist_point[:3]
    scale = [radius, radius, d / 2]
    wrist = create_marker(Marker.CYLINDER, frame, wrist_pose, scale, color, ns, id + 2)

    palm_point = np.dot(pose_mat, np.array([0.0, 0.0, -d, 1]))
    palm_pose = pose_mat.copy()
    palm_pose[:3, 3] = palm_point[:3]
    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_rotvec([0, np.pi / 2, 0]).as_matrix()
    palm_pose = np.dot(palm_pose, rot)
    scale = [radius, radius, w]
    palm = create_marker(Marker.CYLINDER, frame, palm_pose, scale, color, ns, id + 3)

    return [left, right, wrist, palm]


def create_marker(type, frame, pose, scale=[1, 1, 1], color=(1, 1, 1, 1), ns="", id=0):
    if np.isscalar(scale):
        scale = [scale, scale, scale]
    msg = Marker()
    msg.header.frame_id = frame
    msg.header.stamp = rospy.Time()
    msg.ns = ns
    msg.id = id
    msg.type = type
    msg.action = Marker.ADD
    msg.pose = matrix_to_pose_msg(pose)
    msg.scale = Vector3(*scale)
    msg.color = ColorRGBA(*color)
    return msg


def pose_msg_to_matrix(pose_msg: Pose):
    """将 ROS 的 Pose 消息转换为变换矩阵"""
    translation = np.array(
        [pose_msg.position.x, pose_msg.position.y, pose_msg.position.z]
    )
    rotation = np.array(
        [
            pose_msg.orientation.x,
            pose_msg.orientation.y,
            pose_msg.orientation.z,
            pose_msg.orientation.w,
        ]
    )
    # from_quat(x, y, z, w)
    rotation_matrix = Rotation.from_quat(rotation).as_matrix()
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix[:3, :3]
    transform_matrix[:3, 3] = translation
    return transform_matrix


def matrix_to_pose_msg(pose_mat):
    """将变换矩阵转换为 ROS 的 Poe 消息"""
    pose = Pose()
    # as_quat()->(x, y, z, w)
    translation = Rotation.from_matrix(pose_mat[:3, :3]).as_quat()
    pose.orientation.x = translation[0]
    pose.orientation.y = translation[1]
    pose.orientation.z = translation[2]
    pose.orientation.w = translation[3]
    pose.position.x = pose_mat[0, 3]
    pose.position.y = pose_mat[1, 3]
    pose.position.z = pose_mat[2, 3]
    return pose


def forward_pose(pose_mat, length):
    point = np.dot(pose_mat, np.array([0.0, 0.0, length, 1]))
    new_pose_mat = pose_mat.copy()
    new_pose_mat[:3, 3] = point[:3]
    return new_pose_mat


if __name__ == "__main__":
    rospy.init_node("gnb_server", log_level=rospy.DEBUG)
    p = GNBServer()
    print("gnb_server就绪")
    rospy.spin()
