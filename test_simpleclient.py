#coding:utf-8
import socket
import struct
import abc
import threading
import psutil
import shutil
from collections import namedtuple, deque
from enum import Enum
from pathlib import Path
import numpy as np
import numpy.linalg
import cv2
import os
import open3d as o3d
from utils import DEPTH_SCALING_FACTOR, project_on_depth, project_on_pv

np.warnings.filterwarnings('ignore')

# Definitions
# Protocol Header Format
# see https://docs.python.org/2/library/struct.html#format-characters
VIDEO_STREAM_HEADER_FORMAT = "@qIIII18f"

VIDEO_FRAME_STREAM_HEADER = namedtuple(
    'SensorFrameStreamHeader',
    'Timestamp ImageWidth ImageHeight PixelStride RowStride fx fy '
    'PVtoWorldtransformM11 PVtoWorldtransformM12 PVtoWorldtransformM13 PVtoWorldtransformM14 '
    'PVtoWorldtransformM21 PVtoWorldtransformM22 PVtoWorldtransformM23 PVtoWorldtransformM24 '
    'PVtoWorldtransformM31 PVtoWorldtransformM32 PVtoWorldtransformM33 PVtoWorldtransformM34 '
    'PVtoWorldtransformM41 PVtoWorldtransformM42 PVtoWorldtransformM43 PVtoWorldtransformM44 '
)

RM_STREAM_HEADER_FORMAT = "@qIIII16f"

RM_FRAME_STREAM_HEADER = namedtuple(
    'SensorFrameStreamHeader',
    'Timestamp ImageWidth ImageHeight PixelStride RowStride '
    'rig2worldTransformM11 rig2worldTransformM12 rig2worldTransformM13 rig2worldTransformM14 '
    'rig2worldTransformM21 rig2worldTransformM22 rig2worldTransformM23 rig2worldTransformM24 '
    'rig2worldTransformM31 rig2worldTransformM32 rig2worldTransformM33 rig2worldTransformM34 '
    'rig2worldTransformM41 rig2worldTransformM42 rig2worldTransformM43 rig2worldTransformM44 '
)

CALIB_HEADER_FORMAT = "@qIIII16f"
CALIB_HEADER = namedtuple(
    'CalibHeader',
    'cameraViewMatrixM11 cameraViewMatrixM12 cameraViewMatrixM13 cameraViewMatrixM14 '
    'cameraViewMatrixM21 cameraViewMatrixM22 cameraViewMatrixM23 cameraViewMatrixM24 '
    'cameraViewMatrixM31 cameraViewMatrixM32 cameraViewMatrixM33 cameraViewMatrixM34 '
    'cameraViewMatrixM41 cameraViewMatrixM42 cameraViewMatrixM43 cameraViewMatrixM44 '
)

# Each port corresponds to a single stream type
VIDEO_STREAM_PORT = 23940
AHAT_STREAM_PORT = 23941
CALIB_PORT = 23942

HOST = '192.168.110.183'

HundredsOfNsToMilliseconds = 1e-4
MillisecondsToSeconds = 1e-3

def load_lut(lut_filename):
    with open(lut_filename, mode='rb') as depth_file:
        lut = np.frombuffer(depth_file.read(), dtype="f")
        lut = np.reshape(lut, (-1, 3))       #reshape into 3 columns
    return lut

def load_extrinsics(extrinsics_path):
    assert Path(extrinsics_path).exists()
    mtx = np.loadtxt(str(extrinsics_path), delimiter=',').reshape((4, 4))
    return mtx

def get_points_in_cam_space(img, lut):
    img = np.tile(img.flatten().reshape((-1, 1)), (1, 3))
    points = img * lut
    remove_ids = np.where(np.sum(points, axis=1) < 1e-6)[0]
    points = np.delete(points, remove_ids, axis=0)
    points /= 1000.
    return points

def cam2world(points, rig2cam, rig2world):
    homog_points = np.hstack((points, np.ones((points.shape[0], 1))))
    cam2world_transform = rig2world @ np.linalg.inv(rig2cam)
    world_points = cam2world_transform @ homog_points.T
    return world_points.T[:, :3], cam2world_transform

def save_ply(output_path, points, rgb, cam2world_transform=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    pcd.estimate_normals()
    if cam2world_transform is not None:
        # Camera center
        camera_center = (cam2world_transform) @ np.array([0, 0, 0, 1])
        o3d.geometry.PointCloud.orient_normals_towards_camera_location(pcd, camera_center[:3])
    o3d.io.write_point_cloud(output_path, pcd)

class SensorType(Enum):
    VIDEO = 1
    AHAT = 2
    LONG_THROW_DEPTH = 3
    LF_VLC = 4
    RF_VLC = 5



class FrameReceiverThread(threading.Thread):
    def __init__(self, host, port, header_format, header_data):
        super(FrameReceiverThread, self).__init__()
        self.header_size = struct.calcsize(header_format)
        self.header_format = header_format
        self.header_data = header_data
        self.host = host
        self.port = port
        self.latest_frame = None
        self.latest_header = None
        self.socket = None

    def get_data_from_socket(self):
        # read the header in chunks
        reply = self.recvall(self.header_size)

        if not reply:
            print('ERROR: Failed to receive data from stream.')
            return

        data = struct.unpack(self.header_format, reply)
        header = self.header_data(*data)

        # read the image in chunks
        image_size_bytes = header.ImageHeight * header.RowStride
        image_data = self.recvall(image_size_bytes)

        return header, image_data

    def recvall(self, size):
        msg = bytes()
        while len(msg) < size:
            part = self.socket.recv(size - len(msg))
            if part == '':
                break  # the connection is closed
            msg += part
        return msg

    def start_socket(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))
        # send_message(self.socket, b'socket connected at ')
        print('INFO: Socket connected to ' + self.host + ' on port ' + str(self.port))

    def start_listen(self):
        t = threading.Thread(target=self.listen)
        t.start()

    @abc.abstractmethod
    def listen(self):
        return

    @abc.abstractmethod
    def get_mat_from_header(self, header):
        return

class VideoReceiverThread(FrameReceiverThread):
    def __init__(self, host):
        super().__init__(host, VIDEO_STREAM_PORT, VIDEO_STREAM_HEADER_FORMAT,
                         VIDEO_FRAME_STREAM_HEADER)

    def listen(self):
        while True:
            self.latest_header, image_data = self.get_data_from_socket()
            self.latest_frame = np.frombuffer(image_data, dtype=np.uint8).reshape((self.latest_header.ImageHeight,
                                                                                   self.latest_header.ImageWidth,
                                                                                   self.latest_header.PixelStride))

    def get_mat_from_header(self, header):
        pv_to_world_transform = np.array(header[7:24]).reshape((4, 4)).T
        return pv_to_world_transform

    def get_focal_length(self, header):
        focal_length = np.array(header[5:7])
        return focal_length

    def get_timestamp(self, header):
        rgb_timestamp = np.array(header[0])
        return rgb_timestamp


class AhatReceiverThread(FrameReceiverThread):
    def __init__(self, host):
        super().__init__(host,
                         AHAT_STREAM_PORT, RM_STREAM_HEADER_FORMAT, RM_FRAME_STREAM_HEADER)


    def listen(self):
        while True:
            self.latest_header, image_data = self.get_data_from_socket()
            self.latest_frame = np.frombuffer(image_data, dtype=np.uint16).reshape((self.latest_header.ImageHeight,
                                                                                    self.latest_header.ImageWidth))

    def get_mat_from_header(self, header):
        rig_to_world_transform = np.array(header[5:22]).reshape((4, 4)).T
        return rig_to_world_transform

    def get_timestamp(self, header):
        rig_timestamp = np.array(header[0])
        return rig_timestamp

class CalibReceiverThread(FrameReceiverThread):
    def __init__(self, host):
        super().__init__(host, CALIB_PORT, VIDEO_STREAM_HEADER_FORMAT, RM_FRAME_STREAM_HEADER)

    def listen(self):
        while True:
            self.latest_header= self.get_data_from_socket()



if __name__ == '__main__':
    video_receiver = VideoReceiverThread(HOST)
    video_receiver.start_socket()

    ahat_receiver = AhatReceiverThread(HOST)
    ahat_receiver.start_socket()

    calib_receiver = CalibReceiverThread(HOST)
    calib_receiver.start_socket()

    video_receiver.start_listen()
    ahat_receiver.start_listen()
    calib_receiver.start_listen()

    file = "D:\\calibration"
    if not os.path.exists(file):
        os.mkdir(file)

    folder = "D:\\pinhole_projection"
    if os.path.exists(folder):
        shutil.rmtree(folder)
    if not os.path.exists(folder):
        os.mkdir(folder)
    folder_ply = "D:\\pinhole_projection\\ply"
    if not os.path.exists(folder_ply):
        os.mkdir(folder_ply)

    depth_path = 'D:\\pinhole_projection\\depth.txt'
    rgb_path = 'D:\\pinhole_projection\\rgb.txt'
    traj_path = 'D:\\pinhole_projection\\trajectory.xyz'
    odo_path = 'D:\\pinhole_projection\\odometry.log'

    lut_path = 'D:\\calibration\\Depth AHaT_lut.bin'
    rig2campath = 'D:\\calibration\\Depth AHaT_extrinsics.txt'
    # lookup table to extract xyz from depth
    lut = load_lut(lut_path)
    # from camera to rig space transformation (fixed)
    rig2cam = load_extrinsics(rig2campath)
    # Create virtual pinhole camera
    scale = 1
    width = 320 * scale
    height = 288 * scale
    focal_length = 200 * scale
    intrinsic_matrix = np.array([[focal_length, 0, width / 2.],
                                 [0, focal_length, height / 2.],
                                 [0, 0, 1.]])
    # Save virtual pinhole information inside calibration.txt
    intrinsic_path = 'D:\\pinhole_projection\\calibration.txt'
    intrinsic_list = [intrinsic_matrix[0, 0], intrinsic_matrix[1, 1],
                      intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]]
    with open(str(intrinsic_path), "w") as p:
        p.write(f"{intrinsic_list[0]} \
                {intrinsic_list[1]} \
                {intrinsic_list[2]} \
                {intrinsic_list[3]} \n")
    rgb_timestamps=[]
    depth_timestamps=[]
    # create folders
    folder_rgb = (folder + "/rgb")
    if not os.path.exists(folder_rgb):
        os.mkdir(folder_rgb)
    file_rgb_raw = (folder + "/rgb_raw")
    if not os.path.exists(file_rgb_raw):
        os.mkdir(file_rgb_raw)
    file_depth_raw = (folder + "/depth_raw")
    if not os.path.exists(file_depth_raw):
        os.mkdir(file_depth_raw)
    folder_depth = (folder + "/depth")
    if not os.path.exists(folder_depth):
        os.mkdir(folder_depth)
    i = 0
    while (i < 30):

        if np.any(video_receiver.latest_frame) and np.any(ahat_receiver.latest_frame):

            # if not os.path.exists(file):
            #     os.mkdir(file)
            # cv2.imshow('Photo Video Camera Stream', video_receiver.latest_frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            # cv2.imshow('Depth Camera Stream', ahat_receiver.latest_frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            # save rgb image names (timestamp)
            rgb_timestamp = video_receiver.get_timestamp(video_receiver.latest_header)
            pv2world = video_receiver.get_mat_from_header(video_receiver.latest_header)
            #print(pv2world)
            rgb_focal_length = video_receiver.get_focal_length(video_receiver.latest_header)
            #print(rgb_focal_length)
            principal_point = np.array([380.129,199.54])
            if len(rgb_timestamps) == 0:
                rgb_timestamps.append(rgb_timestamp)
                with open('D:\\pinhole_projection\\rgb.txt', 'a+') as f:
                    f.write(str(rgb_timestamp)+'.png'+'\n')
            elif len(rgb_timestamps) > 0 and rgb_timestamp != rgb_timestamps[-1]:
                rgb_timestamps.append(rgb_timestamp)
                with open('D:\\pinhole_projection\\rgb.txt', 'a+') as f:
                    f.write(str(rgb_timestamp)+'.png'+'\n')

            # save raw rgb images
            cv2.imwrite(file_rgb_raw +"/" + '%d.png'%rgb_timestamp, video_receiver.latest_frame)
            img_pv = cv2.imread(file_rgb_raw + "/" + '%d.png' % rgb_timestamp, -1)

            # save depth image names (timestamp)
            depth_timestamp = video_receiver.get_timestamp(ahat_receiver.latest_header)
            cv2.imwrite(file_depth_raw +"/" + '%d.png'%depth_timestamp, (ahat_receiver.latest_frame))
            img_depth = cv2.imread(file_depth_raw +"/" + '%d.png'%depth_timestamp, -1)
            height_depth, width_depth = img_depth.shape[0], img_depth.shape[1]
            assert len(lut) == width_depth * height_depth
            points = get_points_in_cam_space(img_depth, lut)

            # save rig_to_world_transform matrix
            rig2world = ahat_receiver.get_mat_from_header(ahat_receiver.latest_header)
            #print(rig2world)
            xyz, cam2world_transform = cam2world(points, rig2cam, rig2world)
            camera_center = cam2world_transform @ np.array([0, 0, 0, 1])
            traj = camera_center[:3]
            pose = cam2world_transform
            print(rgb_focal_length)

            # Project from depth to pv going via world space
            rgb, depth = project_on_pv(xyz, img_pv, pv2world, rgb_focal_length, principal_point)
            # Project depth on virtual pinhole camera and save corresponding
            rgb_proj, depth = project_on_depth(points, rgb, intrinsic_matrix, width, height)
            depth = (depth * DEPTH_SCALING_FACTOR).astype(np.uint16)
            cv2.imwrite(folder_depth + "/" + '%d.png' % depth_timestamp, (depth).astype(np.uint16))

            # save single ply
            output_path = folder_ply+"/" + '%d.ply'%depth_timestamp
            colored_points = rgb[:, 0] > 0
            xyz = xyz[colored_points]
            rgb = rgb[colored_points]
            save_ply(output_path, points, rgb, cam2world_transform=cam2world_transform)

            # save rgb proj image
            cv2.imwrite(folder_rgb +"/" + '%d.png'%rgb_timestamp, rgb_proj)
            if len(depth_timestamps) == 0:
                depth_timestamps.append(depth_timestamp)
                with open('D:\\pinhole_projection\\depth.txt', 'a+') as f:
                    f.write(str(depth_timestamp)+'.png'+'\n')
                with open(str(odo_path), "a+") as of:
                    of.write(f"{i} {i} {i}\n")
                    of.write(f"{pose[0, 0]} {pose[0, 1]} {pose[0, 2]} {pose[0, 3]}\n")
                    of.write(f"{pose[1, 0]} {pose[1, 1]} {pose[1, 2]} {pose[1, 3]}\n")
                    of.write(f"{pose[2, 0]} {pose[2, 1]} {pose[2, 2]} {pose[2, 3]}\n")
                    of.write(f"{pose[3, 0]} {pose[3, 1]} {pose[3, 2]} {pose[3, 3]}\n")
                    of.close()
                with open(str(traj_path), "a+") as tf:
                    tf.write(f"{traj}\n")
                    tf.close()
            elif len(depth_timestamps) > 0 and depth_timestamp != depth_timestamps[-1]:
                print(depth_timestamp, depth_timestamps[-1])
                depth_timestamps.append(depth_timestamp)
                with open('D:\\pinhole_projection\\depth.txt', 'a+') as f:
                    f.write(str(depth_timestamp)+'.png'+'\n')
                with open(str(odo_path), "a+") as of:
                    of.write(f"{i} {i} {i}\n")
                    of.write(f"{pose[0,0]} {pose[0,1]} {pose[0,2]} {pose[0,3]}\n")
                    of.write(f"{pose[1,0]} {pose[1,1]} {pose[1,2]} {pose[1,3]}\n")
                    of.write(f"{pose[2,0]} {pose[2,1]} {pose[2,2]} {pose[2,3]}\n")
                    of.write(f"{pose[3,0]} {pose[3,1]} {pose[3,2]} {pose[3,3]}\n")
                    of.close()
                with open(str(traj_path), "a+") as tf:
                    tf.write(f"{traj}\n")
                    tf.close()
            i = i + 1
            print(i)
            if (i > 30):
                break
    pid = os.getpid()
    p = psutil.Process(pid)
    p.terminate()


