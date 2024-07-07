#!/usr/bin/env python3
"""
A ROS node for publishing visuotactile images.
"""
__author__ = 'Paul-Otto Müller'
__date__ = '13.03.2023'

import numpy as np

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from ffmpegcv.ffmpeg_reader_camera import query_camera_devices
import threading
import os

import canfnet.visuotactile_sensor.visuotactile_interface as vistac_interface
from canfnet.utils.utils import PrintColors

VISTAC_DEVICE: str = rospy.get_param('/visuotactile_sensor_node/tactile_device', 'GelSightMini')
DIGIT_SERIAL_NR: str = rospy.get_param('/visuotactile_sensor_node/digit_serial', 'D20025')
TACTILE_CAM_UNDISTORT: bool = rospy.get_param('/visuotactile_sensor_node/tactile_cam_undistort', True)

def get_camera_id(camera_name):
    cam_num = None
    if os.name == 'nt':
        cam_num = find_cameras_windows(camera_name)
    else:
        cams = []
        for file in os.listdir("/sys/class/video4linux"):
            real_file = os.path.realpath("/sys/class/video4linux/" + file + "/name")
            with open(real_file, "rt") as name_file:
                name = name_file.read().rstrip()
            if camera_name in name:
                cams.append(int(re.search("\d+$", file).group(0)) ) #-1
                found = "FOUND!"
            else:
                found = "      "
            print("{} {} -> {}".format(found, file, name))
        if cams:
            cam_num = min(cams)
            print("CAM ID:", cam_num)
    return cam_num

def get_camera_names():
    devices = query_camera_devices()
    camera_ids = list(devices.keys())
    names = []
    for id in camera_ids:
        name = devices[id][0]
        if "GelSight Mini" in name:
            print(f"Found GelSight Mini: {devices[id]}, id: {id}")
            names.append(name)
    return names


def run(id):

    bridge: CvBridge = CvBridge()
    pub_rate: int = 60
    pub_rate = 1
    visuotactile_device = vistac_interface.GelSightMini(device_id=id,
                                                            undistort_image=TACTILE_CAM_UNDISTORT)

    visuotactile_device.connect()
    
    pub: rospy.Publisher = rospy.Publisher(f'/canfnet/visuotactile_image{id}', Image, queue_size=pub_rate)

    rate = rospy.Rate(pub_rate)
    while not rospy.is_shutdown():
        image_: np.ndarray = visuotactile_device.get_image()
        pub.publish(bridge.cv2_to_imgmsg(image_, encoding='rgb8'))
        rate.sleep()


if __name__ == '__main__':
    rospy.init_node(f'visuotactile_sensor_node', anonymous=True)
    rospy.loginfo(f"{PrintColors.OKBLUE} f[VisuotactileSensorNode] Node has been initialized. {PrintColors.ENDC}")
    device_names = get_camera_names()
    #device_names = ["GelSight Mini R0B 2BDA-69HN", "GelSight Mini R0B 2BE7-169L"]
    threads = []
    for device_name in device_names:
        id = get_camera_id(device_name)
        t = threading.Thread(target=run, args=(id,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()
