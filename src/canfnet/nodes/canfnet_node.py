#!/usr/bin/env python3
"""
A ROS node for estimating and publishing a normal force and its distribution from visuotactile images.
"""
__author__ = 'Paul-Otto Müller'
__date__ = '13.03.2023'

from pathlib import Path

import cv2
import rospy
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import CompressedImage, Image
from typing import Tuple, Union, Optional
from cv_bridge import CvBridge
from torch import device, Tensor

import canfnet.unet.predict as unet
from canfnet.msg import UNetEstimation
from canfnet.unet.unet import UNet
import threading
from functools import partial
from ffmpegcv.ffmpeg_reader_camera import query_camera_devices
import torch

def get_camera_ids():
    devices = query_camera_devices()
    camera_ids = list(devices.keys())
    ids = []
    for id in camera_ids:
        name = devices[id][0]
        if "GelSight Mini" in name:
            print(f"Found GelSight Mini: {devices[id]}, id: {id}")
            ids.append(id)
    return ids

FILE_DIR: Path = Path(__file__).parent.parent.resolve()
MODEL_PATH = rospy.get_param('/canfnet_node/model', Path(FILE_DIR, 'models', 'GelSightMini',
                                                         'model_23-02-2023_16-49-19_256_gelsight_mini.pth'))
VISTAC_DEVICE: str = rospy.get_param('/canfnet_node/tactile_device', 'GelSightMini')
TORCH_DEVICE: device = torch.cuda.set_device(0)
CANFNET_FORCE_FILT: bool = rospy.get_param('/canfnet_node/canfnet_force_filt', True)
UNET: UNet

# GelSight Mini.
NORM_IMG: Optional[Tuple[Tensor, Tensor]] = (Tensor([0.4907543957, 0.4985137582, 0.4685586393]),
                                             Tensor([0.0307641067, 0.0246135872, 0.0398434214]))  # (mean, std).
NORM_DIS: Optional[Tuple[Tensor, Tensor]] = (Tensor([-6.0596092226e-05]), Tensor([0.0002053244]))  # (mean, std).

# DIGIT.
if VISTAC_DEVICE == 'DIGIT':
    NORM_IMG: Optional[Tuple[Tensor, Tensor]] = (Tensor([0.5024564266, 0.4860377908, 0.5020657778]),
                                                 Tensor([0.0415902548, 0.0462468602, 0.0575232506]))  # (mean, std).
    NORM_DIS: Optional[Tuple[Tensor, Tensor]] = (Tensor([-0.0001196197]), Tensor([0.0003911761]))  # (mean, std).


def decode_compressed_image(image: CompressedImage) -> np.ndarray:
    """
    Decodes compressed images.

    :param image: A compressed image.
    :return: The decompressed image in RGB format.
    """
    return cv2.imdecode(np.frombuffer(image.data, np.uint8), cv2.IMREAD_COLOR)


def publish_canfnet_estimation(image: Union[Image, CompressedImage], id) -> None:
    """
    Callback function for estimating the normal force and its distribution from visuotactile images and publishing
    both together with an RGB color image of the normal force distribution.

    :param image: A compressed or uncompressed visuotactile image.
    :return: None
    """
    bridge: CvBridge = CvBridge()
    pub: rospy.Publisher = rospy.Publisher(f'/canfnet/unet_estimation{id}', numpy_msg(UNetEstimation), queue_size=1)
    pub_img: rospy.Publisher = rospy.Publisher(f'/canfnet/unet_estimation_image{id}', Image, queue_size=1)
    unet_est: UNetEstimation = numpy_msg(UNetEstimation)()

    if isinstance(image, CompressedImage):
        image = np.asarray(decode_compressed_image(image))
    else:
        image = np.asarray(bridge.imgmsg_to_cv2(image, desired_encoding='rgb8'))

    if NORM_IMG is None:
        image = image.transpose((2, 0, 1))

    force, f_dis = unet.predict(image, UNET, TORCH_DEVICE, norm_img=NORM_IMG,
                                norm_dis=NORM_DIS, force_filter=CANFNET_FORCE_FILT)
    
    unet_est.force = force
    unet_est.force_dis = f_dis.reshape(f_dis.shape[0] * f_dis.shape[1])
    unet_est.header.stamp = rospy.Time.now()
    unet_est.header.seq += 1
    unet_est.header.frame_id = f'visuotactile_sensor{id}'

    # Convert the force distribution to a color image for display.
    norm = mpl.colors.Normalize(vmin=-0.00075, vmax=-0.00012)
    cmap = cm.get_cmap('RdYlBu_r')
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    f_dis_img = m.to_rgba(f_dis[:, :, 0], bytes=True, norm=True)
    f_dis_img = bridge.cv2_to_imgmsg(f_dis_img, encoding='rgba8')

    # Publish the normal force and its distribution as float array and as RGB color image.
    pub.publish(unet_est)
    pub_img.publish(f_dis_img)


def vistac_listener(id) -> None:
    """
    A listener for published visuotactile images.

    :return: None
    """
    global UNET
    UNET = unet.load_unet(MODEL_PATH, torch_device=TORCH_DEVICE)

    
    rospy.Subscriber(f'/canfnet/visuotactile_image{id}', Image, publish_canfnet_estimation, callback_args=id)

    rospy.spin()


if __name__ == '__main__':
    ids = get_camera_ids()
    rospy.init_node('unet', anonymous=True)
    threads = []
    for id in ids:
        t = threading.Thread(target=vistac_listener, args=(id,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()
