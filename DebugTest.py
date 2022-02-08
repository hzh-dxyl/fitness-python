# 导入opencv工具包
from unicodedata import name
import cv2
import numpy as np

# 导入姿势类型
from fitnesstype import Factory
# 导入姿势识别器
from poseutil import PoseDetector

if __name__ == '__main__':
    # 打开视频文件
    cap = cv2.VideoCapture(0)
    # cap.set(3,1920)
    # cap.set(4,1080)
    # 姿势识别器
    detector = PoseDetector()
    # 姿势名称
    posture = 'situp'

    fitness = Factory.get_fitness_type(posture)
    # 方向与个数
    dir = 0  # 0为躺下，1为坐起
    count = 0

    while True:
        # 读取摄像头，img为每帧图片
        success, img = cap.read()
        if success:
            dir, count = fitness.check_pose(detector, img, dir, count)
            # 打开一个Image窗口显示视频图片
            cv2.imshow('Image', img)

        else:
            # 视频结束退出
            break

        # 如果按下q键，程序退出
        key = cv2.waitKey(1)
        if key == ord('4'):
            break

    # 关闭摄像头
    cap.release()
    # 关闭程序窗口
    cv2.destroyAllWindows()
