from abc import abstractclassmethod
import time
import cv2
import numpy as np
from poseutil import PoseDetector


class Fitness():
    '''
    所有健身类型的父类
    '''
    def check_pose():
        pass


class Factory():
    '''
    工厂类
    '''
    @abstractclassmethod
    def get_fitness_type(self, name):
        if name == 'pullup':
            return Pullups()
        if name == 'situp':
            return Situp()
        if name == 'pushup':
            return Pushup()
        if name == 'squat':
            return Squat()
        if name == 'plank':
            return Plank()
        if name == 'lunge':
            return Lunge()


class Pullups(Fitness):
    '''
    引体向上类
    '''

    def check_pose(self, detector, img, dir, count):
        # 检测视频图片帧中人体姿势
        img = detector.find_pose(img, draw=True)
        # 获取人体姿势列表数据
        positions = detector.find_positions(img)
        # 获取图像尺寸
        h, w, c = img.shape

        if positions:
            # 右手肘的角度
            right_angle = detector.find_angle(img, 12, 14, 16)
            # 以170到20度检测右手肘弯曲的程度
            right_per = np.interp(right_angle, (20, 170), (100, 0))
            # 进度条高度数据
            right_bar = np.interp(right_angle, (20, 170), (h//2, 0))
            # 使用opencv画进度条和写右手肘弯曲的程度
            cv2.rectangle(img, (w//5-12, h//2-100),
                          (w//5+12, h-100), (0, 255, 0), 3)
            cv2.rectangle(img, (w//5-12, h-100-int(right_bar)),
                          (w//5+12, h-100), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(int(right_per)) + '%', (w//5-24, h-70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # 左手肘的角度
            left_angle = detector.find_angle(img, 11, 13, 15)
            left_per = np.interp(left_angle, (20, 170), (100, 0))
            left_bar = np.interp(left_angle, (20, 170), (h//2, 0))

            cv2.rectangle(img, (w//5*4-12, h//2-100),
                          (w//5*4+12, h-100), (0, 255, 0), 3)
            cv2.rectangle(img, (w//5*4-12, h-100-int(left_bar)),
                          (w//5*4+12, h-100), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(int(left_per)) + '%', (w//5*4-24, h-70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # 检测个数，我这里设置的是从20%做到80%，就认为是一个

            if (left_per >= 80 and right_per >= 80):
                if dir == 0:
                    count = count + 0.5
                    dir = 1

            if (left_per <= 20 and right_per <= 20):
                if dir == 1:
                    count = count + 0.5
                    dir = 0

            # 在视频上显示完成个数
            cv2.putText(img, str(int(count)), (100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 4)
        return dir, count


class Situp(Fitness):
    '''
    仰卧起坐类
    '''

    def check_pose(self, detector, img, dir, count):
        # 识别姿势
        img = detector.find_pose(img, draw=True)
        # 获取姿势数据
        positions = detector.find_positions(img)
        h, w, c = img.shape

        if positions:
            rectw = w//8
            if rectw <= 100:
                rectw = 100
            # 获取仰卧起坐的角度
            angle1 = detector.find_angle(img, 27, 25, 23)
            angle2 = detector.find_angle(img, 25, 23, 11)
            angle3 = 180 - detector.find_angle(img, 27, 23, 11)
            # 进度条长度, angle2取最大值时进度条为0，<60时为100
            bar = np.interp(angle2, (60, 100), (rectw*2, 0))
            cv2.rectangle(img, (w // 2 - rectw, h - 150),
                          (w//2+rectw, h - 100), (0, 255, 0), 2)
            cv2.rectangle(img, (w // 2 - rectw, h - 150),
                          (int(bar)+w//2-rectw, h - 100), (0, 255, 0), cv2.FILLED)
            if angle1 in range(60, 120) and (angle2 + angle3) > 100:
                if angle2 <= 60:
                    if dir == 0:
                        count = count+0.5
                        dir = 1
                if angle2 >= 100:
                    if dir == 1:
                        count = count + 0.5
                        dir = 0
            cv2.putText(img, str(int(count)), (100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 4, cv2.LINE_AA)
        return dir, count


class Pushup(Fitness):
    '''
    俯卧撑类
    '''

    def check_pose(self, detector, img, dir, count):
        # 识别姿势
        img = detector.find_pose(img, draw=True)
        # 获取姿势数据
        positions = detector.find_positions(img)
        # 方向与个数# 1为下，0为上
        h, w, c = img.shape

        if positions:
            rectw = w//8
            if rectw <= 100:
                rectw = 100
            # 获取俯卧撑的角度
            angle1 = detector.find_angle(img, 12, 24, 26)
            angle2 = detector.find_angle(img, 12, 14, 16)
            # 进度条长度
            bar = np.interp(angle2, (45, 150), (0, rectw*2))
            cv2.rectangle(img, (w // 2 - rectw, h - 150),
                          (w//2+rectw, h - 100), (0, 255, 0), 2)
            cv2.rectangle(img, (w // 2 - rectw, h - 150),
                          (int(bar)+w//2-rectw, h - 100), (0, 255, 0), cv2.FILLED)
            # 角度小于60度认为撑下
            if angle2 <= 60 and angle1 >= 160:
                if dir == 0:
                    count = count + 0.5
                    dir = 1
            # 角度大于125度认为撑起
            if angle2 >= 125 and angle1 >= 160:
                if dir == 1:
                    count = count + 0.5
                    dir = 0
            cv2.putText(img, str(int(count)), (100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 4)
        return dir, count


class Squat(Fitness):
    '''
    深蹲类
    '''

    def check_pose(self, detector, img, dir, count):
        # 识别姿势
        img = detector.find_pose(img, draw=True)
        # 获取姿势数据
        positions = detector.find_positions(img)
        # 方向与个数# 1为下，0为上
        h, w, c = img.shape

        if positions:
            rectw = w//8
            if rectw <= 100:
                rectw = 100
            # 获取俯卧撑的角度
            angle1 = detector.find_angle(img, 24, 26, 28)
            angle2 = detector.find_angle(img, 23, 25, 27)
            # 进度条长度
            bar = np.interp((angle1+angle2)//2, (90, 130), (rectw*2, 0))
            cv2.rectangle(img, (w // 2 - rectw, h - 150),
                          (w//2+rectw, h - 100), (0, 255, 0), 2)
            cv2.rectangle(img, (w // 2 - rectw, h - 150),
                          (int(bar)+w//2-rectw, h - 100), (0, 255, 0), cv2.FILLED)
            # 角度小于60度认为撑下
            if angle1 <= 90 and angle2 <= 90:
                if dir == 0:
                    count = count + 0.5
                    dir = 1
            # 角度大于125度认为撑起
            if angle2 >= 130 and angle1 >= 130:
                if dir == 1:
                    count = count + 0.5
                    dir = 0
            cv2.putText(img, str(int(count)), (100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 4)
        return dir, count


class Plank(Fitness):
    '''
    平板支撑类
    '''

    def check_pose(self, detector, img, currTime, count):
        # 识别姿势
        img = detector.find_pose(img, draw=True)
        # 获取姿势数据
        positions = detector.find_positions(img)
        h, w, c = img.shape

        if positions:
            rectw = w//8
            if rectw <= 100:
                rectw = 100
            # 获取腰部的角度
            angle1 = detector.find_angle(img, 11, 23, 25)
            # 获取肘部的角度
            angle2 = detector.find_angle(img, 11, 13, 15)

            temp = time.time()
            # 角度小于60度认为撑下
            if angle1 <= 180 and angle1 >= 150:
                if angle2 >= 70 and angle2 <= 110:
                    count += temp - currTime
            currTime = temp
            min = int(count//60)
            sec = int(count % 60)
            if min >= 60:
                min = 0
            if(min < 10):
                str1 = '0'+str(min)
            else:
                str1 = str(min)
            if(sec < 10):
                str2 = '0'+str(sec)
            else:
                str2 = str(sec)

            cv2.putText(img, str1 + ' : ' + str2, (100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 4)
        return currTime, count

class Lunge(Fitness):
    '''
    箭步蹲类
    '''

    def check_pose(self, detector, img, dir, count):
        # 识别姿势
        img = detector.find_pose(img, draw=True)
        # 获取姿势数据
        positions = detector.find_positions(img)
        h, w, c = img.shape

        if positions:
            rectw = w//8
            if rectw <= 100:
                rectw = 100
            # 获取右膝盖的角度
            angle1 = detector.find_angle(img, 23, 25, 27)
            # 获取左膝盖的角度
            angle2 = detector.find_angle(img, 24, 26, 28)

            if angle1 <= 90 and angle1 <= 90:
                if dir==0:
                    count += 0.5
                    dir = 1
            if angle1 >= 160 and angle2 >=160:
                if dir==1:
                    count += 0.5
                    dir = 0
            cv2.putText(img, str(int(count)), (100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 4)
        return dir, count
