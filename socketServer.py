from socketserver import BaseRequestHandler,ThreadingTCPServer
import threading
import socket
import time
import cv2
import numpy
import copy
import sys
from fitnesstype import Factory

from poseutil import PoseDetector

BUF_SIZE=1024

def recvall(sock, count):
        buf = b''  # buf是一个byte类型
        temp=count
        while count:
            # 接受TCP套接字的数据。数据以字符串形式返回，count指定要接收的最大数据量.
            newbuf = sock.recv(count)
            if not newbuf: return None
            buf += newbuf
            count -= len(newbuf)
        if temp == 16: 
            print('b'+buf)
        return buf

class Handler(BaseRequestHandler):
    def handle(self):
        address,pid = self.client_address
        print('%s connected!'%address)
        # 创建一个PoseDetector类的对象
        detector = PoseDetector()
        
        # 方向和完成次数的变量
        dir = 0  #0为向下，1为向上
        count = 0
        conn = self.request
        length=recvall(conn,16)
        postureData = recvall(conn, int(length))  # 根据获得的文件长度，获取图片文件
        posture=bytes.decode(postureData)
        # posture='pullup'
        print(posture)
        #获取对应的姿势类
        fitness=Factory.get_fitness_type(posture)
        while True:
            #start = time.time()  # 用于计算帧率信息
            length = recvall(conn, 16)  # 获得图片文件的长度,16代表获取长度
            stringData = recvall(conn, int(length))  # 根据获得的文件长度，获取图片文件
            data = numpy.frombuffer(stringData, numpy.uint8)  # 将获取到的字符流数据转换成1维数组
            decimg = cv2.imdecode(data, cv2.IMREAD_COLOR)  # 将数组解码成图像
            try:
                dir,count=fitness.check_pose(detector,decimg,dir,count)
            except Exception as e:
                print(e)
                continue
            # cv2.imshow('SERVER', decimg)  # 显示图像
            # cv2.waitKey(1)
            # # 进行下一步处理
            # # 将帧率信息回传，主要目的是测试可以双向通信
            # end = time.time()
            # seconds = end - start
            # fps = 1 / seconds
            #返回已处理图像到客户端
            #conn.send(bytes(str(int(fps)), encoding='utf-8'))
            image = copy.deepcopy(decimg)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            result, imgencode = cv2.imencode('.jpg', image)
            # 建立矩阵
            data = numpy.array(imgencode)
            # 将numpy矩阵转换成字符形式，以便在网络中传输
            img_Data = data.tobytes()
            # 先发送要发送的数据的长度
            # ljust() 方法返回一个原字符串左对齐,并使用空格填充至指定长度的新字符串
            conn.send(str.encode(str(len(img_Data)).ljust(16)))
            # # print(img_Data)
            # # 发送数据
            conn.send(img_Data)
        



            

if __name__ == '__main__':
    if len(sys.argv)>1:
        HOST=sys.argv[1].split()[0]
        PORT=int(sys.argv[1].split()[1])
    else:
        HOST = '0.0.0.0'
        PORT = 8004
    ADDR = (HOST,PORT)
    server = ThreadingTCPServer(ADDR,Handler)  #参数为监听地址和已建立连接的处理类
    print('listening')
    server.serve_forever()  #监听，建立好TCP连接后，为该连接创建新的socket和线程，并由处理类中的handle方法处理
    print(server)
