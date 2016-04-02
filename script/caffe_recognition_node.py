#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 画像を受け取ってCNNで認識して結果を返すノード
# 入力: sensor_msgs/Image
# 出力: string

from __future__ import print_function
import sys, argparse
import chainer
from chainer import cuda
from caffe_recognition import caffenet
import rospy
from sensor_msgs.msg import Image
from caffe_recognition.srv import ClassifyImage
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from rospeex_if import ROSpeexInterface

class CaffeRecognitionServer:
    def __init__(self, network):
        self.network = network

        self.bridge = CvBridge()

        # 画像の購読
        rospy.Subscriber("/camera/image_raw", Image, self.image_callback, queue_size=1)
        
        # サービスの初期化
        rospy.Service('recognize', ClassifyImage, self.recognize)

        self.pub_cropped = rospy.Publisher('/image_cropped', Image, queue_size=1)

        self.categories = xp.loadtxt("synset_words_jp.txt", str, delimiter="\t")
        #self.categories = xp.loadtxt("synset_words.txt", str, delimiter="\t")

        #self.rospeex = ROSpeexInterface()
        #self.rospeex.init()

    def recognize(self, req):
        pass

    def image_callback(self, data):
        """画像を取得する"""
        try:
            self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        
        """ネットワークを適用する"""
        # 画像を中心から224x224にリサイズする
        in_size = self.network.in_size

        (h, w, c) = self.image.shape
        #print("(c,w,h)", (c,w,h))
        (cx, cy) = (w // 2, h // 2)
        #print("(cx, cy)", (cx, cy))
        (x_head, x_tail) = (cx - h // 2, cx + h // 2)
        (y_head, y_tail) = (0, h)

        #print("(x_head, x_tail)", (x_head, x_tail))
        #print("(y_head, y_tail)", (y_head, y_tail))

        image = self.image[y_head:y_tail, x_head:x_tail, :]
        image = cv2.resize(image, (in_size, in_size))
        # リサイズ画像の確認
        try:
            self.pub_cropped.publish(self.bridge.cv2_to_imgmsg(image, "bgr8"))
        except CvBridgeError as e:
            print (e)
    
        image = image.transpose(2, 0, 1).astype(xp.float32)
        image -= self.network.mean_image

        x_batch = xp.ndarray((1, 3, in_size, in_size), dtype=xp.float32)
        x_batch[0] = image
        x = chainer.Variable(x_batch, volatile=True)
        score = self.network.predict(x)

        prediction = zip(score.data[0].tolist(), self.categories)
        prediction.sort(cmp=lambda x, y: cmp(x[0], y[0]), reverse=True)

        print ('\033c')
        for rank, (score, name) in enumerate(prediction[:5], start=1):
            print('#%d | %s | %4.1f%%' % (rank, name, score * 100))

if __name__ == "__main__":
    # ROSノードの初期化
    rospy.init_node('caffe_recognition_server')

    # parse arguments
    sys.argv = rospy.myargv()
    parser = argparse.ArgumentParser(description='Object recognition by caffe CNN')
    parser.add_argument('--model_type',
                        choices=('alexnet', 'caffenet', 'googlenet'),
                        help='Model type (alexnet, caffenet, googlenet)', default='googlenet')
    parser.add_argument('--model',
                        help='Pretrained model file name')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='Zero-origin GPU ID (nevative value indicates CPU)')
    args = parser.parse_args()

    if args.gpu >= 0:
        cuda.check_cuda_available()
    xp = cuda.cupy if args.gpu >= 0 else np

    if args.model_type == 'googlenet':
        network = caffenet.GoogleNet(xp, args.model)
    else:
        print('googlenet only', file=stderr)

    server = CaffeRecognitionServer(network)

    rospy.spin()





