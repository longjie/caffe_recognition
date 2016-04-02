#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from rospeex_if import ROSpeexInterface

if __name__ == "__main__":
    # ROSノードの初期化
    rospy.init_node('check_rospeex')

    interface = ROSpeexInterface()
    interface.init()

    # open file
    f = open('synset_words_jp.txt', 'r')

    r = rospy.Rate(0.2) # 10hz
    for line in f:
        if rospy.is_shutdown():
            break
        line = line.rstrip()
        value = line.split(' ')
        print value[1]
        interface.say(value[1], 'ja', 'nict')
        r.sleep()

