#! /usr/bin/env python3
import rospy

if __name__ == '__main__':
    rospy.init_node('my_first_node')

    rospy.loginfo("This is my first node!")
    rospy.Rate(10)

    while not rospy.is_shutdown():
        rospy.loginfo("Hello, World!")
        rospy.sleep(0.1)
