#! /usr/bin/env python

import rospy
from sensor_msgs.msg import JointState
import time

joint_positions = [0.0, -1.5, 0.0, 0.62, 0.0, 0.0]  

def publish_joint_states():
    count = 1
    rospy.init_node('joint_state_publisher_gui_controller', anonymous=True)
    pub = rospy.Publisher('/joint_states', JointState, queue_size=10)
    rate = rospy.Rate(100)  # 10hz
    joint_state = JointState()
    joint_state.name = ['joint2_to_joint1', 'joint3_to_joint2', 'joint4_to_joint3', 'joint5_to_joint4', 'joint6_to_joint5', 'joint6output_to_joint6']

    while not rospy.is_shutdown():
        
        if count == 1 and joint_positions[0] < 3.14:
            joint_positions[0] += 0.062
        elif count == 1 and joint_positions[0] >= 3.14:
            count = 2

        if count == 2 and joint_positions[0] > -3.14:
            joint_positions[0] -= 0.062
        elif count == 2 and joint_positions[0] <= -3.14:
            count = 1
        
        
        joint_state.header.stamp = rospy.Time.now()
        
        joint_state.position = joint_positions
        rospy.loginfo(joint_state)
        joint_state.velocity = []
        joint_state.effort = []

        pub.publish(joint_state)
        rate.sleep()

if __name__ == '__main__':

    try:
        publish_joint_states()
    except rospy.ROSInterruptException:
        pass