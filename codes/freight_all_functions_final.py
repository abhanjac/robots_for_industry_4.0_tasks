#!/usr/bin/env python

import rospy, tf, numpy as np, copy, cv2, os, sys, termios, tty, time, signal
import subprocess, datetime
from matplotlib import pyplot as plt
from cv_bridge import CvBridge, CvBridgeError
from moveit_msgs.msg import MoveItErrorCodes, MoveGroupAction, MoveGroupGoal
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
from moveit_python import MoveGroupInterface, PlanningSceneInterface
from geometry_msgs.msg import PoseStamped, Pose, PoseArray, Point, Quaternion, Twist
from nav_msgs.msg import Odometry
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.msg import PointHeadAction, PointHeadGoal
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from control_msgs.msg import GripperCommandAction, GripperCommandGoal
from sensor_msgs.msg import JointState, Image, CameraInfo
from power_msgs.msg import BatteryState
import actionlib, moveit_msgs.msg
from geometry_msgs.msg import *
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.msg import Constraints, JointConstraint, PositionConstraint
from moveit_msgs.msg import OrientationConstraint, BoundingVolume

from moveit_msgs.msg import PlaceLocation
from trajectory_msgs.msg import JointTrajectory
from moveit_python import PickPlaceInterface
from moveit_python.geometry import rotate_pose_msg_by_euler_angles
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseWithCovarianceStamped

################################################################################
################################################################################

# Move base using navigation stack
class MoveBaseClient(object):

    def __init__(self):
        self.client = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        rospy.loginfo("Waiting for moveBaseFetch...")
        self.client.wait_for_server()
        
        # This Subscriber will help to update the current pose of the robot 
        # in some pose variables.
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, 
                                 callback=self._current_amcl_pose_callback)

        # Variables to hold the current pose of the robot. Initializing this to None.
        self.current_amcl_pose = None
        
        # By just calling this wait_for_message function, the Subscribers are 
        # called once and hence the callback functions are run once, which updates
        # the 'self.' variables which the callback functions are supposed to update, 
        # otherwise they will always start with the initialized values.
        msg = rospy.wait_for_message('/amcl_pose', PoseWithCovarianceStamped)

################################################################################

    def _current_amcl_pose_callback(self, msg):
        '''
        Callback function of the Subscriber. Updates the current amcl pose of 
        the robot.
        '''
        # Just updating the callback function.
        msg = rospy.wait_for_message('/amcl_pose', PoseWithCovarianceStamped)
        self.current_amcl_pose = msg.pose.pose

################################################################################

    def goto_with_angle(self, x, y, theta, frame="map"):
        '''
        This function takes in the target x, y position and the final yaw angle 
        and positions the robot to that location. After positioning properly, it 
        orients the robot at the target yaw angle. z for position is always 0 
        as this is a ground robot, and x and y for orientation are always 0 as 
        there will be no pitch or roll.
        '''
        move_goal = MoveBaseGoal()
        move_goal.target_pose.pose.position.x = x
        move_goal.target_pose.pose.position.y = y
        move_goal.target_pose.pose.orientation.z = np.sin(theta/2.0)
        move_goal.target_pose.pose.orientation.w = np.cos(theta/2.0)
        move_goal.target_pose.header.frame_id = frame
        move_goal.target_pose.header.stamp = rospy.Time.now()

        # TODO wait for things to work
        self.client.send_goal(move_goal)
        self.client.wait_for_result()

################################################################################

    def goto(self, targetPose, frame="map"):
        '''
        This function takes in the target x, y, z position and x, y, z, w 
        orientation for the final pose of the robot. After positioning properly, 
        it orients the robot at the target orientation. z for position should 
        always be 0 as this is a ground robot.
        '''
        move_goal = MoveBaseGoal()
        move_goal.target_pose.pose = targetPose
        move_goal.target_pose.header.frame_id = frame
        move_goal.target_pose.header.stamp = rospy.Time.now()

        # TODO wait for things to work
        self.client.send_goal(move_goal)
        self.client.wait_for_result()

################################################################################
################################################################################

class freightRobotClassObj(object):
    '''
    This class will incorporate all the basic movements that the freight can do.
    '''
    def __init__(self):
        self.nodeName = 'freight_all_functions' 
        rospy.init_node(self.nodeName)    # Initializing node.
        self.namespace = rospy.get_namespace()
        # Appending namespace to node name.
        rospy.resolve_name(self.nodeName, caller_id=self.namespace)

##########----------------------------------------------------------------------

        # Create a publisher which can 'talk' to robot and tell it to move
        # Tip: You may need to change cmd_vel_mux/input/navi to /cmd_vel 
        # if you're not using robot2.
        self._cmd_vel_pub = rospy.Publisher(self.namespace + 'base_controller/command', 
                                               Twist, queue_size=5)
        
        # Subscribing to the odom topic to see how far the robot has moved.
        rospy.Subscriber(self.namespace + 'odom', Odometry, self._odom_callback, 
                                                         queue_size=10)

        # Variables to hold the Odometry of the robot. Initializing this to None.
        self.odom = None
        
##########----------------------------------------------------------------------

        self.refFrame = 'base_link'

##########----------------------------------------------------------------------

        # Subscribing to the battery_state topic to get the current battery voltage.
        rospy.Subscriber(self.namespace + 'battery_state', BatteryState, 
                                                     self._battery_state_callback)

        self.currentBattChargeLvl = 0.0    # Initializing battery charge level variable.

##########----------------------------------------------------------------------

        # By just calling this wait_for_message function, the Subscribers are 
        # called once and hence the callback functions are run once, which updates
        # the 'self.' variables which the callback functions are supposed to update, 
        # otherwise they will always start with the initialized values.
        msg = rospy.wait_for_message(self.namespace + 'battery_state', BatteryState)
        msg = rospy.wait_for_message(self.namespace + 'odom', Odometry)

################################################################################

    def _odom_callback(self, msg):
        '''
        Callback function of the Subscriber. Updates the x, y coordinates of the robot.
        '''
        self.odom = msg.pose.pose

################################################################################

    def _battery_state_callback(self, msg):
        '''
        Callback function that saves the robot battery voltage in currentBattState.
        '''
        self.currentBattChargeLvl = msg.charge_level
        
################################################################################

    def _linear_distance(self, pose1, pose2):
        '''
        Calculates the linear distance between two poses.
        '''
        pos1, pos2 = pose1.position, pose2.position
        dx, dy, dz = pos1.x - pos2.x, pos1.y - pos2.y, pos1.z - pos2.z
        return np.sqrt(dx * dx + dy * dy + dz * dz)

################################################################################

    def _x_axis_from_quaternion(self, q):
        '''
        Calculates x value from Quaternion.
        '''
        m = tf.transformations.quaternion_matrix([q.x, q.y, q.z, q.w])
        return m[:3, 0]  # First column

################################################################################

    def _yaw_from_quaternion(self, q, outputInDegrees=False):
        '''
        Calculates yaw angle from Quaternion. Output angle is in radians.
        If outputInDegrees flag is True then the output value is given in degrees.
        '''
        m = tf.transformations.quaternion_matrix([q.x, q.y, q.z, q.w])
        #print(m)
        #print(m[1, 0])
        angle = np.arctan2(m[1, 0], m[0, 0])
        if outputInDegrees:     angle = np.rad2deg(angle)
        return angle

################################################################################

    def _yaw_from_quaternion_old(self, q, outputInDegrees=False):
        '''
        Calculates yaw angle from Quaternion. Output angle is in radians.
        If outputInDegrees flag is True then the output value is given in degrees.
        '''
        m = tf.transformations.quaternion_matrix([q.x, q.y, q.z, q.w])
        #print(m)
        #print(m[1, 0])
        angle = np.arctan(m[1, 0] / m[0, 0])
        if outputInDegrees:     angle = np.rad2deg(angle)
        return angle

################################################################################

    def _pitch_from_quaternion(self, q, outputInDegrees=False):
        '''
        Calculates pitch angle from Quaternion. Output angle is in radians.
        If outputInDegrees flag is True then the output value is given in degrees.
        '''
        angle = np.arctan2(2*q.x*q.w - 2*q.y*q.z, 1 - 2*q.x*q.x - 2*q.z*q.z);
        if outputInDegrees:     angle = np.rad2deg(angle)
        return angle

################################################################################

    def _angularDistance(self, pose1, pose2, outputInDegrees=False):
        '''
        Calculates the angular distance between two poses. Output angle is in radians.
        If outputInDegrees flag is True then the output value is given in degrees.
        '''
        q1, q2 = pose1.orientation, pose2.orientation
        y1, y2 = self._yaw_from_quaternion(q1), self._yaw_from_quaternion(q2)
        angle = np.fabs(y1 - y2) % (2 * np.pi)
        if outputInDegrees:     angle = np.rad2deg(angle)
        return angle

################################################################################

    def moveBase(self, linearSpeed, angularSpeed, verbose=True):
        '''
        Moves the base instantaneously at given linear and angular speeds.
        'Instantaneously' means that this method must be called continuously in
        a loop for the robot to move.
        Args:
        linear_speed: The forward/backward speed, in meters/second. A
        positive value means the robot should move forward.
        angular_speed: The rotation speed, in radians/second. A positive
        value means the robot should rotate clockwise.
        verbose:    if True, prints the status of the task in the end.
        Sample usage:
        while CONDITION:
            robot.moveBase(0.2, 0)
        '''
        cmd = Twist()
        cmd.linear.x = linearSpeed
        cmd.angular.z = angularSpeed
        self._cmd_vel_pub.publish(cmd)
        if verbose:     print('Moving base...\n')

################################################################################

    def baseGoForward(self, distance, speed=0.3, verbose=True):
        '''
        Moves the robot a certain distance.
        It's recommended that the robot move slowly. If the robot moves too
        quickly, it may overshoot the target. Note also that this method does
        not know if the robot's path is perturbed (e.g., by teleop). It stops
        once the distance traveled is equal to the given distance.
        You cannot use this method to move less than 1 cm.
        Args:
        distance: The distance, in meters, to rotate. A positive value
        means forward, negative means backward.
        max_speed: The maximum speed to travel, in meters/second.
        Sample usage:
        robot.baseGoForward(0.1)
        '''
        while self.odom is None:    # Waiting for odom variable to be updated.
            rospy.sleep(0.1)
            
        start = copy.deepcopy(self.odom)
        direction = -1 if distance < 0 else 1
        
        # Robot will stop if we don't keep telling it to move.  
        # How often should we tell it to move? 25 hz.
        rate = rospy.Rate(25)

        while self._linear_distance(start, self.odom) < np.fabs(distance):
            if rospy.is_shutdown():     break
            self.moveBase(direction * speed, 0, verbose=False)
            rate.sleep()
            
        if verbose:     print('Moving base forward...\n')

################################################################################

    def turnBase(self, angularDistance, speed=0.3, verbose=True):
        '''
        Rotates the robot a certain angle.
        This illustrates how to turn the robot by checking that the X-axis of
        the robot matches that of the goal.
        Args:
        angularDistance: The angle, in radians, to rotate. A positive
        value rotates counter-clockwise.
        speed: The maximum angular speed to rotate, in radians/second.
        Sample usage:
        robot.turnBase(30 * np.pi / 180)
        '''
        while self.odom is None:    # Waiting for odom variable to be updated.
            rospy.sleep(0.1)
        
        direction = -1 if angularDistance < 0 else 1

        currentYaw = self._yaw_from_quaternion(self.odom.orientation)
        goalYaw = currentYaw + angularDistance
        goalXaxis = np.array([np.cos(goalYaw), np.sin(goalYaw), 0])
        xAxis = self._x_axis_from_quaternion(self.odom.orientation)

        # Robot will stop if we don't keep telling it to move.  
        # How often should we tell it to move? 25 hz.
        rate = rospy.Rate(100)

        while not np.allclose(xAxis, goalXaxis, atol=0.01):
            if rospy.is_shutdown():     break
            self.moveBase(0, direction * speed, verbose=False)
            xAxis = self._x_axis_from_quaternion(self.odom.orientation)
            rate.sleep()

        if verbose:     print('Turned with x axis measurment...\n')

################################################################################

    def turnBaseAlternate(self, angularDistance, speed=0.3, verbose=True):
        '''
        Rotates the robot a certain angle.
        This illustrates how to turn the robot using yaw angles and careful
        accounting.
        Args:
        angularDistance: The angle, in radians, to rotate. A positive
        value rotates counter-clockwise.
        speed: The maximum angular speed to rotate, in radians/second.
        '''
        while self.odom is None:    # Waiting for odom variable to be updated.
            rospy.sleep(0.1)
        
        direction = -1 if angularDistance < 0 else 1

        currentCoord = self._yaw_from_quaternion(self.odom.orientation) % (2 * np.pi)
        endCoord = (currentCoord + angularDistance) % (2 * np.pi)
        rate = rospy.Rate(25)

        while True:
            if rospy.is_shutdown():     break
            currentCoord = self._yaw_from_quaternion(self.odom.orientation) % (2 * np.pi)
            remaining = (direction * (endCoord - currentCoord)) % (2 * np.pi)
            if remaining < 0.01:        break

            speed = max(0.25, min(1, remaining))
            self.moveBase(0, direction * speed, verbose=False)
            rate.sleep()
        
        if verbose:     print('Turned with yaw angle measurement...\n')

################################################################################
################################################################################
################################################################################
################################################################################

if __name__ == '__main__':
    
    freightRobot = freightRobotClassObj()

############------------------------Testing ground movements freight----------------------------------------------    

    #freightRobot.baseGoForward(0.5)
    #freightRobot.baseGoForward(-0.5)
    #freightRobot.turnBase(45*np.pi/180)
    #freightRobot.turnBase(-45*np.pi/180)
    ###freightRobot.turnBaseAlternate(-60 * np.pi / 180)
    
###########------------------------------Navigating the freight---------------------------------------------

    # Make sure sim time is working
    while not rospy.Time.now():
        pass

    # First moving into an empty region. This will help the fetch to have a better 
    # estimate of its current position.
    freightRobot.baseGoForward(0.5)

    # Setup clients
    moveBaseFreight = MoveBaseClient()

    rospy.loginfo("Moving to table...")

    targetPose = Pose(Point(0.58, 5.94, 0.0), Quaternion(0.0, 0.0, -0.77522, 0.6317))
    moveBaseFreight.goto(targetPose)
    print(moveBaseFreight.current_amcl_pose)
    
    targetPose = Pose(Point(-1.87, -0.221, 0.0), Quaternion(0.0, 0.0, -0.13775, 0.9905))
    moveBaseFreight.goto(targetPose)
    print(moveBaseFreight.current_amcl_pose)

    freightRobot.baseGoForward(1.1, verbose=False)