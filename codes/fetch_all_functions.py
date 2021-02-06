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
from ar_track_alvar_msgs.msg import AlvarMarkers
from visualization_msgs.msg import Marker
import moveit_commander
from geometry_msgs.msg import *
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.msg import Constraints, JointConstraint, PositionConstraint
from moveit_msgs.msg import OrientationConstraint, BoundingVolume

from moveit_msgs.msg import PlaceLocation
from trajectory_msgs.msg import JointTrajectory
from moveit_python import PickPlaceInterface
from moveit_python.geometry import rotate_pose_msg_by_euler_angles
from grasping_msgs.msg import FindGraspableObjectsAction, FindGraspableObjectsGoal
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseWithCovarianceStamped

import tensorflow as tfl
from utils import *
from network import *

################################################################################
################################################################################
################################################################################
################################################################################

# This class is copied from moveit_goal_builder.py file obtained from: 
# https://github.com/subodh-malgonde/Robotics/blob/indigo-devel/fetch_api/src/fetch_api/moveit_goal_builder.py

class MoveItGoalBuilder(object):
    '''
    Builds a MoveGroupGoal.
    Example:
        # To do a reachability check from the current robot pose.       # To move to a current robot pose with a few options changed.
        builder = MoveItGoalBuilder()                                   builder = MoveItGoalBuilder()
        builder.set_pose_goal(pose_stamped)                             builder.set_pose_goal(pose_stamped)
        builder.allowed_planning_time = 5                               builder.replan = True
        builder.plan_only = True                                        builder.replan_attempts = 10
        goal = builder.build()                                          goal = builder.build()
    Here are the most common class attributes you might set before calling build(), and their default values:
    allowed_planning_time: float=10. How much time to allow the planner, in seconds. group_name: string='arm'. Either 'arm' or 'arm_with_torso'.
    num_planning_attempts: int=1. How many times to compute the same plan (most planners are randomized). The shortest plan will be used.
    plan_only: bool=False. Whether to only compute the plan but not execute it.
    replan: bool=False. Whether to come up with a new plan if there was an error executing the original plan.
    replan_attempts: int=5. How many times to do replanning. replay_delay: float=1. How long to wait in between replanning, in seconds. 
    tolerance: float=0.01. The tolerance, in meters for the goal pose.
    '''

    def __init__(self):
        self.allowed_planning_time, self.fixed_frame, self.gripper_frame, self.group_name = 10.0, 'base_link', 'gripper_link', 'arm'
        self.planning_scene_diff, self.planning_scene_diff.is_diff, self.plan_only = moveit_msgs.msg.PlanningScene(), True, False
        self.planning_scene_diff.robot_state.is_diff, self.replan, self.look_around, self.max_acceleration_scaling_factor = True, False, False, 0
        self.max_velocity_scaling_factor, self.num_planning_attempts, self.planner_id, self._tf_listener = 0, 1, 'RRTConnectkConfigDefault', tf.TransformListener()
        self.replan_attempts, self.replan_delay, self.tolerance, self.start_state = 5, 1, 0.01, moveit_msgs.msg.RobotState()
        self.start_state.is_diff, self._orientation_constraints, self._pose_goal, self._joint_names, self._joint_positions = True, [], None, None, None

    def set_pose_goal(self, pose_stamped):
        '''
        Sets a pose goal. Pose and joint goals are mutually exclusive. The most recently set goal wins. pose_stamped: A geometry_msgs/PoseStamped.
        '''
        self._pose_goal, self._joint_names, self._joint_positions = pose_stamped, None, None
        
    def set_joint_goal(self, joint_names, joint_positions):
        '''
        Set a joint-space planning goal. joint_names: A list of strings. The names of the joints in the goal. joint_positions: A list of floats. 
        The joint angles to achieve.
        '''
        self._joint_names, self._joint_positions, self._pose_goal = joint_names, joint_positions, None

    def add_path_orientation_constraint(self, o_constraint):
        '''
        Adds an orientation constraint to the path. o_constraint: A moveit_msgs/OrientationConstraint.
        '''
        self._orientation_constraints.append(copy.deepcopy(o_constraint))
        self.planner_id = 'RRTConnectkConfigDefault'

    def build(self, tf_timeout=rospy.Duration(5.0)):
        '''
        Builds the goal message. To set a pose or joint goal, call set_pose_goal or set_joint_goal before calling build. To add a path orientation constraint, call
        add_path_orientation_constraint first. tf_timeout: rospy.Duration. How long to wait for a TF message. Only used with pose id. Returns: moveit_msgs/MoveGroupGoal
        '''
        goal = MoveGroupGoal()
        goal.request.start_state = copy.deepcopy(self.start_state)  # Set start state

        if self._pose_goal is not None:     # Set goal constraint
            self._tf_listener.waitForTransform(self._pose_goal.header.frame_id, self.fixed_frame, rospy.Time.now(), tf_timeout)
            try:
                pose_transformed = self._tf_listener.transformPose(self.fixed_frame, self._pose_goal)
            except (tf.LookupException, tf.ConnectivityException):
                return None
            c1 = Constraints()
            c1.position_constraints.append(PositionConstraint())
            c1.position_constraints[0].header.frame_id = self.fixed_frame
            c1.position_constraints[0].link_name = self.gripper_frame
            b, s = BoundingVolume(), SolidPrimitive()
            s.dimensions, s.type = [self.tolerance * self.tolerance], s.SPHERE
            b.primitives.append(s)
            b.primitive_poses.append(self._pose_goal.pose)
            c1.position_constraints[0].constraint_region, c1.position_constraints[0].weight = b, 1.0
            c1.orientation_constraints.append(OrientationConstraint())
            c1.orientation_constraints[0].header.frame_id, c1.orientation_constraints[0].orientation = self.fixed_frame, pose_transformed.pose.orientation
            c1.orientation_constraints[0].link_name, c1.orientation_constraints[0].absolute_x_axis_tolerance = self.gripper_frame, self.tolerance
            c1.orientation_constraints[0].absolute_y_axis_tolerance, c1.orientation_constraints[0].absolute_z_axis_tolerance = self.tolerance, self.tolerance
            c1.orientation_constraints[0].weight = 1.0
            goal.request.goal_constraints.append(c1)
        elif self._joint_names is not None:
            c1 = Constraints()
            for i in range(len(self._joint_names)):
                c1.joint_constraints.append(JointConstraint())
                c1.joint_constraints[i].joint_name, c1.joint_constraints[i].position = self._joint_names[i], self._joint_positions[i]
                c1.joint_constraints[i].tolerance_above, c1.joint_constraints[i].tolerance_below = self.tolerance, self.tolerance
                c1.joint_constraints[i].weight = 1.0
            goal.request.goal_constraints.append(c1)

        # Set path constraints
        goal.request.path_constraints.orientation_constraints = self._orientation_constraints
        # Set trajectory constraints
        goal.request.planner_id= self.planner_id    # Set planner ID (name of motion planner to use)
        goal.request.group_name = self.group_name   # Set group name
        goal.request.num_planning_attempts = self.num_planning_attempts # Set planning attempts
        goal.request.allowed_planning_time = self.allowed_planning_time # Set planning time
        # Set scaling factors
        goal.request.max_acceleration_scaling_factor = self.max_acceleration_scaling_factor
        goal.request.max_velocity_scaling_factor = self.max_velocity_scaling_factor
        goal.planning_options.planning_scene_diff = copy.deepcopy(self.planning_scene_diff) # Set planning scene diff
        goal.planning_options.plan_only = self.plan_only    # Set is plan only
        goal.planning_options.look_around = self.look_around    # Set look around
        # Set replanning options
        goal.planning_options.replan, goal.planning_options.replan_attempts = self.replan, self.replan_attempts
        goal.planning_options.replan_delay = self.replan_delay
        return goal

################################################################################
################################################################################
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
################################################################################
################################################################################

class fetchRobotClassObj(object):
    '''
    This class will incorporate all the basic movements that the fetch can do.
    This includes what the fetch arm needs to do, like auto-tuck, move to home 
    position, etc.
    '''
    def __init__(self):
        self.nodeName = 'fetch_all_functions' 
        rospy.init_node(self.nodeName)    # Initializing node.
        self.namespace = rospy.get_namespace()
        # Appending namespace to node name.
        rospy.resolve_name(self.nodeName, caller_id=self.namespace)

        # This robotName helps to communicate with the correct robot.
        # The namespace is like '/fetch/' so to make the robotname just 'fetch' 
        # we are ignoring the first and last letters in the string.
        self.robotName = self.namespace[1:-1]
        
        # Defining some parameters that are used to map the pixel distance in the 
        # images into real world distances. These are obtained using linear regression.
        self.thetaD, self.thetaX, self.thetaY, self.theta = 0.056, -0.134, -0.003, 43.018

################################################################################

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
        
################################################################################

        # This is used for printing the current position of the arm.
        self.listenerGripperPose = tf.TransformListener()

        # Create move group interface for a fetch robot
        self.armMoveGroup = MoveGroupInterface('arm_with_torso', 'base_link')

        # Arm joint names.
        self.armJointNames = ['torso_lift_joint', 'shoulder_pan_joint',
                               'shoulder_lift_joint', 'upperarm_roll_joint',
                               'elbow_flex_joint', 'forearm_roll_joint',
                               'wrist_flex_joint', 'wrist_roll_joint']
        
        # forearm_roll_joint, upperarm_roll_joint and wrist_roll_joint values 
        # cycles from -6.25 to 0 to 6.25. I.e. the position of the joint for 
        # -6.25, 0 and 6.25 and their multiples are the same. 
        # The sign just shows in which direction the joint will rotate. For 
        # orienting thes joints around 0 degree or the 180 degree zone, its 
        # value should be set near 0 or near 3.12 respectively. 
        
        # The wrist_flex_joint and elbow_flex_joint can go to +/-125 degrees. 
        # And the value goes from 0 to +/-2.5. At a value of 1.55 they are oriented 
        # around 90 degrees.
        
        # A value of 0.4 to the torso_lift_joint raises it to 42 cm.
        # And value of 0 to the torso_lift_joint raises it to 0 cm.
        # So a value of 0.1 will set the torso_lift_joint to a height of 10 cm (approx).
        
        # The shoulder_pan_joint is oriented straight ahead when the value is 0.
        # It will orient at 90 deg to the right (by rotating clockwise) with a 
        # max -ve value of -1.6, and it will orient at 90 deg to the left (by 
        # rotating counter-clockwise) with a max +ve value of 1.6.
        
        self.refFrame = 'base_link'

        # Lists of joint angles in the same order as in joint_names.
        self.armHomePose = [0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Lists of joint angles in the same order as in joint_names.
        self.armAutoTuckPoseList = [[0.15, 0.0, 1.5, 0.0, -1.65, -3.05, 1.6, 1.5],
                                     [0.0, 1.5, 1.5, 3.0, -1.65, -3.05, 1.6, 1.5]]
        
        # The safelyUntuckPoseList is basically the reverse list of autoTuckPoseList.
        self.armSafelyUntuckPoseList = self.armAutoTuckPoseList[::-1]
        
        # Position and orientation of two 'wave end poses'.
        self.waveArmPoseList = [Pose(Point(0.042, 0.384, 1.826),
                                Quaternion(0.173, -0.693, -0.242, 0.657)),
                                Pose(Point(0.047, 0.545, 1.822),
                                Quaternion(-0.274, -0.701, 0.173, 0.635))]
                                 
        # This is used for moving the arm without orientation constraints.
        self._arm_move_group_client = actionlib.SimpleActionClient(
            'move_group', MoveGroupAction)
        
################################################################################

        # This is the wrist link not the gripper itself.
        self.wristFrame = 'wrist_roll_link'
        self.gripperFrame = 'gripper_link'
        
        self.gripperJointNames = ['l_gripper_finger_joint', 'r_gripper_finger_joint']
        
        # If the gripper is completely closed then the values of the 
        # l_gripper_finger_joint and the r_gripper_finger_joint are in the order 
        # of 0.00021 or 0.00023 (definitely less than 0.0005).
        # If the gripper is open then the values are of the order 0.04 or 0.05.
        # If the blue bin is gripped from the side, the values are of the order
        # 0.00098 or 0.00096.

        # This is used for opening and closing the gripper.
        self._gripper_client = actionlib.SimpleActionClient(
            self.namespace + 'gripper_controller/gripper_action', GripperCommandAction)
        self._gripper_client.wait_for_server(rospy.Duration(10))

################################################################################

        self.headJointNames = ['head_pan_joint', 'head_tilt_joint']

        self._head_trajectory_client = actionlib.SimpleActionClient(
            self.namespace + 'head_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        self._head_point_client = actionlib.SimpleActionClient(
            self.namespace + 'head_controller/point_head', PointHeadAction)

        while not self._head_trajectory_client.wait_for_server(
                timeout=rospy.Duration(1)) and not rospy.is_shutdown():
            rospy.logwarn('Waiting for head trajectory server...')
        while not self._head_point_client.wait_for_server(
                timeout=rospy.Duration(1)) and not rospy.is_shutdown():
            rospy.logwarn('Waiting for head pointing server...')

################################################################################

        rospy.Subscriber(self.namespace + 'ar_pose_marker', AlvarMarkers, 
                                         callback=self._arTag_callback)

        self.arTagsPoseList = []   # List to hold the AR tag markers.

################################################################################

        # Name of all the joints including arm, head, wheels and gripper.
        self.allJointNames = self.armJointNames + self.headJointNames + \
                             self.gripperJointNames + ['r_wheel_joint', 
                                                  'l_wheel_joint', 'bellows_joint']

################################################################################

        # Subscribing to the joint_state topic to get the current joint poses.
        rospy.Subscriber(self.namespace + 'joint_states', JointState, 
                                                     self._joint_state_callback)
        
        # Dictionaries that holds the current joint states. Initializing it to all 0s.
        # These will be updated in the callback function.
        self.currentArmJointStates = {jointName: 0.0 for jointName in self.armJointNames}
        self.currentAllJointStates = {jointName: 0.0 for jointName in self.allJointNames}

        # Pose object that holds the current pose of the gripper. Initializing to all 0s.
        # These will be updated in the callback function.
        self.currentGripperPose = PoseStamped()
        self.currentGripperPose.header.frame_id = self.refFrame
        self.currentGripperPose.pose.position = Point(0.0, 0.0, 0.0)
        self.currentGripperPose.pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
        
################################################################################

        # Subscribing to the battery_state topic to get the current battery voltage.
        rospy.Subscriber(self.namespace + 'battery_state', BatteryState, 
                                                     self._battery_state_callback)

        self.currentBattChargeLvl = 0.0    # Initializing battery charge level variable.

################################################################################

        # Subscribing to the image topics to get the current battery voltage.
        
        # uint16 depth image in mm, published at VGA resolution (640x480) at 15Hz.
        rospy.Subscriber(self.namespace + 'head_camera/depth/image_raw', Image, 
                                                     self._depth_image_callback)
        
        # 2d color data, published at VGA resolution (640x480) at 15Hz.
        rospy.Subscriber(self.namespace + 'head_camera/rgb/image_raw', Image, 
                                                     self._color_image_callback)
        
        self.colorBridge = CvBridge()    # Converts sensor_msgs.msg.Image into cv2 image.
        self.depthBridge = CvBridge()
        self.depthImg, self.depthImgColorMap, self.colorImg = None, None, None
        self.imgH, self.imgW = 480, 640
        
################################################################################

        # Subscribing to the head camera intrinsic parameters.
        
        self.cameraOpticalRgbFrame = 'head_camera_rgb_optical_frame'
        self.cameraRgbFrame = 'head_camera_rgb_frame'
        self.cameraOpticalDepthFrame = 'head_camera_depth_optical_frame'
        self.cameraDepthFrame = 'head_camera_depth_frame'
        
        # Used to get the camera parameters.
        self.listenerCameraPose = tf.TransformListener()

        rospy.Subscriber(self.namespace + 'head_camera/rgb/camera_info', CameraInfo, 
                                                     self._camera_info_callback)
        
        # These will hold the camera intrinsic parameters.
        self.FX, self.CX, self.FY, self.CY = 0, 0, 0, 0
        self.Kmatrix = np.zeros((3,3))

        # These will hold the camera extrinsic parameters.
        # Pose object that holds the current pose of the camera. Initializing to all 0s.
        # These will be updated in the callback function.
        self.currentCameraPose = PoseStamped()
        self.currentCameraPose.header.frame_id = self.refFrame
        self.currentCameraPose.pose.position = Point(0.0, 0.0, 0.0)
        self.currentCameraPose.pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
        self.camToBaseLinkTrMatrix = np.zeros((3,4))

        # Details of the camera field of views (in degrees).
        self.hFov, self.vFov = 54, 45

        # Create a publisher which will publish the array of 3D points corresponding 
        # to all the points inside the bounding box of the identified object.
        self._arr_of_3D_points_pub = rospy.Publisher(self.namespace + 'visualization_marker', 
                                                Marker, queue_size=5)
        
        # Create a folder to save the images from the onboard camera.
        self.imgSaveFolder = './saved_images'
        if not os.path.exists(self.imgSaveFolder):   os.mkdirs(self.imgSaveFolder)

################################################################################

        # Create a publisher which will publish the marker in the point cloud 
        # domain for an object to be picked up.
        self._obj_marker_pub = rospy.Publisher(self.namespace + 'visualization_marker', 
                                                Marker, queue_size=5)
        
################################################################################

        # By just calling this wait_for_message function, the Subscribers are 
        # called once and hence the callback functions are run once, which updates
        # the 'self.' variables which the callback functions are supposed to update, 
        # otherwise they will always start with the initialized values.
        msg = rospy.wait_for_message(self.namespace + 'joint_states', JointState)
        msg = rospy.wait_for_message(self.namespace + 'battery_state', BatteryState)
        msg = rospy.wait_for_message(self.namespace + 'odom', Odometry)
        msg = rospy.wait_for_message(self.namespace + 'ar_pose_marker', AlvarMarkers)
        msg = rospy.wait_for_message(self.namespace + 'head_camera/depth/image_raw', Image)
        msg = rospy.wait_for_message(self.namespace + 'head_camera/rgb/image_raw', Image)
        msg = rospy.wait_for_message(self.namespace + 'head_camera/rgb/camera_info', CameraInfo)

################################################################################        
################################################################################

    def _odom_callback(self, msg):
        '''
        Callback function of the Subscriber. Updates the x, y coordinates of the robot.
        '''
        self.odom = msg.pose.pose

################################################################################
################################################################################

    def _depth_image_callback(self, msg):
        '''
        Callback function of the Subscriber. Updates the depth images from camera.
        '''
        # Try to convert the ROS Image message to a CV2 Image
        try:
            img = self.depthBridge.imgmsg_to_cv2(msg, 'passthrough')
            
            # Clipping all values beyond 2.5m as the its too far for any analysis.
            img = np.clip(img, 0, 2500)
            
            # Converting pixel values to cm so that 2500mm can become 
            # 250cm and thereby represented as 8 bit int. Grayscale depth image.
            img = np.asarray(img / 10.0, dtype=np.uint8)
            self.depthImg = copy.deepcopy(img)
            
            # Preparing this frame for recording depth video, 
            # so it has to be of 3 channel.
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            self.depthImgColorMap = cv2.applyColorMap(img, cv2.COLORMAP_JET)

        except CvBridgeError, e:
            rospy.logerr('CvBridge Error: {0}'.format(e))

################################################################################
################################################################################

    def _color_image_callback(self, msg):
        '''
        Callback function of the Subscriber. Updates the rgb images from camera.
        '''
        # Try to convert the ROS Image message to a CV2 Image
        
        try:
            img = self.colorBridge.imgmsg_to_cv2(msg, 'passthrough')
            self.colorImg = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        except CvBridgeError, e:
            rospy.logerr('CvBridge Error: {0}'.format(e))

################################################################################
################################################################################

    def _joint_state_callback(self, msg):
        '''
        Callback function that saves the joint positions in the currentArmJointStates
        and currentAllJointStates variables when a joint_states message is received.
        This function also updates the current position and orientation of the 
        gripper as well in the currentGripperPose variable.
        '''
        for i, jointName in enumerate(msg.name):
            self.currentAllJointStates[jointName] = msg.position[i]
            if jointName in self.armJointNames:
                self.currentArmJointStates[jointName] = msg.position[i]
        
        currentFrame = self.gripperFrame
        targetFrame = self.refFrame
        try:
            time = self.listenerGripperPose.getLatestCommonTime(targetFrame, currentFrame)
            self.listenerGripperPose.waitForTransform(targetFrame, currentFrame, rospy.Time(), 
                                                        rospy.Duration(3.0))
            (position, orientation) = self.listenerGripperPose.lookupTransform(targetFrame, 
                                                                   currentFrame, time)
            self.currentGripperPose.pose.position = Point(position[0], position[1], 
                                                           position[2])
            self.currentGripperPose.pose.orientation = Quaternion(orientation[0], 
                                                          orientation[1], orientation[2], 
                                                          orientation[3])
        
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn('Something wrong with transform request: ' + str(e))
            pass

################################################################################
################################################################################

    def _camera_info_callback(self, msg):
        '''
        Callback function that saves the head camera intrinsic parameters.
        It also listens to the transform between the refFrame and the camera and 
        updates the extrinsic parameters.
        '''
        
        # Updating the intrinsic parameters.
        self.FX, self.CX, self.FY, self.CY = msg.K[0], msg.K[2], msg.K[4], msg.K[5]
        self.Kmatrix = np.array([[self.FX, 0.0, self.CX],
                                 [0.0, self.FY, self.CY],
                                 [0.0, 0.0,     1.0]])
        
        ## Updating the extrinsic parameters.
        currentFrame = self.cameraRgbFrame
        targetFrame = self.refFrame
        #try:
            #time = self.listenerGripperPose.getLatestCommonTime(targetFrame, currentFrame)
            #self.listenerGripperPose.waitForTransform(targetFrame, currentFrame, rospy.Time(), 
                                                        #rospy.Duration(3.0))
            #(position, orientation) = self.listenerGripperPose.lookupTransform(targetFrame, 
                                                                   #currentFrame, time)
            #self.currentCameraPose.pose.position = Point(position[0], position[1], 
                                                           #position[2])
            #self.currentCameraPose.pose.orientation = Quaternion(orientation[0], 
                                                          #orientation[1], orientation[2], 
                                                          #orientation[3])
        
        # Updating the extrinsic parameters.
        try:
            time = self.listenerGripperPose.getLatestCommonTime(targetFrame, currentFrame)
            self.listenerGripperPose.waitForTransform(targetFrame, currentFrame, rospy.Time(), 
                                                        rospy.Duration(3.0))
            (position, orientation) = self.listenerGripperPose.lookupTransform(targetFrame, 
                                                                   currentFrame, time)
            self.currentCameraPose.pose.position = Point(position[0], position[1], 
                                                           position[2])
            self.currentCameraPose.pose.orientation = Quaternion(orientation[0], 
                                                          orientation[1], orientation[2], 
                                                          orientation[3])

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn('Something wrong with transform request: ' + str(e))
            pass
        
        # Converting the values from position and quaternion form into a 4x4 matrix.
        transformerRos = tf.TransformerROS()
        self.camToBaseLinkTrMatrix = transformerRos.fromTranslationRotation(position, orientation)
        
        # Extrinsic parameter matrix is a 3x4 matrix. So ignoring the last 0,0,0,1 row.
        self.camToBaseLinkTrMatrix = np.array(self.camToBaseLinkTrMatrix[:-1])

################################################################################
################################################################################

    def _battery_state_callback(self, msg):
        '''
        Callback function that saves the robot battery voltage in currentBattState.
        '''
        self.currentBattChargeLvl = msg.charge_level
        
################################################################################
################################################################################

    def _linear_distance(self, pose1, pose2):
        '''
        Calculates the linear distance between two poses.
        '''
        pos1, pos2 = pose1.position, pose2.position
        dx, dy, dz = pos1.x - pos2.x, pos1.y - pos2.y, pos1.z - pos2.z
        return np.sqrt(dx * dx + dy * dy + dz * dz)

################################################################################
################################################################################

    def _x_axis_from_quaternion(self, q):
        '''
        Calculates x value from Quaternion.
        '''
        m = tf.transformations.quaternion_matrix([q.x, q.y, q.z, q.w])
        return m[:3, 0]  # First column

################################################################################
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
################################################################################

    def _arTag_callback(self, msg):
        '''
        Callback function for AR tag reader.
        It stores the poses of the AR tags (if found) in the list arTagsPoseList.
        '''
        poseList = []
        if len(msg.markers) != 0:
            for marker in msg.markers:
                arTagPose = PoseStamped()
                arTagPose = marker.pose
                poseList.append(arTagPose)

        self.arTagsPoseList = copy.deepcopy(poseList)

################################################################################
################################################################################

    def _arm_move_to_pose(self, pose_stamped, allowed_planning_time=10.0, 
                     execution_timeout=15.0, num_planning_attempts=3, 
                     orientation_constraint=None, plan_only=False, replan=False, 
                     replan_attempts=5, tolerance=0.01):
        '''
        This function is obtained from the following link:
        https://github.com/subodh-malgonde/Robotics/blob/indigo-devel/fetch_api/src/fetch_api/arm.py
        
        Moves the end-effector to a pose, using motion planning.
        pose: geometry_msgs/PoseStamped. The goal pose for the gripper.
        allowed_planning_time: float. The maximum duration to wait for a
        planning result.
        execution_timeout: float. The maximum duration to wait for an arm
        motion to execute (or for planning to fail completely), in seconds.
        num_planning_attempts: int. The number of times to compute the same
        plan. The shortest path is ultimately used. For random
        planners, this can help get shorter, less weird paths.
        orientation_constraint: moveit_msgs/OrientationConstraint. An
        orientation constraint for the entire path.
        plan_only: bool. If True, then this method does not execute the
        plan on the robot. Useful for determining whether this is likely to succeed.
        replan: bool. If True, then if an execution fails (while the arm is
        moving), then come up with a new plan and execute it.
        replan_attempts: int. How many times to replan if the execution fails.
        tolerance: float. The goal tolerance, in meters.
        '''
        goal_builder = MoveItGoalBuilder()
        goal_builder.set_pose_goal(pose_stamped)
        if orientation_constraint is not None:
            goal_builder.add_path_orientation_contraint(orientation_constraint)
        goal_builder.allowed_planning_time = allowed_planning_time
        goal_builder.num_planning_attempts = num_planning_attempts
        goal_builder.plan_only = plan_only
        goal_builder.replan = replan
        goal_builder.replan_attempts = replan_attempts
        goal_builder.tolerance = tolerance
        goal = goal_builder.build()
        if goal is not None:
            self._arm_move_group_client.send_goal(goal)
            self._arm_move_group_client.wait_for_result(rospy.Duration(execution_timeout))
            result = self._arm_move_group_client.get_result()

        if result:
            if result.error_code.val == MoveItErrorCodes.SUCCESS:
                print('Goal can be reached...\n')
                return True
            else:
                print('error1')
                return False
        else:
            print('error2')
            return False

################################################################################
################################################################################

    def pixel3Dcoordinate(self, x, y):
        '''
        This function takes in the x and y coordinates of a point in the image 
        seen by the head camera rgb frame, and transforms it into the x, y, z 
        3D coordinates wrt the base_link reference frame using the camera parameters.
        '''
        point3D = np.matmul(self.invKRTmatrix, [[x],[y],[1]])
        point3D = point3D / point3D[3]      # Normalizing the point.
        point3D = point3D[:-1]      # Removing the last element which is 1.
        point3D = point3D.reshape(3)
        return point3D

################################################################################
################################################################################

    def pixel3DcoordinateUsingFOV(self, x, y, winW=50, winH=50):
        '''
        This function takes in the x and y coordinates of a point in the image 
        seen by the head camera rgb frame, and transforms it into the x, y, z 
        3D coordinates wrt the base_link reference frame using the field of view 
        of the camera.
        '''
        # Since the inputs will signify pixel coordinates here, so we have to 
        # make sure they dont go out of the bounds of the image. This may happen 
        # the bounding box in the image is partly outside the image boundaries.
        x1, y1 = x, y
        x1, y1 = min(x1, self.imgW), min(y1, self.imgH)
        x1, y1 = max(x1, 0), max(y1, 0)
        
        # Sometimes if we take only the depth value at a certain point in the 
        # depth image, it may happen that the depth at that point is read as 0, 
        # because there are no reflected ir light getting back to the camera 
        # from there. So, instead we are taking a winW x winH (by default 50x50) 
        # rectangle around the point and drawing the histogram of the depths in 
        # all the points of the rectangle. The bin with the highest number of 
        # pixels with non-zero depths is the actual depth of the point.
        
        # Since the inputs will signify pixel coordinates here, so we have to 
        # make sure they dont go out of the bounds of the image. This may happen 
        # the 50x50 rectangle is partly outside the image boundaries.
        h, w, roiDlw, roiDup = winH, winW, 5, 250
        x2, y2, x3, y3 = x1-int(w/2), y1-int(h/2), x1+int(w/2), y1+int(h/2)
        x2, y2 = min(x2, self.imgW), min(y2, self.imgH)
        x2, y2 = max(x2, 0), max(y2, 0)
        x3, y3 = min(x3, self.imgW), min(y3, self.imgH)
        x3, y3 = max(x3, 0), max(y3, 0)
        
        boxImg = depthImg[y2:y3, x2:x3]
        # Creating a histogram of the depth pixel values from the depth image in 
        # the region surrounding the point x1, y1.
        boxPixelList = boxImg.reshape(boxImg.shape[0]*boxImg.shape[1]).tolist()
        histVals, histBins = np.histogram(boxPixelList, bins=list(range(roiDlw, roiDup)))
        #plt.hist(boxImg.ravel(), (roiDup-roiDlw), [roiDlw,roiDup])
        #plt.show()
        #print(histVals, histBins)
        
        # Sorting the histogram value in descending order and the bins are also 
        # sorted accordingly.
        histValsSorted, histBinsSorted = zip(*sorted(zip(histVals, histBins), \
                                                key=lambda x: x[0], reverse=True))
        # Taking the bin having the max number of pixels as the approximate 
        # distance of the bin from the camera.
        d = histBinsSorted[0]
        
        # Making the coordinates relative to the center of the image.
        normalizedX, normalizedY = x1 - self.imgW * 0.5, y1 - self.imgH * 0.5
        d = d*0.01      # Converting the depth to meters.
        
        # Now finding the angle at which the pixel is wrt the field of view.
        # This angle is always made positive, as the direction of the final 
        # world point will be made by judging the sign of normalizedX and normalizedY.
        hAngle = np.fabs(normalizedX * self.hFov / self.imgW)
        vAngle = np.fabs(normalizedY * self.vFov / self.imgH)
        
        right = d * np.sin(np.deg2rad(hAngle))
        right = right * (-2 * (normalizedX < 0) + 1)
        down = d * np.sin(np.deg2rad(vAngle))
        down = down * (-2 * (normalizedY < 0) + 1)
        front = d * np.cos(np.deg2rad(hAngle)) * np.cos(np.deg2rad(vAngle))
                
        # Shifting to frame relative to camera (right handed coordinates).
        X, Y, Z = front, -1*right, -1*down
        
        point3D = np.matmul(self.camToBaseLinkTrMatrix, np.array([[X],[Y],[Z],[1]]))
        point3D = point3D.reshape(3)
        
        return point3D

################################################################################
################################################################################

    def robustDepthWithHist(self, x, y, winW=50, winH=50):
        '''
        Sometimes if we take only the depth value at a certain point in the 
        depth image, it may happen that the depth at that point is read as 0, 
        because there are no reflected ir light getting back to the camera 
        from there. So, instead we are taking a winW x winH (by default 50x50) 
        rectangle around the point and drawing the histogram of the depths in 
        all the points of the rectangle. The bin with the highest number of 
        pixels with non-zero depths is the actual depth of the point.
        This function takes in the x and y coordinates of a point in the image 
        seen by the head camera rgb frame, and then finds the depth at that 
        location using a histogram.
        '''
        # Since the inputs will signify pixel coordinates here, so we have to 
        # make sure they dont go out of the bounds of the image. This may happen 
        # the bounding box in the image is partly outside the image boundaries.
        x1, y1 = x, y
        x1, y1 = min(x1, self.imgW), min(y1, self.imgH)
        x1, y1 = max(x1, 0), max(y1, 0)
                
        # Since the inputs will signify pixel coordinates here, so we have to 
        # make sure they dont go out of the bounds of the image. This may happen 
        # the 50x50 rectangle is partly outside the image boundaries.
        h, w, roiDlw, roiDup = winH, winW, 5, 250
        x2, y2, x3, y3 = x1-int(w/2), y1-int(h/2), x1+int(w/2), y1+int(h/2)
        x2, y2 = min(x2, self.imgW), min(y2, self.imgH)
        x2, y2 = max(x2, 0), max(y2, 0)
        x3, y3 = min(x3, self.imgW), min(y3, self.imgH)
        x3, y3 = max(x3, 0), max(y3, 0)
        
        boxImg = depthImg[y2:y3, x2:x3]
        # Creating a histogram of the depth pixel values from the depth image in 
        # the region surrounding the point x1, y1.
        boxPixelList = boxImg.reshape(boxImg.shape[0]*boxImg.shape[1]).tolist()
        histVals, histBins = np.histogram(boxPixelList, bins=list(range(roiDlw, roiDup)))
        #plt.hist(boxImg.ravel(), (roiDup-roiDlw), [roiDlw,roiDup])
        #plt.show()
        #print(histVals, histBins)
        
        # Sorting the histogram value in descending order and the bins are also 
        # sorted accordingly.
        histValsSorted, histBinsSorted = zip(*sorted(zip(histVals, histBins), \
                                                key=lambda x: x[0], reverse=True))
        # Taking the bin having the max number of pixels as the approximate 
        # distance of the bin from the camera.
        d = histBinsSorted[0]
        
        return d

################################################################################
################################################################################

    def getch(self):
        '''
        This function is used to detect any keyboard key press event, in case we
        have to use it for breaking a while loop or something.
        It returns the character that is pressed. This while loop will not 
        break by simple ctrl+c, it has to have a break statement. Use the 'esc' 
        #like the following:
        while True:
            char = getch()
            print(char)
            if ord(char) & 0xFF == 27:   break
        '''
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

################################################################################
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

    def armFollowTrajectory(self, gripperPoseList, verbose=True):
        '''
        This function takes in a list of poses which contains a set of positions 
        and orientations. These are the poses through which the gripper moves 
        and follows a trajectory.
        '''
        # Construct a 'pose_stamped' message as required by moveToPose.
        gripperPoseStamped = PoseStamped()
        gripperPoseStamped.header.frame_id = self.refFrame

        for pose in gripperPoseList:
            if rospy.is_shutdown():     break

            # Finish building the Pose_stamped message.
            # If the message stamp is not current it could be ignored.
            gripperPoseStamped.header.stamp = rospy.Time.now()
            
            # Set the message pose.
            gripperPoseStamped.pose = pose

            # Move gripper frame to the pose specified.
            self.armMoveGroup.moveToPose(gripperPoseStamped, self.gripperFrame)

        # This stops all arm movement goals.
        # It should be called when a program is exiting so movement stops.
        self.armMoveGroup.get_move_action().cancel_all_goals()
        if verbose:     print('Arm followed trajectory...\n')

################################################################################
################################################################################

    def armFollowTrajectoryWithoutOri(self, gripperPoseListWithoutOri, verbose=True):
        '''
        This function takes in a list of poses which contains a set of positions 
        and orientations. These are the poses through which the gripper moves 
        and follows a trajectory. gripperPoseListWithoutOri is a list of Point objects.
        '''
        # Construct a 'pose_stamped' message as required by moveToPose.
        gripperPoseStamped = PoseStamped()
        gripperPoseStamped.header.frame_id = self.refFrame

        for position in gripperPoseListWithoutOri:
            if rospy.is_shutdown():     break

            # Finish building the Pose_stamped message.
            # If the message stamp is not current it could be ignored.
            gripperPoseStamped.header.stamp = rospy.Time.now()
            
            gripperPoseStamped.pose.position = position

            # Move gripper frame to the pose specified.
            self._arm_move_to_pose(gripperPoseStamped)

        # This stops all arm movement goals.
        # It should be called when a program is exiting so movement stops.
        self.armMoveGroup.get_move_action().cancel_all_goals()
        if verbose:     print('Arm followed trajectory without orientation...\n')

################################################################################
################################################################################

    def waveArm(self, howLongToWave=4, verbose=True):
        '''
        This function makes the fetch wave its arm.
        '''
        # First go to the home pose.
        self.armGotoHomePose()
        
        # Keep waving till counter expires.
        i = 0
        while i < howLongToWave:
            if rospy.is_shutdown():     break
            self.armFollowTrajectory(self.waveArmPoseList)
            i += 1
        if verbose:     print('Waved arm...\n')
    
################################################################################
################################################################################

    def moveSingleJoint(self, jointName, jointState=0.0, verbose=True):
        '''
        This function changes the state of a single arm joint keeping the 
        others the same.
        '''
        if jointName not in self.armJointNames:
            print('\nWrong joint name provided...')
            return
        
        for i in range(5):
            # If joints are already in right positions then break. Else try 5 times 
            # to put them in the right positions. It has to be run after running 
            # the moveMultipleJoints function as this function does not always 
            # moves the joints properly by mistake.
            if self.areJointsInRightPoitions({jointName: jointState}, verbose=False):
                break

            # Just updating the callback function.
            msg = rospy.wait_for_message(self.namespace + 'joint_states', JointState)
        
            requiredPose = []
            for currentJointName in self.armJointNames:
                # The joint names are called from the self.armJointNames list and 
                # not directly from the self.currentAllJointStates dictionary, because
                # the sequence in which the joints are listed in the self.armJointNames
                # list is important. The required pose to be given to the armMoveGroup
                # should be in this sequence itself.
                
                # Get the current state of all the arm joints.
                currentJointState = self.currentArmJointStates[currentJointName]
                
                # Calculating the new joint state of the desired joint.
                if currentJointName == jointName:     currentJointState = jointState
                requiredPose.append(currentJointState)
            
            self.armMoveGroup.moveToJointPosition(self.armJointNames, requiredPose, wait=True)
            
            # This stops all arm movement goals.
            # It should be called when a program is exiting so movement stops.
            self.armMoveGroup.get_move_action().cancel_all_goals()
            if verbose:     print('Joint moved...\n')
    
################################################################################
################################################################################

    def moveMultipleJoints(self, jointStateDict, verbose=True):
        '''
        This function changes the state of a multiple arm joints. Input jointStateDict
        is a dictionary where the keys are the joint names and the values are the
        desired joint states. If some joints are not mentioned in this dictionary
        then the state of that joint is kept unchanged.
        '''
        for k, v in jointStateDict.items():
            if k not in self.armJointNames:
                print('\nWrong joint name provided...')
                return

        for i in range(5):
            # If joints are already in right positions then break. Else try 5 times 
            # to put them in the right positions. It has to be run after running 
            # the moveMultipleJoints function as this function does not always 
            # moves the joints properly by mistake.
            if self.areJointsInRightPoitions(jointStateDict, verbose=False):
                break

            # Just updating the callback function.
            msg = rospy.wait_for_message(self.namespace + 'joint_states', JointState, timeout=2)

            requiredPose = []
            for currentJointName in self.armJointNames:
                # The joint names are called from the self.armJointNames list and 
                # not directly from the self.currentAllJointStates dictionary, because
                # the sequence in which the joints are listed in the self.armJointNames
                # list is important. The required pose to be given to the armMoveGroup
                # should be in this sequence itself.
                
                # Get the current state of all the arm joints.
                currentJointState = self.currentArmJointStates[currentJointName]
                
                # Calculating the new joint state of the desired joint.
                if currentJointName in jointStateDict:
                    currentJointState = jointStateDict[currentJointName]
                
                requiredPose.append(currentJointState)
            
            self.armMoveGroup.moveToJointPosition(self.armJointNames, requiredPose, wait=True)
            
            # This stops all arm movement goals.
            # It should be called when a program is exiting so movement stops.
            self.armMoveGroup.get_move_action().cancel_all_goals()
            if verbose:     print('Joint moved...\n')
    
################################################################################
################################################################################

    def armGotoHomePose(self, verbose=True):
        '''
        This function takes the fetch arm to the home pose, the arm is completely 
        straight in this pose.

        home pose: 
        position: x: 0.961639, y: -0.003010, z: 0.793820
        orientation: x: 0.003870, y: 0.000608, z: -0.001593, w: 0.999991
        '''
        # Plans the joints in joint_names to angles in pose.
        self.armMoveGroup.moveToJointPosition(self.armJointNames, self.armHomePose, wait=True)
        
        # This stops all arm movement goals.
        # It should be called when a program is exiting so movement stops.
        self.armMoveGroup.get_move_action().cancel_all_goals()
        if verbose:     print('Arm to home position...\n')

################################################################################
################################################################################

    def gripperToPose1(self, gripperTargetPose, verbose=True):
        '''
        This function takes the gripper of the fetch to a given target pose.
        But you have to know for sure that the position and orientation will 
        give proper inverse kinematics solutions.
        '''
        # Construct a 'pose_stamped' message as required by moveToPose.
        gripperPoseStamped = PoseStamped()
        gripperPoseStamped.header.frame_id = self.refFrame

        # If the message stamp is not current it could be ignored.
        gripperPoseStamped.header.stamp = rospy.Time.now()
        # Set the message pose.
        gripperPoseStamped.pose = gripperTargetPose

        # Move gripper frame to the pose specified.
        self.armMoveGroup.moveToPose(gripperPoseStamped, self.gripperFrame)
        
        # This stops all arm movement goals.
        # It should be called when a program is exiting so movement stops.
        self.armMoveGroup.get_move_action().cancel_all_goals()
        if verbose:     print('Gripper to pose...\n')

################################################################################
################################################################################

    def gripperToPose(self, gripperTargetPose, planOnly=False, verbose=True):
        '''
        This function takes the gripper of the fetch to a given target pose.
        But this function only works for the locations near the front of the 
        robot, it throws errors if you try to reach places near the top or near
        the head even in the orientation is correct.
        Providing the orientation is not compulsory, it can work without orientation
        values as well. In that case the input can simply be a Point object instead
        or a Pose object. But There will be some orientations where the arm will
        not be able to reach. But for the positions in the front of the robot 
        in general it will be able to move it simple with the position values.
        Sometimes there may be a need to just check if it is possible to reach 
        a particular pose or not. So in that case the planOnly flag should be 
        made true, it will only return a True or False, but will not move the arm.
        '''
        # Construct a 'pose_stamped' message as required by moveToPose.
        gripperPoseStamped = PoseStamped()
        gripperPoseStamped.header.frame_id = self.refFrame

        # If the message stamp is not current it could be ignored.
        gripperPoseStamped.header.stamp = rospy.Time.now()
        
        # Checking if the input is a Pose object or a Point object.
        if type(gripperTargetPose) == Pose:   # It has both position and orientation.
            gripperPoseStamped.pose = gripperTargetPose     # Set the pose.
        elif type(gripperTargetPose) == Point:    # It has only position.
            gripperPoseStamped.pose.position = gripperTargetPose    # Set the position.        

        # Move gripper frame to the pose specified.
        posePossible = self._arm_move_to_pose(gripperPoseStamped, plan_only=planOnly)

        # This stops all arm movement goals.
        # It should be called when a program is exiting so movement stops.
        self.armMoveGroup.get_move_action().cancel_all_goals()
        if verbose:     print('Gripper to pose...\n')
        
        return posePossible     # Boolean flag indicating if given pose is possible.
    
################################################################################
################################################################################

    def areJointsInRightPoitions(self, jointStateDict, verbose=True):
        '''
        This function checks if the joints in the arm are all in right position 
        or not. It has to be run after running the moveMultipleJoints function 
        as this function does not always moves the joints properly by mistake.
        '''
        for k, v in jointStateDict.items():
            if k not in self.armJointNames:
                print('\nWrong joint name provided...')
                return
        
        # Just updating the callback function.
        msg = rospy.wait_for_message(self.namespace + 'joint_states', JointState, timeout=2)
        
        inRightPositions = True
        for currentJointName in self.armJointNames:
            # The joint names are called from the self.armJointNames list and 
            # not directly from the self.currentAllJointStates dictionary, because
            # the sequence in which the joints are listed in the self.armJointNames
            # list is important. The required pose to be given to the armMoveGroup
            # should be in this sequence itself.
            
            # Get the current state of all the arm joints.
            currentJointState = self.currentArmJointStates[currentJointName]
            
            # Calculating the new joint state of the desired joint.
            if currentJointName in jointStateDict:
                targetJointState = jointStateDict[currentJointName]
            
                # Checking if the current joint is not in the target joint state.
                if abs(currentJointState - targetJointState) > 0.018:
                    inRightPositions = False
                    if verbose:     print('Arm is not in right position...\n')
                    break

        return inRightPositions
            
################################################################################
################################################################################

    def isArmTucked(self, verbose=True):
        '''
        This function checks if the arm is already in tucked position or not.
        '''
        tuckedStateList = self.armAutoTuckPoseList[-1]
        isTucked = True
        for j, currentJointName in enumerate(self.armJointNames):
            if rospy.is_shutdown():     break
        
            # The joint names are called from the self.armJointNames list and 
            # not directly from the self.currentAllJointStates dictionary, because
            # the sequence in which the joints are listed in the self.armJointNames
            # list is important. The required pose to be given to the armMoveGroup
            # should be in this sequence itself.
            # Arm joint names:
            # self.armJointNames = ['torso_lift_joint', 'shoulder_pan_joint',
            #     'shoulder_lift_joint', 'upperarm_roll_joint', 'elbow_flex_joint', 
            #     'forearm_roll_joint', 'wrist_flex_joint', 'wrist_roll_joint']
            
            # Ignore the torso_lift_joint.
            if currentJointName == 'torso_lift_joint':      continue

            # Get the current state of all the arm joints.
            currentJointState = self.currentArmJointStates[currentJointName]

            tuckedState = tuckedStateList[j]
            
            # If the current joint is not in the tucked state, then tuck it.
            if abs(currentJointState - tuckedState) > 0.018:
                isTucked = False
                if verbose:     print('Arm is not tucked...\n')
                break
                    
        return isTucked
            
################################################################################
################################################################################

    def autoTuckArm(self, verbose=True):
        '''
        This function tucks back the fetch arm automatically.
        '''
        if self.isArmTucked(verbose=False):
            self.moveSingleJoint('torso_lift_joint', 0.0, verbose=False)
            self.gripperClose(verbose=False)
            return
            
        for pose in self.armAutoTuckPoseList:
            if rospy.is_shutdown():     break

            # Plans the joints in joint_names to angles in pose.
            self.armMoveGroup.moveToJointPosition(self.armJointNames, pose, wait=False)

            # Since we passed in wait=False above we need to wait here.
            self.armMoveGroup.get_move_action().wait_for_result()
            
            self.gripperClose(verbose=False)     # Close gripper.

        # This stops all arm movement goals.
        # It should be called when a program is exiting so movement stops.
        self.armMoveGroup.get_move_action().cancel_all_goals()
        if verbose:     print('Arm tucked...\n')

################################################################################
################################################################################

    def safelyUntuckArm(self, verbose=True):
        '''
        This function untucks the arm safely to the home position from the tucked 
        position without any collision. It basically runs the autoTuckPoseList 
        in the opposite manner.
        '''
        for pose in self.armSafelyUntuckPoseList:
            if rospy.is_shutdown():     break

            # Plans the joints in joint_names to angles in pose.
            self.armMoveGroup.moveToJointPosition(self.armJointNames, pose, wait=False)

            # Since we passed in wait=False above we need to wait here.
            self.armMoveGroup.get_move_action().wait_for_result()

        # This stops all arm movement goals.
        # It should be called when a program is exiting so movement stops.
        self.armMoveGroup.get_move_action().cancel_all_goals()
        if verbose:     print('Arm safely untucked...\n')

################################################################################
################################################################################

    def gripperOpen(self, gap=0.10, verbose=True):
        '''
        Opens the gripper.
        '''
        goal = GripperCommandGoal()
        # The position for a fully-open gripper (meters) is 0.1.
        goal.command.position = gap
        self._gripper_client.send_goal_and_wait(goal, rospy.Duration(10))
        if verbose:     print('Opened gripper...\n')

################################################################################
################################################################################

    def gripperClose(self, gap=0.0, maxEffort=100, verbose=True):
        '''
        Closes the gripper.
        maxEffort: The maximum effort, in Newtons, to use. Note that this should 
        not be less than 35N, or else the gripper may not close.
        gap:    The gap between the fingers of the gripper.
        This gap is the gap between the metal pieces of the gripper excluding 
        the rubber padding.
        '''
        goal = GripperCommandGoal()
        # The position for a fully-closed gripper (meters) is 0.0.
        goal.command.position = gap
        goal.command.max_effort = maxEffort
        self._gripper_client.send_goal_and_wait(goal, rospy.Duration(10))
        if verbose:     print('Closed gripper...\n')

################################################################################
################################################################################

    def headLookAt(self, x, y, z, frame_id=None, verbose=True):
        '''
        Moves the head to look at a point in space.
        frame_id: The name of the frame in which x, y, and z are specified.
        x: The x value of the point to look at.
        y: The y value of the point to look at.
        z: The z value of the point to look at.
        '''
        goal = PointHeadGoal()
        if frame_id is None:    frame_id = self.refFrame
        goal.target.header.frame_id = frame_id
        goal.target.point.x = x
        goal.target.point.y = y
        goal.target.point.z = z

        goal.min_duration = rospy.Duration(2.5)
        self._head_point_client.send_goal(goal)
        self._head_point_client.wait_for_result()
        if verbose:     print('Head in position...\n')

################################################################################
################################################################################

    def headPanHoriTilt(self, pan, tilt, verbose=True):
        '''
        Moves the head by setting pan/tilt angles.
        Input angles are in degrees.
        pan: The pan angle, in radians. A positive value is clockwise.
        tilt: The tilt angle, in radians. A positive value is downwards.
        These angles are absolute values of the angles relative to the base_link
        frame, i.e. is pan, tilt = 0, 0 means the head will look straight ahead.
        '''
        # Converting the pan and tilt angles into radians.
        pan, tilt = np.deg2rad(pan), np.deg2rad(tilt)
        # Making sure the pan in always within -pi/2 and pi/2.
        pan = min(max(pan, -np.pi/2), np.pi/2)
        # Making sure the tilt in always within -pi/2 and pi/4.
        tilt = min(max(tilt, -np.pi/2), np.pi/4)

        point = JointTrajectoryPoint()
        point.positions = [pan, tilt]
        point.time_from_start = rospy.Duration(2.5)
        
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = self.headJointNames
        goal.trajectory.points.append(point)
        
        self._head_trajectory_client.send_goal(goal)
        self._head_trajectory_client.wait_for_result()
        if verbose:     print('Head moved...\n')

################################################################################
################################################################################

    def findARtags(self, maxAttempts=5, verbose=True):
        '''
        Function that controls the head of the fetch robot and looks for AR tags.
        The head will move when using this function.
        It tries to find the AR tag by rotating the head. It will do this operation
        5 times at most, if it does not find a tag in 5 times, it exists this 
        function.
        If tags are found then the function exists immediately, and the location 
        of the tag is stored in the list arTagsPoseList.
        '''
        i = 0

        while len(self.arTagsPoseList) == 0 and i < maxAttempts:
            rospy.sleep(0.1)        # Waiting for the list to be updated.
            if rospy.is_shutdown():     break
            
            ## Rotate head to find AR tags.
            #for y in np.arange(-180, 180, 10):
                #if rospy.is_shutdown():     break
                #self.headPanHoriTilt(y, 0)                
                ## Break as soon as an AR tag is found.
                #if len(self.arTagsPoseList) != 0:     break

            # Rotate base to find AR tags.
            nIterations = 12
            for a in range(nIterations):  # Rotate by 360 deg till AR code found.
                if rospy.is_shutdown():     break
                self.turnBase(-360/nIterations * np.pi/180, speed=0.2, verbose=False)
                # Break as soon as an AR tag is found.
                if len(self.arTagsPoseList) != 0:     break

            i += 1
            if verbose:     print('Looking for AR tags...')
            
        # Flag that indicates that some tags are found.
        tagsFound = True if len(self.arTagsPoseList) > 0 else False
        arTagPose = self.arTagsPoseList[0]
        
        return tagsFound, arTagPose

################################################################################
################################################################################

    def findBigBlueBin_old(self, display=False):
        '''
        This function finds the big blue bin that the fetch has to put the 
        objects in.
        '''
        self.turnBase(-90*np.pi/180, speed=0.2, verbose=False)
        
        colorUpThresh = np.array([220, 80, 255])
        colorLwThresh = np.array([120, 10, 0])
        
        colorImg = copy.deepcopy(self.colorImg)
        depthImg = copy.deepcopy(self.depthImg)
        depthImgColorMap = copy.deepcopy(self.depthImgColorMap)
        
        # Filtering the maskedColorImg with the color range of the object.
        filteredImg = cv2.inRange(colorImg, colorLwThresh, colorUpThresh)
        # Dilating to fill up gaps and overlap the objects inside the bins.
        dilateKernel = np.ones((20, 20), np.uint8)
        filteredImg = cv2.dilate(filteredImg, dilateKernel)
        erodeKernel = np.ones((10, 10), np.uint8)   # Eroding to remove noise.
        filteredImg = cv2.erode(filteredImg, erodeKernel)
        
        # Now multiplying the filteredImg with the blue channel of the depthImgColorMap.
        # Because the bin is mostly visible in its blue channel.
        combinedImg = cv2.bitwise_and(depthImgColorMap[:,:,0], filteredImg)
        
        # Now take out the non zero pixels from this combined image using otsu's thresholding.
        _, thresholdImg = cv2.threshold(combinedImg, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # Now taking out the largest contour out of this thresholdImg.
        returnedTuple = cv2.findContours(thresholdImg, method=cv2.CHAIN_APPROX_SIMPLE, \
                                                        mode=cv2.RETR_LIST)
        contours = returnedTuple[-2]
        
        # Finding the largest contour.
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        largestContour = contours[0]
        rectX, rectY, rectW, rectH = cv2.boundingRect(largestContour)

        # Now drawing the rectangle around the largest contour in a blankImg.
        blankImg = np.zeros(thresholdImg.shape, dtype=np.uint8)
        cv2.rectangle(blankImg, (rectX, rectY), (rectX+rectW, rectY+rectH), 255, -1)
        
        ## Now finding the region of the depth image inside this rectangle, that is 
        ## more than 1m in depth.
        #depthImgAbove1m = cv2.bitwise_and(depthImg, blankImg)
        #depthImgAbove1m = np.array((depthImgAbove1m > 90) * 255, dtype=np.uint8)
        #cv2.imshow('Depth Image above 1m', depthImgAbove1m)
        
        # Now finding a point in the bin based on which the fetch will move forward
        # or backward a bit to position itself properly so that the arm can reach 
        # within the range of the bin. The ideal depth is 125 cm (1.25 m).
        # But this is not considering the fact that the head camera is having a 
        # tilt and the point may lie on the inner vertical wall of the bin or on the 
        # bottom surface of the bin or may also be on an object already inside the bin.
        targetPtX, targetPtY = int(rectX+rectW/2), int(rectY+rectH/3)
        targetDepth = depthImg[targetPtY, targetPtX]
        print(targetDepth)
        
        # Finding the target point in 3D.
        target3Dposi = self.pixel3DcoordinateUsingFOV(targetPtX, targetPtY)

        # Now move forward so that the bin is within reachable range.
        self.baseGoForward(targetDepth*0.01-1.20, speed=0.1, verbose=False)

        # Now we are moving the arm to that location in 3D keeping the z and the 
        # orientation of the gripper arm the same.
        target3Dposi[0] -= targetDepth*0.01-1.20    # Adjusting x to offset base movement.
        
        # This is to offset the gripper is now pointed downwards. So the overall x 
        # position will be less by the wrist to gripper finger lenght which is 0.35 m.
        target3Dposi[0] -= 0.35
        target3Dposi[2] = self.currentGripperPose.pose.position.z
        target3Dori = self.currentGripperPose.pose.orientation
        
        #cv2.drawContours(blankImg, [largestContour], -1, (0,255,0), 2)
        cv2.rectangle(colorImg, (rectX, rectY), (rectX+rectW, rectY+rectH), (0,0,255), 3)
        cv2.circle(colorImg, (targetPtX, targetPtY), 2, (0,0,255), 3)
        
        if display:
            cv2.imshow('Color Image', colorImg)
            #cv2.imshow('Depth Color Map', depthImgColorMap)
            #cv2.imshow('Filtered Image', filteredImg)
            #cv2.imshow('Combined Image', combinedImg)
            #cv2.imshow('Threshold Image', thresholdImg)

            key = cv2.waitKey(0)
            #if key & 0xFF == 27:    break    # break with esc key.
            #elif key & 0xFF == ord('q'):    break    # break with 'q' key.

            cv2.destroyAllWindows()
        
        position = Point(target3Dposi[0], target3Dposi[1], target3Dposi[2])
        orientation = target3Dori
        
        wristFlexJointState = self.currentArmJointStates['wrist_flex_joint']
        
        self.gripperToPose(Pose(position, orientation), verbose=False)
        
        # Straightening the joints.
        jointStateDict = {'elbow_flex_joint': 0.0, 'shoulder_lift_joint': 0.0, \
                          'wrist_flex_joint': wristFlexJointState}
        self.moveMultipleJoints(jointStateDict, verbose=False)

        # Now lower the torso to put the object inside the bin.
        self.moveSingleJoint('torso_lift_joint', 0.0, verbose=False)
        self.gripperOpen(gap=0.05, verbose=False)
        self.moveSingleJoint('torso_lift_joint', 0.4, verbose=False)
        
        # Put the arm to the side now.
        self.moveSingleJoint('shoulder_pan_joint', -1.5, verbose=False)

        # Turn the base back to the original location facing the table.
        self.baseGoForward(-1*(targetDepth*0.01-1.20), speed=0.1, verbose=False)
        self.turnBase(90*np.pi/180, speed=0.2, verbose=False)

################################################################################
################################################################################

    def findBigBlueBin(self, objectName=None, display=False):
        '''
        This function finds the big blue bin that the fetch has to put the 
        objects in.
        The object name has to be provided so that the arm moves accordingly.
        '''
        turnAngle = 70
        self.turnBase(-1*turnAngle*np.pi/180, speed=0.4, verbose=False)
        
        ## Rotating the roll joints by 90 degrees so that the arm can fold back 
        ## on the horizontal plane keeping the gripper orientation the same.
        ## The upperarm_roll_joint and forearm_roll_joint are moved in a way to 
        ## offset their rotations.
        #upperarmRollJointNewState = 1.56
        #forearmRollJointState = self.currentArmJointStates['forearm_roll_joint']
        #forearmRollJointNewState = forearmRollJointState - 1.56

        #jointStateDict = {'upperarm_roll_joint': upperarmRollJointNewState, 
                          #'forearm_roll_joint': forearmRollJointNewState}        
        #self.moveMultipleJoints(jointStateDict, verbose=False)
        
        colorUpThresh = np.array([220, 80, 255])
        colorLwThresh = np.array([120, 10, 0])
        
        # Just updating the callback function.
        msg = rospy.wait_for_message(self.namespace + 'head_camera/depth/image_raw', Image)
        msg = rospy.wait_for_message(self.namespace + 'head_camera/rgb/image_raw', Image)

        colorImg = copy.deepcopy(self.colorImg)
        depthImg = copy.deepcopy(self.depthImg)
        depthImgColorMap = copy.deepcopy(self.depthImgColorMap)
        
        # Filtering the maskedColorImg with the color range of the object.
        filteredImg = cv2.inRange(colorImg, colorLwThresh, colorUpThresh)
        # Dilating to fill up gaps and overlap the objects inside the bins.
        dilateKernel = np.ones((20, 20), np.uint8)
        filteredImg = cv2.dilate(filteredImg, dilateKernel)
        erodeKernel = np.ones((10, 10), np.uint8)   # Eroding to remove noise.
        filteredImg = cv2.erode(filteredImg, erodeKernel)
        
        # Now multiplying the filteredImg with the blue channel of the depthImgColorMap.
        # Because the bin is mostly visible in its blue channel.
        combinedImg = cv2.bitwise_and(depthImgColorMap[:,:,0], filteredImg)
        
        # Now take out the non zero pixels from this combined image using otsu's thresholding.
        _, thresholdImg = cv2.threshold(combinedImg, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # Now taking out the largest contour out of this thresholdImg.
        returnedTuple = cv2.findContours(thresholdImg, method=cv2.CHAIN_APPROX_SIMPLE, \
                                                        mode=cv2.RETR_LIST)
        contours = returnedTuple[-2]
        
        # Finding the largest contour.
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        largestContour = contours[0]
        rectX, rectY, rectW, rectH = cv2.boundingRect(largestContour)

        ## Now drawing the rectangle around the largest contour in a blankImg.
        #blankImg = np.zeros(thresholdImg.shape, dtype=np.uint8)
        #cv2.rectangle(blankImg, (rectX, rectY), (rectX+rectW, rectY+rectH), 255, -1)
        ##cv2.imshow('Blank Image', blankImg)
        
        ## Now finding the region of the depth image inside this rectangle, that is 
        ## more than 1m in depth.
        #depthImgAbove1m = cv2.bitwise_and(depthImg, blankImg)
        #depthImgAbove1m = np.array((depthImgAbove1m > 90) * 255, dtype=np.uint8)
        #cv2.imshow('Depth Image above 1m', depthImgAbove1m)
        
        # Now finding a point in the bin based on which the fetch will move forward
        # or backward a bit to position itself properly so that the arm can reach 
        # within the range of the bin. The ideal depth is 68 cm (0.68 m).
        # But this is not considering the fact that the head camera is having a 
        # tilt and the point may lie on the inner vertical wall of the bin or on the 
        # bottom surface of the bin or may also be on an object already inside the bin.
        targetPtX, targetPtY = int(rectX+rectW/2), int(rectY+rectH/4)
        targetDepth = self.robustDepthWithHist(targetPtY, targetPtX, winH=100, winW=100)        
        print('targetDepth, targetPtX, targetPtY: {}, {}, {}'.format(targetDepth, targetPtX, targetPtY))
        
        # Finding the target point in 3D.
        target3Dposi = self.pixel3DcoordinateUsingFOV(targetPtX, targetPtY)
        print('target3Dposi: {}'.format(target3Dposi))
        
        #cv2.drawContours(blankImg, [largestContour], -1, (0,255,0), 2)
        rawColorAndDepthMapImg = np.hstack((colorImg, depthImgColorMap))
        cv2.rectangle(colorImg, (rectX, rectY), (rectX+rectW, rectY+rectH), (0,0,255), 3)
        cv2.circle(colorImg, (targetPtX, targetPtY), 2, (0,0,255), 3)
        cv2.circle(depthImgColorMap, (targetPtX, targetPtY), 2, (0,0,255), 3)
        
        if display:
            predColorAndDepthMapImg = np.hstack((colorImg, depthImgColorMap))
            cv2.imshow('Color and Depth Color Map Image', predColorAndDepthMapImg)
            #cv2.imshow('Color Image', colorImg)
            #cv2.imshow('Depth Color Map', depthImgColorMap)
            #cv2.imshow('Filtered Image', filteredImg)
            #cv2.imshow('Combined Image', combinedImg)
            #cv2.imshow('Threshold Image', thresholdImg)

            key = cv2.waitKey(0)
            if key & 0xFF == ord('s'):    # Save image in the imgSaveFolder with timeStamp.
                imgFileName = 'predicted_color_and_depth_map_img_{}.png'.format(timeStamp())
                cv2.imwrite(os.path.join(fetchRobot.imgSaveFolder, imgFileName), predColorAndDepthMapImg)
                imgFileName = 'raw_color_and_depth_map_img_{}.png'.format(timeStamp())
                cv2.imwrite(os.path.join(fetchRobot.imgSaveFolder, imgFileName), rawColorAndDepthMapImg)

            cv2.destroyAllWindows()
        
        #if objectName == 'nuts' or objectName == 'coins' or objectName == 'washers' or \
           #objectName == 'gears' or objectName == 'emptyBin':
            #jointStateDict = {'shoulder_pan_joint': 0.0, 'shoulder_lift_joint': 1.5, 
                               #'elbow_flex_joint': 0.0}
            #self.moveMultipleJoints(jointStateDict, verbose=False)

        #elif objectName == 'crankArmW' or objectName == 'crankArmX' or objectName == 'crankShaft':
            #self.moveSingleJoint('shoulder_pan_joint', 0.0, verbose=False)
            #self.moveSingleJoint('shoulder_lift_joint', 0.0, verbose=False)
        
        self.moveSingleJoint('shoulder_pan_joint', 0.0, verbose=False)
        self.moveSingleJoint('shoulder_lift_joint', 0.0, verbose=False)

        print('current gripper pose: {}'.format(self.currentGripperPose))

        # Now move so that the bin is within reachable range.
        
        # The range is made 3/4th of the actual distance between the target point 
        # and current gripper pose.
        # The distance is only made a fraction of the actual value to account for 
        # the gripper size, object size and the inaccuracies in measurement.
        translation = (target3Dposi[0] - 0.83)*0.6
        #translation = (target3Dposi[0] - self.currentGripperPose.pose.position.x)*0.6
        
        perpendicular = target3Dposi[1] - self.currentGripperPose.pose.position.y
        base = self.currentGripperPose.pose.position.x
        
        # Calculating angle from absolute value, as the sign only determines in 
        # which direction the rotation will take place.
        # The angle is only made a fraction of the actual value to account for 
        # the gripper size, object size and the inaccuracies in measurement.
        angle = np.arctan(abs(perpendicular/base))*0.6
        rotation = angle if perpendicular > 0 else -1*angle
        
        print('perpendicular: {}, rotation: {}'.format(perpendicular, rotation))

        self.baseGoForward(translation, speed=0.1, verbose=False)
        #self.turnBase(rotation, speed=0.4, verbose=False)
        
        time.sleep(5)

        # Now lower the torso to put the object inside the bin.
        self.moveSingleJoint('torso_lift_joint', 0.0, verbose=False)
        self.gripperOpen(gap=0.05, verbose=False)
        self.moveSingleJoint('torso_lift_joint', 0.4, verbose=False)
        
        # Put the arm to the side now.
        self.moveSingleJoint('shoulder_pan_joint', -1.5, verbose=False)

        # Turn the base back to the original location facing the table.
        #self.turnBase(-1*rotation, speed=0.4, verbose=False)
        self.baseGoForward(-1*translation, speed=0.1, verbose=False)
        self.turnBase(turnAngle*np.pi/180, speed=0.4, verbose=False)
        
        time.sleep(5)

################################################################################
################################################################################
################################################################################
################################################################################

if __name__ == '__main__':
    
    fetchRobot = fetchRobotClassObj()
    
################################################################################
    
    #fetchRobot.baseGoForward(-0.5)
    #fetchRobot.turnBase(45*np.pi/180)
    #fetchRobot.turnBase(-45*np.pi/180)

    ##fetchRobot.moveSingleJoint('torso_lift_joint', 0.4, verbose=False)
    ##jointStateDict = {'shoulder_pan_joint': 0.0,
                       ##'shoulder_lift_joint': 0.0, 'upperarm_roll_joint': 0.0,
                       ##'elbow_flex_joint': 0.0, 'forearm_roll_joint': 0.0,
                       ##'wrist_flex_joint': 0.0, 'wrist_roll_joint': 0.0}
    ##fetchRobot.moveMultipleJoints(jointStateDict, verbose=False)
    #fetchRobot.moveSingleJoint('shoulder_pan_joint', -1.6, verbose=False)

##########------------------------------Navigating the fetch---------------------------------------------

    # Make sure sim time is working
    while not rospy.Time.now():
        pass

    # First moving into an empty region. This will help the fetch to have a better 
    # estimate of its current position.
    fetchRobot.baseGoForward(0.5)

    # Setup clients
    moveBaseFetch = MoveBaseClient()

    rospy.loginfo("Moving to table...")
    
    targetPose = Pose(Point(0.58, 5.94, 0.0), Quaternion(0.0, 0.0, -0.77522, 0.6317))
    moveBaseFetch.goto(targetPose)
    print(moveBaseFetch.current_amcl_pose)
    
    targetPose = Pose(Point(-1.64, 0.684, 0.0), Quaternion(0.0, 0.0, -0.77522, 0.6317))
    moveBaseFetch.goto(targetPose)
    print(moveBaseFetch.current_amcl_pose)

    fetchRobot.baseGoForward(0.10, speed=0.1, verbose=False)
    fetchRobot.turnBase(90*np.pi/180, speed=0.4, verbose=False)
    fetchRobot.baseGoForward(0.1, speed=0.02, verbose=False)
    fetchRobot.baseGoForward(0.6, speed=0.1, verbose=False)

    time.sleep(10)

###########--------------------------Main testing--------------------------------------------

    # Putting arm and robot to start position.
    jointStateDict = {'torso_lift_joint': 0.15, 'shoulder_pan_joint': 0.0,
                       'shoulder_lift_joint': 1.5, 'upperarm_roll_joint': 0.0,
                       'elbow_flex_joint': -1.65, 'forearm_roll_joint': -3.05,
                       'wrist_flex_joint': 1.6, 'wrist_roll_joint': 1.5}
    fetchRobot.moveMultipleJoints(jointStateDict, verbose=False)
    
    fetchRobot.moveSingleJoint('shoulder_pan_joint', 1.6, verbose=False)
    
    jointStateDict = {'shoulder_lift_joint': 0.0, 'upperarm_roll_joint': 0.0, 
                      'elbow_flex_joint': 0.0, 'forearm_roll_joint': 0.0, 
                      'wrist_flex_joint': 0.0, 'wrist_roll_joint': 0.0}
    fetchRobot.moveMultipleJoints(jointStateDict, verbose=False)
    
    fetchRobot.moveSingleJoint('torso_lift_joint', 0.4, verbose=False)

    fetchRobot.moveSingleJoint('shoulder_pan_joint', -1.5, verbose=False)
    fetchRobot.gripperOpen(verbose=False)
    #fetchRobot.gripperToPose(Point(0.5, -0.75, 1.1), verbose=False)
    forwardOffset = 0.4
    fetchRobot.baseGoForward(forwardOffset, speed=0.1, verbose=False)
    headTiltHori, headPanHori, cameraHeight = 50, 0, 1
    fetchRobot.headPanHoriTilt(0, headTiltHori, verbose=False)
    
    key = ord('`')
    detector = networkDetector()
    count = 0       # Indicates the number of times no object is detected.
    repeatFlag = 'False'    # Indicates if object detection attempt has to be repeated or not.

    # Show the converted image.
    while not rospy.is_shutdown():
        
        if repeatFlag:
            # If the repeatFlag is True, then it means the robot is not able to 
            # detect an object properly, hence its position near the table is 
            # adjusted randomly to a slight extent.
            translation = np.random.randint(6) + 1     # To ignore the zero value.
            translation *= -0.01
            fetchRobot.baseGoForward(translation, speed=0.05, verbose=False)

            rotation = np.random.randint(8) - 3     # To have both +ve and -ve values.
            fetchRobot.turnBase(rotation*np.pi/180, speed=0.4, verbose=False)
            
            # Now reverse the movement, but this is also done sometimes randomly.
            if np.random.randint(2) == 1:
                fetchRobot.baseGoForward(-1*translation, speed=0.05, verbose=False)
 
        # Just updating the callback function.
        msg = rospy.wait_for_message(fetchRobot.namespace + 'head_camera/depth/image_raw', Image)
        msg = rospy.wait_for_message(fetchRobot.namespace + 'head_camera/rgb/image_raw', Image)
        
        # Prediction from network.
        colorImg = copy.deepcopy(fetchRobot.colorImg)
        depthImg = copy.deepcopy(fetchRobot.depthImg)
        depthImgColorMap = copy.deepcopy(fetchRobot.depthImgColorMap)
        
        rawImg = copy.deepcopy(colorImg)
        img = copy.deepcopy(rawImg)
        
        imgBatch = np.array([img])
        
        inferLayerOut, inferPredLogits, inferPredResult, _, _ = detector.batchInference(imgBatch)
        
        detectedBatchClassScores, _, detectedBatchClassNames, detectedBatchBboxes \
                                            = nonMaxSuppression(inferPredResult)
        
        # The output of the nonMaxSuppression is in the form of a batch.
        # So extracting the contents of this batch since there is an output 
        # of only one image in this batch.        
        detectedBatchClassScores = detectedBatchClassScores[0]
        detectedBatchClassNames = detectedBatchClassNames[0]
        detectedBatchBboxes = detectedBatchBboxes[0]
        
################################################################################

        # Draw the detected results now.
        if len(detectedBatchClassNames) == 0:
            print('\nNo objects detected...\n')
            if count < 3:
                count += 1
                print('No object detected trial: {}'.format(count))
                repeatFlag = True       # Adjust position near table.
                time.sleep(10)
                cv2.imshow('Predicted Image', img)
                key = cv2.waitKey(0)
                if key & 0xFF == ord('r'):   count = 0
                elif key & 0xFF == ord('q'): break
                cv2.destroyAllWindows()
                continue
            else:
                repeatFlag = False
                break

        pdx = 0
        p = detectedBatchClassNames[pdx]
        x, y, w, h = detectedBatchBboxes[pdx].tolist()

        # Only draw the bounding boxes for the non-rbc entities.
        if p == 'nuts' or p == 'coins' or p == 'washers' or p == 'gears' or p == 'emptyBin':
            color = (0,0,255)       # RED.
            label = p
            print(p, x, y, w, h)
            count, repeatFlag = 0, False
        if p == 'crankArmW' or p == 'crankArmX' or p == 'crankShaft':
            color = (0,255,0)       # GREEN.
            label = 'crankShaft'
            print(p, x, y, w, h)
            count, repeatFlag = 0, False
            if w > 300:
                # Width of the bounding box is too large indicates wrong bounding box prediction.
                count, repeatFlag = 0, True     # Adjust position near table.
                continue
        
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.circle(img, (int(x+w/2), int(y+h/2)), 2, color, 2)
        cv2.putText(img, label, (x+5, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        #cv2.circle(depthImgColorMap, (int(x+w/2), int(y+h/2)), 2, (0,0,255), 2)
        
        score = detectedBatchClassScores[pdx]
        #print(p, score)

        cv2.imshow('Predicted Image', img)
        #cv2.imshow('Color Image', colorImg)
        #cv2.imshow('Depth Image', depthImg)
        #cv2.imshow('Depth Color Map', depthImgColorMap)

        key = cv2.waitKey(0)
        if key & 0xFF == 27:    break    # break with esc key.
        elif key & 0xFF == ord('q'):    break    # break with 'q' key.
        elif key & 0xFF == ord('s'):    # Save image in the imgSaveFolder with timeStamp.
            imgFileName = 'raw_img_{}.png'.format(timeStamp())
            cv2.imwrite(os.path.join(fetchRobot.imgSaveFolder, imgFileName), rawImg)
            imgFileName = 'predicted_img_{}.png'.format(timeStamp())
            cv2.imwrite(os.path.join(fetchRobot.imgSaveFolder, imgFileName), img)
        elif key & 0xFF == ord('r'):    # Repeat.
            cv2.destroyAllWindows()
            count, repeatFlag = 0, True     # Adjust position near table.
            time.sleep(5)
            continue
        
        cv2.destroyAllWindows()

        # Finding the center point of the bounding box as this point is very 
        # likely to be located on the object itself, even if the bounds of 
        # the object bounding box are outside the table region.
        # The distances of the bounding box boundaries from this center point
        # are also determined in 3D and stored in some parameters.
        center3D = fetchRobot.pixel3DcoordinateUsingFOV(x+w/2, y+h/2)
        leftPt3D = fetchRobot.pixel3DcoordinateUsingFOV(x, y+h/2)
        frontPt3D = fetchRobot.pixel3DcoordinateUsingFOV(x+w/2, y)
        yLim3D = np.fabs(center3D[1] - leftPt3D[1])
        xLim3D = np.fabs(center3D[0] - frontPt3D[0])
        
        #point3Darr = fetchRobot.arrayOf3DcoordinatesUsingFOV(x, y, w, h, p)
        #print(point3Darr.shape)
        rospy.set_param('markerX', float(center3D[0]))
        rospy.set_param('markerY', float(center3D[1]))
        rospy.set_param('markerZ', float(center3D[2]))
        rospy.set_param('limitX', float(xLim3D))
        rospy.set_param('limitY', float(yLim3D))
        rospy.set_param('objectName', str(p))
        #print(p, center3D)

        print('Running c++ file for processing the point cloud...')
        subprocess.call(["rosrun", "objectPickup", "extract_object"])

        # Updating the parameter values into some variables. These are created 
        # by the c++ file.
        objectName = rospy.get_param('objectName')
        objPosiX = rospy.get_param(objectName + '_posiX')
        objPosiY = rospy.get_param(objectName + '_posiY')
        objPosiZ = rospy.get_param(objectName + '_posiZ')
        objOriX = rospy.get_param(objectName + '_oriX')
        objOriY = rospy.get_param(objectName + '_oriY')
        objOriZ = rospy.get_param(objectName + '_oriZ')
        objOriW = rospy.get_param(objectName + '_oriW')
        objScaleX = rospy.get_param(objectName + '_scaleX')
        objScaleY = rospy.get_param(objectName + '_scaleY')
        objScaleZ = rospy.get_param(objectName + '_scaleZ')
        tablePosiZ = rospy.get_param('table_posiZ')
        tableScaleZ = rospy.get_param('table_scaleZ')
        
        #print(objPosiX, objPosiY, objPosiZ)
        #print(objOriX, objOriY, objOriZ, objOriW)
        #print(objScaleX, objScaleY, objScaleZ)
        
################################################################################

        # Initializing the position of the arm before moving it to pickup object.
        jointStateDict = {'shoulder_pan_joint': -1.5, 'shoulder_lift_joint': 0.0, 
                          'upperarm_roll_joint': 0.0, 'elbow_flex_joint': 0.0, 
                          'forearm_roll_joint': 0.0, 'wrist_flex_joint': 0.0, 
                          'wrist_roll_joint': 0.0}
        fetchRobot.moveMultipleJoints(jointStateDict, verbose=False)

        if objectName == 'nuts' or objectName == 'coins' or objectName == 'washers' or \
           objectName == 'gears' or objectName == 'emptyBin':
            print('\n\n{}\n\n'.format(objectName))
            # Setting the position and orientation for the gripper.
            # The gripper is positioned such that it can hold one of the side walls 
            # of the blue bin. Hence there is some offset along the y axis.
            # But it will orient the gripper a bit above the actual height of the 
            # object so that it does not hit the object by accident. Hence the z offset.
            hOffset = 0.1
            position = Point(objPosiX, objPosiY + objScaleY*0.5, objPosiZ + hOffset)
            orientation = Quaternion(objOriX, objOriY, objOriZ, objOriW)
            angles = tf.transformations.euler_from_quaternion([objOriX, objOriY, objOriZ, objOriW])
            
            print(objOriX, objOriY, objOriZ, objOriW)
            print(np.rad2deg([angles]))

            jointStateDict = {'shoulder_pan_joint': 0.0, 'shoulder_lift_joint': 0.0, 
                              'upperarm_roll_joint': 0.0, 'elbow_flex_joint': 0.0, 
                              'forearm_roll_joint': 0.0, 'wrist_flex_joint': 0.0, 
                              'wrist_roll_joint': 0.0}
            fetchRobot.moveMultipleJoints(jointStateDict, verbose=False)
            fetchRobot.gripperOpen(verbose=False)
       
            # We want to grab the bin from the top vertically. So fixing an orientation
            # rotated along the y axis and then moving the arm.
            # But sometimes if the rotation along y axis is made 90 degrees, then 
            # the arm is not able to reach the position. Hence we have made it 80 degrees.
            rotatedOrientation = tf.transformations.quaternion_from_euler(0, np.deg2rad(90), angles[2])
            rotatedOrientation = Quaternion(*rotatedOrientation)
            ret = fetchRobot.gripperToPose(Pose(position, rotatedOrientation), verbose=False)
            
            if ret == False:    # This means the pose cannot be reached. 
                ## Then just rotate the shoulder_pan_joint a bit towards the table.
                #shoulderPanJointState = fetchRobot.currentArmJointStates['shoulder_pan_joint']
                #shoulderPanJointNewState = shoulderPanJointState + 0.25
                #fetchRobot.moveSingleJoint('shoulder_pan_joint', shoulderLiftJointNewState, verbose=False)
                count, repeatFlag = 0, True     # Adjust position near table.
                continue

            print('current joint states: {}'.format(fetchRobot.currentArmJointStates))

            ## Now rotating only the wrist joint to be across the yaw orientation of the 
            ## object, so that its easy to pick up.
            ## A value of 3 to the wrist joint rotates it to pi radians.
            #fetchRobot.moveSingleJoint('wrist_roll_joint', angles[2]*3/np.pi, verbose=False)
            
            # Now lower the torso_lift_joint and grip the object. If we have 
            # set the arm at a height of 0.1 m above the z position of the object,
            # then we have to lower the torso by 0.1 m. Lowering the torso by 0.1 m
            # means decrementing the current value of the torso_lift_joint by 0.1.
            torsoLiftJointState = fetchRobot.currentArmJointStates['torso_lift_joint']
            torsoLiftJointNewState = torsoLiftJointState - 0.1
            fetchRobot.moveSingleJoint('torso_lift_joint', torsoLiftJointNewState, verbose=False)
            fetchRobot.gripperClose(verbose=False)
            fetchRobot.moveSingleJoint('torso_lift_joint', 0.4, verbose=False)

            # The shoulder_lift_joint is raised a bit so that the picked up objects dont collide 
            # with other objects on the table while moving.
            shoulderLiftJointState = fetchRobot.currentArmJointStates['shoulder_lift_joint']
            shoulderLiftJointNewState = shoulderLiftJointState - 0.2
            fetchRobot.moveSingleJoint('shoulder_lift_joint', shoulderLiftJointNewState, verbose=False)
            
################################################################################

            #print(fetchRobot.currentArmJointStates)
            
            # Now after gripping the object we are making the arm as straight as 
            # possible, so that it is easier to put it on the freight.
            # For this we are straightening the upperarm_roll_joint.
            # And to have the gripper in the same position, the value of this 
            # joint will have to be counteracted by the forearm_roll_joint with 
            # the same values. This will work since these two roll joints of the 
            # fetch are identical mechanically. The adjustments for the 
            # shoulder_lift_joint is also included and it is made horizontal. 
            # Now because of all these adjustments the wrist_flex_joint may 
            # become bent and the object may no longer be horizontal.
            # So, while the forearm_roll_joint is being adjusted, based on its 
            # value the wrist_flex_joint is also adjusted to that the object 
            # stays horizontal. The shoulder_pan_joint is also moved to the side.
            upperarmRollJointState = fetchRobot.currentArmJointStates['upperarm_roll_joint']
            elbowFlexJointState = fetchRobot.currentArmJointStates['elbow_flex_joint']
            
            forearmRollJointState = fetchRobot.currentArmJointStates['forearm_roll_joint']
            forearmRollJointNewState = forearmRollJointState + upperarmRollJointState
            #wristFlexJointState = fetchRobot.currentArmJointStates['wrist_flex_joint']
            #wristFlexJointNewState = wristFlexJointState + elbowFlexJointState

            # forearm_roll_joint values cycles from -6.25 to 0 to 6.25. 
            # I.e. the position of the joint for -6.25, 0 and 6.25 and their 
            # multiples are the same. The sign just shows in which direction 
            # the joint will rotate. So, the wrist_flex_joint should be assigned 
            # a value based on the new value assigned to the forearm_roll_joint.
            # So, first the sign is removed from the assigned value of the 
            # forearm_roll_joint and then it is divided by 6.25 to get the 
            # remainder. After this modulo division, the result will always be
            # withing 0 and 6.25. Now, after the object is picked up and the 
            # flex joints are adjusted to 0, the forearm_roll_joint will always 
            # be oriented near about the 0 degree or the 180 degree zone. So its 
            # value will be near 0 or near 3.12 respectively. Hence based on 
            # that the value of the wrist_flex_joint is assigned is the value is 
            # below of above 1.56 (half of 3.12).
            if (abs(forearmRollJointNewState) % 6.25) > 1.56:
                wristFlexJointNewState = -1.55
            else:
                wristFlexJointNewState = 1.55
            
            elbowFlexJointNewState = 0.65

            # Straightening the joints.
            jointStateDict = {'upperarm_roll_joint': 0.0, 'elbow_flex_joint': elbowFlexJointNewState, 
                              'shoulder_pan_joint': -1.5, 
                              'forearm_roll_joint': forearmRollJointNewState, 
                              'wrist_flex_joint': wristFlexJointNewState}
            fetchRobot.moveMultipleJoints(jointStateDict, verbose=False)
            
            # Straightening the joints again.
            jointStateDict = {'shoulder_lift_joint': 0.0, 'elbow_flex_joint': 0.0}
            fetchRobot.moveMultipleJoints(jointStateDict, verbose=False)
            
################################################################################
            
        elif objectName == 'crankArmW' or objectName == 'crankArmX' or objectName == 'crankShaft':
            print('\n\n{}\n\n'.format(objectName))
            # Setting the position and orientation for the gripper.
            # The gripper is positioned such that it can hold one of the straight 
            # regions of the crank arm parts. 
            # But it will orient the gripper a bit above the actual height of the 
            # object so that it does not hit the object by accident. Hence the z offset.
            hOffset = 0.1
            position = Point(objPosiX, objPosiY, objPosiZ + hOffset)
            orientation = Quaternion(objOriX, objOriY, objOriZ, objOriW)
            angles = tf.transformations.euler_from_quaternion([objOriX, objOriY, objOriZ, objOriW])
            
            print(objOriX, objOriY, objOriZ, objOriW)
            print(np.rad2deg([angles]))
            
            fetchRobot.gripperOpen(verbose=False)

            # We want to grab the bin from the top vertically. So fixing an orientation
            # rotated along the y axis and then moving the arm.
            # But sometimes if the rotation along y axis is made 90 degrees, then 
            # the arm is not able to reach the position. Hence we have made it 80 degrees.
            rotatedOrientation = tf.transformations.quaternion_from_euler(0, np.deg2rad(90), angles[2])
            rotatedOrientation = Quaternion(*rotatedOrientation)
            ret = fetchRobot.gripperToPose(Pose(position, rotatedOrientation), verbose=False)
            
            if ret == False:    # This means the pose cannot be reached.
                ## Then just rotate the shoulder_pan_joint a bit towards the table.
                #shoulderPanJointState = fetchRobot.currentArmJointStates['shoulder_pan_joint']
                #shoulderPanJointNewState = shoulderPanJointState + 0.25
                #fetchRobot.moveSingleJoint('shoulder_pan_joint', shoulderLiftJointNewState, verbose=False)
                count, repeatFlag = 0, True     # Adjust position near table.
                continue

            ## Now rotating only the wrist joint to be across the yaw orientation of the 
            ## object, so that its easy to pick up.
            ## A value of 3 to the wrist joint rotates it to pi radians.
            #fetchRobot.moveSingleJoint('wrist_roll_joint', angles[2]*3/np.pi, verbose=False)
            
            # Now lower the torso_lift_joint and grip the object. If we have 
            # set the arm at a height of 0.1 m above the z position of the object,
            # In this case we lower the torso in such a way that the gripper reaches the 
            # height of the table surface, for a better grip of the crankShaft like objects.
            # So we lower the torso by that many meters. Lowering the torso by 0.1 m
            # means decrementing the current value of the torso_lift_joint by 0.1.
            torsoLiftJointState = fetchRobot.currentArmJointStates['torso_lift_joint']
            torsoLiftJointNewState = torsoLiftJointState - \
                                        (objPosiZ + hOffset - (tablePosiZ + tableScaleZ*0.5))
            fetchRobot.moveSingleJoint('torso_lift_joint', torsoLiftJointNewState, verbose=False)
            fetchRobot.gripperClose(verbose=False)
            fetchRobot.moveSingleJoint('torso_lift_joint', 0.4, verbose=False)

            # The wrist_flex_joint should be made straight so that the object does 
            # not collide with any other object on the table.
            fetchRobot.moveSingleJoint('wrist_flex_joint', 0.0, verbose=False)

################################################################################

            #print(fetchRobot.currentArmJointStates)
            
            # Now after gripping the object we are making the arm as straight as 
            # possible, so that it is easier to put it on the freight.
            # For this we are straightening the upperarm_roll_joint.
            # And to have the gripper in the same position, the value of this 
            # joint will have to be counteracted by the forearm_roll_joint with 
            # the same values. This will work since these two roll joints of the 
            # fetch are identical mechanically. The adjustments for the 
            # shoulder_lift_joint is also included and it is made horizontal. 
            # Now because of all these adjustments the wrist_flex_joint may 
            # become bent and the object may no longer be horizontal.
            # So, while the forearm_roll_joint is being adjusted, based on its 
            # value the wrist_flex_joint is also adjusted to that the object 
            # stays horizontal. The shoulder_pan_joint is also moved to the side.
            upperarmRollJointState = fetchRobot.currentArmJointStates['upperarm_roll_joint']
            elbowFlexJointState = fetchRobot.currentArmJointStates['elbow_flex_joint']
            
            forearmRollJointState = fetchRobot.currentArmJointStates['forearm_roll_joint']
            forearmRollJointNewState = forearmRollJointState + upperarmRollJointState
            #wristFlexJointState = fetchRobot.currentArmJointStates['wrist_flex_joint']
            #wristFlexJointNewState = wristFlexJointState + elbowFlexJointState

            # forearm_roll_joint values cycles from -6.25 to 0 to 6.25. 
            # I.e. the position of the joint for -6.25, 0 and 6.25 and their 
            # multiples are the same. The sign just shows in which direction 
            # the joint will rotate. So, the wrist_flex_joint should be assigned 
            # a value based on the new value assigned to the forearm_roll_joint.
            # So, first the sign is removed from the assigned value of the 
            # forearm_roll_joint and then it is divided by 6.25 to get the 
            # remainder. After this modulo division, the result will always be
            # withing 0 and 6.25. Now, after the object is picked up and the 
            # flex joints are adjusted to 0, the forearm_roll_joint will always 
            # be oriented near about the 0 degree or the 180 degree zone. So its 
            # value will be near 0 or near 3.12 respectively. Hence based on 
            # that the value of the wrist_flex_joint is assigned is the value is 
            # below of above 1.56 (half of 3.12).
            if (abs(forearmRollJointNewState) % 6.25) > 1.56:
                wristFlexJointNewState = -1.55
            else:
                wristFlexJointNewState = 1.55
            
            # Straightening the joints.
            jointStateDict = {'upperarm_roll_joint': 0.0, 'elbow_flex_joint': 0.0, 
                              'shoulder_pan_joint': -1.5, 
                              'forearm_roll_joint': forearmRollJointNewState, 
                              'wrist_flex_joint': wristFlexJointNewState}
            fetchRobot.moveMultipleJoints(jointStateDict, verbose=False)
            
################################################################################

        # If fails to pick up then continue from start of while loop. You have 
        # to re-identify the objects, because there may be some collision with
        # the arm and the objects for which the objects may have moved away 
        # from their positions.
        # To do this the gap between the gripper fingers is checked.
        if fetchRobot.currentAllJointStates['l_gripper_finger_joint'] < 0.0005 and \
           fetchRobot.currentAllJointStates['r_gripper_finger_joint'] < 0.0005:
            count, repeatFlag = 0, True     # Adjust position near table.
            continue
        else:
            count, repeatFlag = 0, False

################################################################################

        # Finding the big blue bin.
        fetchRobot.findBigBlueBin(objectName=objectName, display=True)

################################################################################

        if key & 0xFF == ord('q') or key & 0xFF == 27:
            break    # break with esc key.
        
################################################################################

    if key & 0xFF == ord('q'):
        # Go back to initial position and tuck the arm if 'q' is pressed.
        cv2.destroyAllWindows()
        fetchRobot.baseGoForward(-1*(forwardOffset+0.2), speed=0.1, verbose=False)

        # Initializing the position of the arm before tucking.        
        jointStateDict = {'shoulder_pan_joint': 1.5, 'shoulder_lift_joint': 0.0, 
                          'upperarm_roll_joint': 0.0, 'elbow_flex_joint': 0.0, 
                          'forearm_roll_joint': 0.0, 'wrist_flex_joint': 1.6, 
                          'wrist_roll_joint': 1.5}
        fetchRobot.moveMultipleJoints(jointStateDict, verbose=False)
        
        fetchRobot.autoTuckArm(verbose=False)

################################################################################

    if len(detectedBatchClassNames) == 0:
        # Go back to initial position and tuck the arm and navigate back to 
        # starting position is no other objects are detected.
        cv2.destroyAllWindows()
        fetchRobot.baseGoForward(-1*forwardOffset, speed=0.1, verbose=False)

        # Initializing the position of the arm before tucking.        
        jointStateDict = {'shoulder_pan_joint': 1.5, 'shoulder_lift_joint': 0.0, 
                          'upperarm_roll_joint': 0.0, 'elbow_flex_joint': 0.0, 
                          'forearm_roll_joint': 0.0, 'wrist_flex_joint': 1.6, 
                          'wrist_roll_joint': 1.5}
        fetchRobot.moveMultipleJoints(jointStateDict, verbose=False)
        
        fetchRobot.autoTuckArm(verbose=False)
        
        time.sleep(10)

        #fetchRobot.turnBase(90*np.pi/180)

############--------------------Navigating the fetch back to home-------------------------------

        ## Make sure sim time is working
        #while not rospy.Time.now():
            #pass

        #rospy.loginfo("Moving away from table...")
        
        #targetPose = Pose(Point(-5.35, 6.126, 0.0), Quaternion(0.0, 0.0, 0.92455, 0.38107))
        #moveBaseFetch.goto(targetPose)
        #print(moveBaseFetch.current_amcl_pose)
        
#################################################################################

    elif key & 0xFF == 27:
        # Stays in the same position and quits the code.
        cv2.destroyAllWindows()
    
    
    
    
    