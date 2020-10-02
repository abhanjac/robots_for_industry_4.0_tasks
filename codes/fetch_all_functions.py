#!/usr/bin/env python

import rospy, tf, numpy as np, copy, cv2, os, subprocess, sys, termios, tty, time
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
import tensorflow as tfl
from utils import *
from network import *

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

# The fetch arm has to grab different types of objects. Since this is a factory 
# setting, so the different objects are actually predefined. Every object has to 
# to be identified and there will be some image processing happening that will 
# take into consideration the different attributes about the object. This may 
# include what part of the images will the camera of the fetch focus on when it
# is holding the object in the arm, what should be the joint states to have the 
# arm of the fetch hold the object in a proper manner etc.

obj1 = {'name': 'crankArm', 'objId': 1, 'headPanHori': 0, 'headTiltHori': 0, 
         'roiTlXhori': 180, 'roiTlYhori': 240, 'roiBrXhori': 540, 'roiBrYhori': 480, 
         'roiDlwHori': 60, 'roiDupHori': 85, 'avgAngleHori': 8.39, 
         'jointStateDictHori': {'shoulder_pan_joint': 0.0, 'shoulder_lift_joint': 0.7, 
                                 'upperarm_roll_joint': 0.0, 'elbow_flex_joint': -1.8, 
                                 'forearm_roll_joint': 0.0, 'wrist_flex_joint': 1.1, 
                                 'wrist_roll_joint': 0.0}}

obj2 = {'name': 'blueBin', 'objId': 2, 'headPanHori': 0, 'headTiltHori': 30, 
         'roiTlXhori': 0, 'roiTlYhori': 0, 'roiBrXhori': 640, 'roiBrYhori': 480, 
         'roiDlwHori': 60, 'roiDupHori': 90, 'colorUpThresh': np.array([220, 80, 255]), 
         'colorLwThresh': np.array([120, 10, 0]), 'objL': 19, 'objW': 10, 'objH': 9, 
         'xUpOffset': 25, 'yUpOffset': 0, 'zUpOffset': 0, 
         'xLwOffset': 25, 'yLwOffset': 0, 'zLwOffset': 0,
         'thetaD': 0.056, 'thetaX': -0.134, 'thetaY': -0.003, 'theta': 43.018, 
         'cameraHeight': 1.5, 'jointStateDictHori': {'torso_lift_joint': 0.2}}

################################################################################

class fetchRobotClassObj(object):
    '''
    This class will incorporate all the basic movements that the fetch arm needs 
    to do, like auto-tuck, move to home position etc.
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

        # This is used for printing the current position of the arm.
        self.listenerGripperPose = tf.TransformListener()

        # Create move group interface for a fetch robot
        self.armMoveGroup = MoveGroupInterface('arm_with_torso', 'base_link')

        ## Define ground plane.
        ## This creates objects in the planning scene that mimic the ground.
        ## If these were not in place gripper could hit the ground.
        ## You have to add the 'ns' argument to the PlanningSceneInterface so that 
        ## if your robot has a ROS_NAMESPACE it can be appended to the planningScene
        ## nodes.
        #planningScene = PlanningSceneInterface('base_link', ns=self.namespace)
        #planningScene.removeCollisionObject('my_front_ground')
        #planningScene.removeCollisionObject('my_back_ground')
        #planningScene.removeCollisionObject('my_right_ground')
        #planningScene.removeCollisionObject('my_left_ground')
        #planningScene.addCube('my_front_ground', 2, 1.1, 0.0, -1.0)
        #planningScene.addCube('my_back_ground', 2, -1.2, 0.0, -1.0)
        #planningScene.addCube('my_left_ground', 2, 0.0, 1.2, -1.0)
        #planningScene.addCube('my_right_ground', 2, 0.0, -1.2, -1.0)

        # Arm joint names.
        self.armJointNames = ['torso_lift_joint', 'shoulder_pan_joint',
                               'shoulder_lift_joint', 'upperarm_roll_joint',
                               'elbow_flex_joint', 'forearm_roll_joint',
                               'wrist_flex_joint', 'wrist_roll_joint']
        
        # A value of 3 to the wrist_roll_joint rotates it to pi radians.
        # And value of 0 to the wrist_roll_joint rotates it to 0 radians.
        # A value of 0.4 to the torso_lift_joint raises it to 42 cm.
        # And value of 0 to the torso_lift_joint raises it to 0 cm.
        
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
        
##########----------------------------------------------------------------------

        # This is the wrist link not the gripper itself.
        self.wristFrame = 'wrist_roll_link'
        self.gripperFrame = 'gripper_link'
        
        self.gripperJointNames = ['l_gripper_finger_joint', 'r_gripper_finger_joint']

        # This is used for opening and closing the gripper.
        self._gripper_client = actionlib.SimpleActionClient(
            self.namespace + 'gripper_controller/gripper_action', GripperCommandAction)
        self._gripper_client.wait_for_server(rospy.Duration(10))

##########----------------------------------------------------------------------

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

##########----------------------------------------------------------------------

        rospy.Subscriber(self.namespace + 'ar_pose_marker', AlvarMarkers, 
                                         callback=self._arTag_callback)

        self.arTagsPoseList = []   # List to hold the AR tag markers.

##########----------------------------------------------------------------------

        # Name of all the joints including arm, head, wheels and gripper.
        self.allJointNames = self.armJointNames + self.headJointNames + \
                             self.gripperJointNames + ['r_wheel_joint', 
                                                  'l_wheel_joint', 'bellows_joint']

##########----------------------------------------------------------------------

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
        
##########----------------------------------------------------------------------

        # Subscribing to the battery_state topic to get the current battery voltage.
        rospy.Subscriber(self.namespace + 'battery_state', BatteryState, 
                                                     self._battery_state_callback)

        self.currentBattChargeLvl = 0.0    # Initializing battery charge level variable.

##########----------------------------------------------------------------------

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
        
##########----------------------------------------------------------------------

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

##########----------------------------------------------------------------------

        # Create a publisher which will publish the marker in the point cloud 
        # domain for an object to be picked up.
        self._obj_marker_pub = rospy.Publisher(self.namespace + 'visualization_marker', 
                                                Marker, queue_size=5)
        
##########----------------------------------------------------------------------

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

    def _odom_callback(self, msg):
        '''
        Callback function of the Subscriber. Updates the x, y coordinates of the robot.
        '''
        self.odom = msg.pose.pose

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

    def _arm_move_to_pose(self, pose_stamped, allowed_planning_time=10.0, 
                     execution_timeout=15.0, num_planning_attempts=1, 
                     orientation_constraint=None, plan_only=False, replan=False, 
                     replan_attempts=5, tolerance=0.01):
        '''
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

    def pixel3DcoordinateUsingFOV(self, x, y):
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
        # from there. So, instead we are taking a 50x50 rectangle around the point
        # and drawing the histogram of the depths in all the points of the rectangle.
        # The bin with the highest number of pixels with non-zero depths is the 
        # actual depth of the point.
        
        # Since the inputs will signify pixel coordinates here, so we have to 
        # make sure they dont go out of the bounds of the image. This may happen 
        # the 50x50 rectangle is partly outside the image boundaries.
        h, w, roiDlw, roiDup = 50, 50, 5, 250
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

    def moveSingleJoint(self, jointName, jointState=0.0, verbose=True):
        '''
        This function changes the state of a single arm joint keeping the 
        others the same.
        '''
        if jointName not in self.armJointNames:
            print('\nWrong joint name provided...')
            return
            
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

    def extractObjHoriAngle(self, obj=None, display=False):
        '''
        This function takes in the object (the objects are defined as global
        varibles) and then using the attributes of the objects extracts the 
        object from the image from the camera and calculates how much the angle
        of tilt is with the horizontal.
        '''
        # To extract the object from the image, it should be first placed in a 
        # proper orientation using the fetch arm.
        name, objId = obj['name'], obj['objId'] 
        headPanHori, headTiltHori = obj['headPanHori'], obj['headTiltHori']
        roiTlXhori, roiTlYhori = obj['roiTlXhori'], obj['roiTlYhori']
        roiBrXhori, roiBrYhori = obj['roiBrXhori'], obj['roiBrYhori']
        roiDlwHori, roiDupHori = obj['roiDlwHori'], obj['roiDupHori']
        jointStateDictHori = obj['jointStateDictHori']
               
        self.headPanHoriTilt(headPanHori, headTiltHori, verbose=False)    # Set the head.
        self.moveMultipleJoints(jointStateDictHori, verbose=False)    # Set arm.

        i, nSamples = 0, 100
        
        # This is a mask that will be used to find out the contour of the object.
        mask = np.zeros(self.depthImg.shape, dtype=np.uint8)
        mask = mask[roiTlYhori : roiBrYhori, roiTlXhori : roiBrXhori]

        # Show the converted image.
        while not rospy.is_shutdown() and i < nSamples:
            # Cropping out the portion of the image where the object is visible.
            colorImg = self.colorImg[roiTlYhori : roiBrYhori, roiTlXhori : roiBrXhori]
            depthImg = self.depthImg[roiTlYhori : roiBrYhori, roiTlXhori : roiBrXhori]
            imgH, imgW, _ = colorImg.shape
            
            # Clipping all values beyond the extent of the gripper (which may be 
            # like 0.9 m), such that the focus is only on the object gripped by 
            # the gripper. The depthImg pixels are in cm.
            depthImg = np.clip(depthImg, roiDlwHori, roiDupHori)
            
            # Making the pixels outside the desired depth range to 0.
            lwMask = depthImg != roiDupHori
            upMask = depthImg != roiDlwHori
            maskedDepthImg = np.asarray(depthImg * lwMask * upMask, dtype=np.uint8)
            
            # Now since there are always noisy contours appearing in the images,
            # so several of these masks are logically 'or'ed together to create 
            # a keep a wholistic set of contours in the final combined mask. And
            # after that the largest contour is extracted to calculate the angle.
            maskedDepthImg = cv2.bitwise_or(maskedDepthImg, mask)
            mask = copy.deepcopy(maskedDepthImg)
            
            # Eroding the contours.
            kernel = np.ones((5, 5), np.uint8)
            maskedDepthImg = cv2.erode(maskedDepthImg, kernel)

            # Colormapping the depth image.
            depthImgColorMap = cv2.applyColorMap(maskedDepthImg, cv2.COLORMAP_JET)
            
            i += 1
            
            if display:     # Display the images if this flag is True.
                cv2.imshow('Color Image', colorImg)
                cv2.imshow('Depth Image Colormapped', depthImgColorMap)
                key = cv2.waitKey(1)
                if key & 0xFF == 27:    break   # Break the loop with esc key.
            
        # Finding the largest contour.
        _, contours, _ = cv2.findContours(maskedDepthImg, cv2.RETR_TREE, 
                                                    cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        largestContour = contours[0]
        cv2.drawContours(depthImgColorMap, [largestContour], -1, (0,255,0), 2)
        
        # Finding the angle of rotation of the largest contour. That gives 
        # the angle of rotation of the object.
        rows, cols = depthImgColorMap.shape[:2]
        vx, vy, x, y = cv2.fitLine(largestContour, cv2.DIST_L2, 0, 0.01, 0.01)
        leftY = int((-x*vy/vx) + y)
        rightY = int(((cols-x)*vy/vx) + y)
        cv2.line(depthImgColorMap, (cols-1,rightY), (0,leftY), (0,255,255), 2)
        
        avgAngleHori = np.rad2deg(-1*vy/vx)
        print(avgAngleHori)
        
        if display:     # Display the images if this flag is True.
            cv2.imshow('Color Image', colorImg)
            cv2.imshow('Depth Image Colormapped', depthImgColorMap)
            cv2.waitKey(0)
        
        return colorImg, depthImg, depthImgColorMap, avgAngleHori

################################################################################

    def extractBlueBin(self, obj=None, display=False):
        '''
        This function takes in the object (the objects are defined as global
        varibles) and then using the attributes of the objects extracts the 
        object part of the image.
        Works best when the bins are 10cm from the outer edge of the table and 
        the robot base is about 30cm from the outer edge of the table.
        '''
        # To extract the object from the image, it should be first placed in a 
        # proper orientation using the fetch arm.
        name, objId = obj['name'], obj['objId']
        headPanHori, headTiltHori = obj['headPanHori'], obj['headTiltHori']
        roiTlXhori, roiTlYhori = obj['roiTlXhori'], obj['roiTlYhori']
        roiBrXhori, roiBrYhori = obj['roiBrXhori'], obj['roiBrYhori']
        roiDlwHori, roiDupHori = obj['roiDlwHori'], obj['roiDupHori']
        colorUpThresh, colorLwThresh = obj['colorUpThresh'], obj['colorLwThresh']
        jointStateDictHori, cameraHeight = obj['jointStateDictHori'], obj['cameraHeight']
        objL, objW, objH = obj['objL'], obj['objW'], obj['objH']
        xUpOffset, yUpOffset, zUpOffset = obj['xUpOffset'], obj['yUpOffset'], obj['zUpOffset']
        xLwOffset, yLwOffset, zLwOffset = obj['xLwOffset'], obj['yLwOffset'], obj['zLwOffset']
        thetaD, thetaX = obj['thetaD'], obj['thetaX']
        thetaY, theta = obj['thetaY'], obj['theta']
               
        self.headPanHoriTilt(headPanHori, headTiltHori, verbose=False)    # Set the head.
        self.moveMultipleJoints(jointStateDictHori, verbose=False)    # Set arm.

        i, nSamples = 0, 100

        # This is a mask that will be used to find out the contour of the object.
        mask = np.zeros(self.depthImg.shape, dtype=np.uint8)
        mask = mask[roiTlYhori : roiBrYhori, roiTlXhori : roiBrXhori]
        
        # Show the converted image.
        while not rospy.is_shutdown() and i < nSamples:
            # Cropping out the portion of the image where the object is visible.
            colorImg = self.colorImg[roiTlYhori : roiBrYhori, roiTlXhori : roiBrXhori]
            depthImg = self.depthImg[roiTlYhori : roiBrYhori, roiTlXhori : roiBrXhori]
            imgH, imgW, _ = colorImg.shape
            
            # Clipping all values beyond the extent of the gripper (which may be 
            # like 0.9 m), such that the focus is only on the object gripped by 
            # the gripper. The depthImg pixels are in cm.
            depthImg = np.clip(depthImg, roiDlwHori, roiDupHori)
            
            # Making the pixels outside the desired depth range to 0.
            lwMask = depthImg != roiDupHori
            upMask = depthImg != roiDlwHori
            maskedDepthImg = np.asarray(depthImg * lwMask * upMask, dtype=np.uint8)
            
            # Now since there are always noisy contours appearing in the images,
            # so several of these masks are logically 'or'ed together to create 
            # a keep a wholistic set of contours in the final combined mask.
            maskedDepthImg = cv2.bitwise_or(maskedDepthImg, mask)
            mask = copy.deepcopy(maskedDepthImg)
            
            # Eroding the contours.
            kernel = np.ones((5, 5), np.uint8)
            maskedDepthImg = cv2.erode(maskedDepthImg, kernel)

            # Colormapping the depth image.
            depthImgColorMap = cv2.applyColorMap(maskedDepthImg, cv2.COLORMAP_JET)
            
            i += 1
            
            if display:     # Display the images if this flag is True.
                cv2.imshow('Color Image', colorImg)
                cv2.imshow('Depth Image Colormapped', depthImgColorMap)
                key = cv2.waitKey(1)
                if key & 0xFF == 27:    break   # Break the loop with esc key.
            
        # Masking the unwanted part of the color image with the depth masks.
        lwMaskForColorImg = np.dstack((lwMask, lwMask, lwMask))
        upMaskForColorImg = np.dstack((upMask, upMask, upMask))
        maskedColorImg = np.asarray(colorImg * lwMaskForColorImg * upMaskForColorImg, 
                                                                    dtype=np.uint8)
        
        # Filtering the maskedColorImg with the color range of the object.
        filteredImg = cv2.inRange(maskedColorImg, colorLwThresh, colorUpThresh)
        # Dilating to fill up gaps and overlap the objects inside the bins.
        dilateKernel = np.ones((20, 20), np.uint8)
        filteredImg = cv2.dilate(filteredImg, dilateKernel)
        erodeKernel = np.ones((10, 10), np.uint8)   # Eroding to remove noise.
        filteredImg = cv2.erode(filteredImg, erodeKernel)
        
        # Finding the largest contour.
        _, contours, _ = cv2.findContours(filteredImg, cv2.RETR_TREE, 
                                                            cv2.CHAIN_APPROX_SIMPLE)
                
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # In some cases the contours are segmented into smaller parts. In 
        # those cases there will be multiple adjacent smaller contours. To combine 
        # all of these they are bounded by bounding boxes and the combined contour 
        # of all those bounding boxes is returned. For that, drawing the filled 
        # boxes in a blank image.
        blankImg = np.zeros((colorImg.shape[0], colorImg.shape[1]), dtype=np.uint8)
        for c in contours:
            x,y,w,h = cv2.boundingRect(c)
            blankImg = cv2.rectangle(blankImg, (x,y), (x+w,y+h), 255, -1)
        
        # Finding the final combined contours.
        _, finalContours, _ = cv2.findContours(blankImg, cv2.RETR_TREE, 
                                                            cv2.CHAIN_APPROX_SIMPLE)
        
        # Making a sorted list of the arc lengths of the contours detected.
        # Then taking the difference between the consecutive entries to find the 
        # max gap in the length values of the contours.
        arcLengthList = [cv2.arcLength(c,True) for c in finalContours]
        arcLengthList.sort(reverse=True)
        lengthDiffList = [arcLengthList[l]-arcLengthList[l+1] \
                                            for l in range(len(arcLengthList)-1)]
        # Sometimes only one contours is detected, in that case use that one only.
        largestGap = max(lengthDiffList) if len(lengthDiffList) > 0 else 0
        largestContours = [c for c in finalContours if cv2.arcLength(c,True) > largestGap]
        
        # Now in case if the desired object is not present in front of the camera, 
        # some unwanted object can be misclassified and given as output. Filtering 
        # those out using the size of the contours.
        largestContours = [c for c in largestContours if cv2.contourArea(c) > 5000] 
        
        locationOfObjs = []
        for c in range(len(largestContours)):
            # If valid contours are found only then do the following.
            largestContours = np.array(largestContours)
            x,y,w,h = cv2.boundingRect(largestContours[c])
            boxImg = depthImg[y:y+h, x:x+w]
            # Creating a histogram of the depth pixel values from the depth image in 
            # the region of the largestContours which is basically where the objects are.
            avgDist = np.sum(boxImg)/w/h
            boxPixelList = boxImg.reshape(boxImg.shape[0]*boxImg.shape[1]).tolist()
            histVals, histBins = np.histogram(boxPixelList, bins=list(range(roiDlwHori, roiDupHori)))
            #plt.hist(boxImg.ravel(), (roiDupHori-roiDlwHori), [roiDlwHori,roiDupHori])
            #plt.show()
            #print(histVals, histBins)
            
            # Sorting the histogram value in descending order and the bins are also 
            # sorted accordingly.
            histValsSorted, histBinsSorted = zip(*sorted(zip(histVals, histBins), \
                                                    key=lambda x: x[0], reverse=True))
            # Taking the bin having the max number of pixels as the approximate 
            # distance of the bin from the camera.
            approxDist = histBinsSorted[0]
            
            finalDist = avgDist
            print(finalDist)
            
            # Determining the vertical and horizontal distance of the bin from camera.
            vertDistOfObj = finalDist * np.sin(np.deg2rad(headTiltHori))
            horiDistOfObj = finalDist * np.cos(np.deg2rad(headTiltHori))
            sideDistOfObj = thetaD * finalDist + thetaX * (x+w/2) + thetaY * (y+h/2) + theta
            
            # Now putting upper and lower limits on the extent of the object, adding
            # some standard offsets and also converting the dimensions into meters 
            # (as they are in cm right now).
            xUpLim = float((horiDistOfObj+objL+xUpOffset)*0.01)
            xLwLim = float((horiDistOfObj-objL+xLwOffset)*0.01)
            yUpLim = float((sideDistOfObj+objW)*0.01) 
            yLwLim = float((sideDistOfObj-objW)*0.01)
            zUpLim = float((vertDistOfObj+objH+zUpOffset)*0.01) 
            zLwLim = float((vertDistOfObj-objH+zLwOffset)*0.01)
            
            # Keep in mind that the z distance measured in this case, if from the 
            # camera upto the bin, which is at a lower distance from the camera 
            # as the camera is looking down. So this measurement is actually upside 
            # down. So this has to be subtracted from the robot height and flipped.
            # And only then they are in the proper +z direction in the world frame.
            zUpLim, zLwLim = cameraHeight - zLwLim, cameraHeight - zUpLim
            
            locationOfObjs.append([xLwLim, yLwLim, zLwLim, xUpLim, yUpLim, zUpLim])
            #print(yUpLim, x+w/2, y+h/2, w, h)
            
            #if display:     # Display the images if this flag is True.
                #cv2.drawContours(colorImg, [largestContours], -1, (0,255,0), 2)
                #cv2.imshow('Color Image', colorImg)
                #cv2.imshow('Masked Color Image', maskedColorImg)
                #cv2.imshow('Filtered Image', filteredImg)
                #cv2.imshow('Blank Image', blankImg)
                #cv2.imshow('Depth Image Colormapped', depthImgColorMap)
                #cv2.waitKey(0)

        return depthImgColorMap, blankImg, colorImg, depthImg, largestContours, locationOfObjs

################################################################################



if __name__ == '__main__':
    
    fetchRobot = fetchRobotClassObj()

###########------------------------Testing all functions----------------------------------------------    

    #fetchRobot.safelyUntuckArm()

    ## Position and rotation of two 'wave end poses'
    #gripperPoseList = [Pose(Point(0.961, 0.001, 0.798), Quaternion(0.008, -0.005, 0.000, 0.999)), 
                        #Pose(Point(0.042, 0.384, 1.826), Quaternion(0.173, -0.693, -0.242, 0.657)),
                        #Pose(Point(0.047, 0.545, 1.822), Quaternion(-0.274, -0.701, 0.173, 0.635)),
                        #Pose(Point(0.860, 0.040, 1.033), Quaternion(0.313, -0.139, 0.195, 0.918))]
    #fetchRobot.armFollowTrajectory(gripperPoseList)
    
    ## Position and rotation of two 'wave end poses'
    #gripperPoseListWithoutOri = [Point(0.961, 0.001, 0.798), Point(0.860, 0.040, 1.033), 
                                  #Point(0.961, 0.001, 0.798), Point(0.860, 0.040, 1.033)]
    #fetchRobot.armFollowTrajectoryWithoutOri(gripperPoseListWithoutOri)
    
    #fetchRobot.armGotoHomePose()
    
    ####gripperTargetPose = Pose(Point(0.042, 0.384, 1.826), Quaternion(0.173, -0.693, -0.242, 0.657))
    #gripperTargetPose = Pose(Point(0.961, 0.001, 0.798), Quaternion(0.008, -0.005, 0.000, 0.999))
    
    #posePossible = fetchRobot.gripperToPose(gripperTargetPose)
    #print(posePossible)
    
    #posePossible = fetchRobot.gripperToPose(Point(0.860, 0.040, 1.033))
    #print(posePossible)
    
    #print(fetchRobot.currentGripperPose)
    #print(fetchRobot.currentArmJointStates)
    #print(fetchRobot.currentBattChargeLvl)
    
    #fetchRobot.gripperClose()
    #fetchRobot.gripperOpen()

    #fetchRobot.waveArm(howLongToWave=3)
    
    #fetchRobot.headLookAt(1.0, -0.5, -0.5)
    #fetchRobot.headPanHoriTilt(0, 0)
    
    #fetchRobot.autoTuckArm()
    
    #fetchRobot.baseGoForward(0.5)
    #fetchRobot.baseGoForward(-0.5)
        
    #fetchRobot.turnBase(60 * np.pi / 180)
    #fetchRobot.turnBaseAlternate(-60 * np.pi / 180)
    
###########-----------------------Pick up crank arm from cmm machine 3d printed stand (old design)-----------------------------------------------    
    
    #fetchRobot.headPanHoriTilt(0, 10, verbose=False)
    #fetchRobot.moveSingleJoint('torso_lift_joint', 0.1, verbose=False)
    #fetchRobot.gripperOpen(verbose=False)
    
    #tagsFound, arTagPose = fetchRobot.findARtags(maxAttempts=3, verbose=False)
    
    #if tagsFound:        
        ## Putting arm to start position.
        #fetchRobot.gripperToPose(Point(0.5, 0.5, 0.75), verbose=False)
        
        #arTagY = arTagPose.pose.position.y
        
        ## The robot rotates to make the y offset the minimum.
        #while abs(arTagY) > 0.1:
            #angleOfTurn = -5 if arTagY < 0 else 5
            #fetchRobot.turnBase(angleOfTurn * np.pi/180, speed=0.1, verbose=False)
            #msg = rospy.wait_for_message(fetchRobot.namespace + 'ar_pose_marker', AlvarMarkers)
            #arTagY = fetchRobot.arTagsPoseList[0].pose.position.y
            ##print(arTagY)

        #tagsFound, arTagPose = fetchRobot.findARtags(maxAttempts=3)
        #print(arTagPose.pose.position)

        ## If the arTag is beyond a certain distance (like 1 m) from the robot, 
        ## then it moves forward so that it is withing 1.0 meters and hence within reach.        
        #fetchRobot.baseGoForward(arTagPose.pose.position.x - 1.0, speed=0.1, verbose=False)
        
        #tagsFound, arTagPose = fetchRobot.findARtags(maxAttempts=3)
        #print(arTagPose.pose.position)

        #position = Point(arTagPose.pose.position.x - 0.2, arTagPose.pose.position.y, 
                                     #arTagPose.pose.position.z + 0.21)
        #fetchRobot.gripperToPose(position, verbose=False)
        
        ## Now go forward and grasp the object.
        #fetchRobot.baseGoForward(0.25, speed=0.05, verbose=False)
        #fetchRobot.gripperClose(gap=0.01, verbose=False)
        
        ## Now go backwards.
        #fetchRobot.baseGoForward(-0.5, speed=0.1, verbose=False)
        
###########------------------Pick and place crank arm on cardboard stand on table----------------------------------------------------    
    
    #fetchRobot.moveSingleJoint('torso_lift_joint', 0.2, verbose=False)
    #fetchRobot.headPanHoriTilt(0, 10, verbose=False)
    #fetchRobot.gripperOpen(verbose=False)
    
    #tagsFound, arTagPose = fetchRobot.findARtags(maxAttempts=3, verbose=False)
    
    #if tagsFound:        
        ## Putting arm to start position.
        #fetchRobot.gripperToPose(Point(0.5, 0.5, 0.75), verbose=False)
        
        #arTagY = arTagPose.pose.position.y
        
        ## The robot rotates to make the y offset the minimum.
        #while abs(arTagY) > 0.1:
            #angleOfTurn = -5 if arTagY < 0 else 5
            #fetchRobot.turnBase(angleOfTurn * np.pi/180, speed=0.1, verbose=False)
            #msg = rospy.wait_for_message(fetchRobot.namespace + 'ar_pose_marker', AlvarMarkers)
            #arTagY = fetchRobot.arTagsPoseList[0].pose.position.y
            ##print(arTagY)

        #tagsFound, arTagPose = fetchRobot.findARtags(maxAttempts=3)
        ##print(arTagPose.pose.position)

        ## If the arTag is beyond a certain distance (like 1 m) from the robot, 
        ## then it moves forward so that it is withing 1.0 meters and hence within reach.        
        #fetchRobot.baseGoForward(arTagPose.pose.position.x - 1.0, speed=0.1, verbose=False)
        
        #tagsFound, arTagPose = fetchRobot.findARtags(maxAttempts=3)
        #print(arTagPose.pose.position)

        #position = Point(arTagPose.pose.position.x - 0.2, arTagPose.pose.position.y, 
                                     #arTagPose.pose.position.z + 0.21)
        #fetchRobot.gripperToPose(position, verbose=False)
        
        ## Now go forward and grasp the object.
        #fetchRobot.baseGoForward(0.25, speed=0.05, verbose=False)
        #fetchRobot.gripperClose(gap=0.01, verbose=False)
        
        ## Now go backwards.
        #fetchRobot.moveSingleJoint('torso_lift_joint', 0.3, verbose=False)
        #fetchRobot.baseGoForward(-0.5, speed=0.1, verbose=False)

        ## Putting arm to start position while holding the object.
        #fetchRobot.gripperToPose(Point(0.5, 0.5, 0.75), verbose=False)
        #fetchRobot.moveSingleJoint('torso_lift_joint', 0.1, verbose=False)
        #pickedUp = True

    ## Putting the object back to place.
    
    #if pickedUp:
        #fetchRobot.moveSingleJoint('torso_lift_joint', 0.2, verbose=False)
        #fetchRobot.headPanHoriTilt(0, 10, verbose=False)
        
        #tagsFound, arTagPose = fetchRobot.findARtags(maxAttempts=3, verbose=False)
            
        #if tagsFound:        
            ## Putting arm to start position.
            #fetchRobot.gripperToPose(Point(0.5, 0.5, 0.75), verbose=False)
            
            #arTagY = arTagPose.pose.position.y
            
            ## The robot rotates to make the y offset the minimum.
            #while abs(arTagY) > 0.1:
                #angleOfTurn = -5 if arTagY < 0 else 5
                #fetchRobot.turnBase(angleOfTurn * np.pi/180, speed=0.1, verbose=False)
                #msg = rospy.wait_for_message(fetchRobot.namespace + 'ar_pose_marker', AlvarMarkers)
                #arTagY = fetchRobot.arTagsPoseList[0].pose.position.y
                ##print(arTagY)

            #tagsFound, arTagPose = fetchRobot.findARtags(maxAttempts=3)
            ##print(arTagPose.pose.position)

            ## If the arTag is beyond a certain distance (like 1 m) from the robot, 
            ## then it moves forward so that it is withing 1.0 meters and hence within reach.        
            #fetchRobot.baseGoForward(arTagPose.pose.position.x - 1.0, speed=0.1, verbose=False)
            
            #tagsFound, arTagPose = fetchRobot.findARtags(maxAttempts=3)
            #print(arTagPose.pose.position)

            #position = Point(arTagPose.pose.position.x - 0.2, arTagPose.pose.position.y, 
                                         #arTagPose.pose.position.z + 0.21)
            #fetchRobot.gripperToPose(position, verbose=False)

            ## Lift the torso to avoid collision with the stand.
            #fetchRobot.moveSingleJoint('torso_lift_joint', 0.32, verbose=False)
            
            ## Now go forward and grasp the object.
            #fetchRobot.baseGoForward(0.25, speed=0.05, verbose=False)
            #fetchRobot.moveSingleJoint('torso_lift_joint', 0.26, verbose=False)
            #fetchRobot.gripperOpen(verbose=False)
            #fetchRobot.moveSingleJoint('torso_lift_joint', 0.24, verbose=False)
            
            ## Now go backwards.
            #fetchRobot.baseGoForward(-0.5, speed=0.1, verbose=False)
            #fetchRobot.moveSingleJoint('torso_lift_joint', 0.1, verbose=False)
            #fetchRobot.autoTuckArm(verbose=False)
        
############------------------Pick and place a blue bin on table----------------------------------------------------    
    
    #fetchRobot.moveSingleJoint('torso_lift_joint', 0.2, verbose=False)
    #fetchRobot.headPanHoriTilt(0, 10, verbose=False)
    #fetchRobot.gripperOpen(verbose=False)
    
    #tagsFound, arTagPose = fetchRobot.findARtags(maxAttempts=3, verbose=False)
    
    #if tagsFound:        
        ## Putting arm to start position.
        #fetchRobot.gripperToPose(Point(0.5, 0.5, 0.75), verbose=False)
        
        #arTagY = arTagPose.pose.position.y
        
        ## The robot rotates to make the y offset the minimum.
        #while abs(arTagY) > 0.1:
            #angleOfTurn = -5 if arTagY < 0 else 5
            #fetchRobot.turnBase(angleOfTurn * np.pi/180, speed=0.1, verbose=False)
            #msg = rospy.wait_for_message(fetchRobot.namespace + 'ar_pose_marker', AlvarMarkers)
            #arTagY = fetchRobot.arTagsPoseList[0].pose.position.y
            ##print(arTagY)

        #tagsFound, arTagPose = fetchRobot.findARtags(maxAttempts=3)
        ##print(arTagPose.pose.position)

        ## If the arTag is beyond a certain distance (like 1 m) from the robot, 
        ## then it moves forward so that it is withing 1.0 meters and hence within reach.        
        #fetchRobot.baseGoForward(arTagPose.pose.position.x - 0.9, speed=0.1, verbose=False)
        
        #tagsFound, arTagPose = fetchRobot.findARtags(maxAttempts=3)
        #print(arTagPose.pose.position)
        #print(arTagPose.pose.orientation)
        #fetchRobot._yaw_from_quaternion(arTagPose.pose.orientation)

        #position = Point(arTagPose.pose.position.x - 0.2, arTagPose.pose.position.y - 0.05, 
                                     #arTagPose.pose.position.z)

        ## Making the gripper vertical to pick the bin from the top.
        #rotatedOrientation = tf.transformations.quaternion_from_euler(0, np.deg2rad(90), 0)
        #fetchRobot.gripperToPose(Pose(position, Quaternion(*rotatedOrientation)), verbose=False)
        #print(fetchRobot.currentArmJointStates)

        ## Raise the arm, move forward, lower torso, grip, raise torso and move back.
        #fetchRobot.moveSingleJoint('shoulder_lift_joint', -0.8, verbose=False)
        #fetchRobot.baseGoForward(0.3, speed=0.05, verbose=False)        
        #fetchRobot.moveSingleJoint('torso_lift_joint', 0.1, verbose=False)
        #fetchRobot.gripperClose(gap=0.0, verbose=False)
        #fetchRobot.moveSingleJoint('torso_lift_joint', 0.2, verbose=False)
        #fetchRobot.baseGoForward(-0.5, speed=0.1, verbose=False)
        #pickedUp = True

###########----------------------------------------------------------------------

    #fetchRobot.baseGoForward(0.2, speed=0.05, verbose=False)

    #tagsFound, arTagPose = fetchRobot.findARtags(maxAttempts=3, verbose=False)

    #if tagsFound:
        #print(arTagPose.pose)

    ### Show the converted image.
    ##while not rospy.is_shutdown():
        ##colorImg = copy.deepcopy(fetchRobot.colorImg)
        ##depthImgColorMap = copy.deepcopy(fetchRobot.depthImgColorMap)
        ##cv2.imshow('Color Image', colorImg)
        ##cv2.imshow('Depth Image', depthImgColorMap)
        ##key = cv2.waitKey(1)
        ##if key & 0xFF == 27:    break   # Break the loop with esc key.

    #if tagsFound:        
        #arTagY = arTagPose.pose.position.y
        
        ## The robot rotates to make the y offset the minimum.
        #while abs(arTagY) > 0.1:
            #angleOfTurn = -5 if arTagY < 0 else 5
            #fetchRobot.turnBase(angleOfTurn * np.pi/180, speed=0.1, verbose=False)
            #msg = rospy.wait_for_message(fetchRobot.namespace + 'ar_pose_marker', AlvarMarkers)
            #arTagY = fetchRobot.arTagsPoseList[0].pose.position.y
            #print(arTagY)
        
        #tagsFound, arTagPose = fetchRobot.findARtags(maxAttempts=3)
        #print(arTagPose.pose.position)

        ## If the arTag is beyond a certain distance (like 1 m) from the robot, 
        ## then it moves forward so that it is withing 1.0 meters and hence within reach.        
        #fetchRobot.baseGoForward(arTagPose.pose.position.x - 1.0, speed=0.1, verbose=False)
        
        #tagsFound, arTagPose = fetchRobot.findARtags(maxAttempts=3)
        #print(arTagPose.pose.position)
        
        #position = Point(arTagPose.pose.position.x - 0.2, arTagPose.pose.position.y, 
                                     #arTagPose.pose.position.z + 0.27)
        #fetchRobot.gripperToPose(position, verbose=False)
        
        ## Now go forward and grasp the object.
        #fetchRobot.baseGoForward(0.25, speed=0.05, verbose=False)
        #fetchRobot.gripperOpen(verbose=False)
        
        ## Now go backwards.
        #fetchRobot.baseGoForward(-0.5, speed=0.1, verbose=False)
    
###########----------------------------------------------------------------------

    ## Show the converted image.
    #fetchRobot.moveSingleJoint('wrist_roll_joint', 0.5, verbose=False)
    #colorImg, depthImg, depthImgColorMap, avgAngleHori = fetchRobot.extractObjHoriAngle(obj1, display=True)
    #cv2.imshow('Color Image', colorImg)
    #cv2.imshow('Depth Image Colormapped', depthImgColorMap)
    #cv2.waitKey(0)
    

#########----------------------------------------------------------------------

    ### Show the image.
    ##key = '`'
    ##while True:
        ##cv2.imshow('Color Image', fetchRobot.colorImg)
        ##key = cv2.waitKey(1)
        ##if key & 0xFF == 27:    break   # esc to break.
    
    ## Show the converted image.
    #depthImgColorMap, blankImg, colorImg, depthImg, largestContours, \
                    #locationOfObjs = fetchRobot.extractBlueBin(obj2, display=False)
    
    ##colorImg = cv2.rectangle(colorImg, (260,180), (320,290), (0,0,255), 2)
    ##boxImg = colorImg[180:290, 260:320, 0]
    ##plt.hist(boxImg.ravel(),256,[0,256])
    ##plt.show()
    ##colorUpThresh, colorLwThresh, = np.array([220, 80, 255]), np.array([120, 10, 0])
    ##filteredImg = cv2.inRange(maskedColorImg, colorLwThresh, colorUpThresh)
    
    ##for c in largestContours:   print(cv2.contourArea(c))
    ##print(locationOfObjs)
    
    #cv2.drawContours(colorImg, largestContours, -1, (0,255,0), 2)
        
    ##cv2.imshow('Filtered Image', filteredImg)
    ##cv2.imshow('Blank Image', blankImg)
    #cv2.imshow('Color Image', colorImg)
    ##cv2.imshow('Masked Color Image', maskedColorImg)
    ##cv2.imshow('Depth Image Colormapped', depthImgColorMap)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    #for l in locationOfObjs:
        #xLwLim, yLwLim, zLwLim, xUpLim, yUpLim, zUpLim = l[0], l[1], l[2], l[3], l[4], l[5]      #(0.7160336329635573, -0.10641190771349862, 1.0601063188705235, 1.0960336329635572, 0.09358809228650138, 1.2401063188705235)
        ##xLwLim, yLwLim, zLwLim, xUpLim, yUpLim, zUpLim = 0.7, -0.1, 1.05, 1.0, 0.1, 1.3
        #print(xLwLim, yLwLim, zLwLim, xUpLim, yUpLim, zUpLim)
        
        ## Setting the point cloud cropping parameters using these values. 
        ## If these parameters are not defined earlier, they will be defined, 
        ## else they will be updated.
        #rospy.set_param('crop_min_x', xLwLim)
        #rospy.set_param('crop_min_y', yLwLim)
        #rospy.set_param('crop_min_z', zLwLim)
        #rospy.set_param('crop_max_x', xUpLim)
        #rospy.set_param('crop_max_y', yUpLim)
        #rospy.set_param('crop_max_z', zUpLim)
        #print('Cropping parameters updated...')
        
        #print('Running c++ file for processing the point cloud...')
        #subprocess.call(["rosrun", "perceptionRealPcl", "extract_blue_bin_pose"])
        
        ## Updating the parameter values into some variables. These are created 
        ## by the c++ file.
        #objPosiX = rospy.get_param(obj2['name']+'_posiX')
        #objPosiY = rospy.get_param(obj2['name']+'_posiY')
        #objPosiZ = rospy.get_param(obj2['name']+'_posiZ')
        #objOriX = rospy.get_param(obj2['name']+'_oriX')
        #objOriY = rospy.get_param(obj2['name']+'_oriY')
        #objOriZ = rospy.get_param(obj2['name']+'_oriZ')
        #objOriW = rospy.get_param(obj2['name']+'_oriW')
        #objScaleX = rospy.get_param(obj2['name']+'_scaleX')
        #objScaleY = rospy.get_param(obj2['name']+'_scaleY')
        #objScaleZ = rospy.get_param(obj2['name']+'_scaleZ')
        
        ## Deleting the parameters.
        #rospy.delete_param('crop_min_x'); rospy.delete_param('crop_min_y')
        #rospy.delete_param('crop_min_z'); rospy.delete_param('crop_max_x')
        #rospy.delete_param('crop_max_y'); rospy.delete_param('crop_max_z')
        #print('Cropping parameters deleted...')
        #rospy.delete_param(obj2['name']+'_posiX'); rospy.delete_param(obj2['name']+'_posiY')
        #rospy.delete_param(obj2['name']+'_posiZ'); rospy.delete_param(obj2['name']+'_oriX')
        #rospy.delete_param(obj2['name']+'_oriY'); rospy.delete_param(obj2['name']+'_oriZ')
        #rospy.delete_param(obj2['name']+'_oriW'); rospy.delete_param(obj2['name']+'_scaleX')
        #rospy.delete_param(obj2['name']+'_scaleY'); rospy.delete_param(obj2['name']+'_scaleZ')
        #print('Parameters to store marker pose deleted...')

        ## Using the information about the marker to create a marker and publish 
        ## it so that it can be visualized in rviz.
        #objMarker = Marker()
        #objMarker.ns = obj2['name']
        #objMarker.header.frame_id = fetchRobot.refFrame    # self.refFrame = 'base_link'.
        #objMarker.type = Marker.CUBE
        #objMarker.pose.position.x = objPosiX
        #objMarker.pose.position.y = objPosiY
        #objMarker.pose.position.z = objPosiZ
        #objMarker.pose.orientation.x = objOriX
        #objMarker.pose.orientation.y = objOriY
        #objMarker.pose.orientation.z = objOriZ
        #objMarker.pose.orientation.w = objOriW
        #objMarker.scale.x = objScaleX
        #objMarker.scale.y = objScaleY
        #objMarker.scale.z = objScaleZ
        #objMarker.color.g, objMarker.color.a = 1, 0.3
        
        #startTime = time.time()
        #while not rospy.is_shutdown():
            #fetchRobot._obj_marker_pub.publish(objMarker)    # Publish Marker.
            #char = fetchRobot.getch()
            #if ord(char) & 0xFF == 27:   break      # Break if 'q' is pressed.
            ##if time.time() - startTime > 3:     break   # Break after 3 seconds.
        
        ## Setting the position and orientation for the gripper.
        ## The gripper is positioned such that it can hold one of the side walls 
        ## of the blue bin. Hence there is some offset along the y axis.
        #position = Point(objPosiX, objPosiY + objScaleY/2, objPosiZ + objScaleZ)
        #orientation = Quaternion(objOriX, objOriY, objOriZ, objOriW)

        ## Putting arm to start position.
        #fetchRobot.gripperToPose(Point(0.5, 0.75, 1.3), verbose=False)
        #fetchRobot.gripperOpen(verbose=False)

        ## We want to grab the bin from the top vertically. So fixing an orientation
        ## rotated along the y axis and then moving the arm.
        #rotatedOrientation = tf.transformations.quaternion_from_euler(0, np.deg2rad(45), 0)
        #rotatedOrientation = Quaternion(*rotatedOrientation)
        #fetchRobot.gripperToPose(Pose(position, rotatedOrientation), verbose=False)

        ## Tilting the wrist more and grab the bin by the sidewall.
        #position = Point(objPosiX, objPosiY + objScaleY/2, objPosiZ + objScaleZ/2)
        #rotatedOrientation = tf.transformations.quaternion_from_euler(0, np.deg2rad(60), 0)
        #rotatedOrientation = Quaternion(*rotatedOrientation)
        #fetchRobot.gripperToPose(Pose(position, rotatedOrientation), verbose=False)
        #fetchRobot.gripperClose(verbose=False)
        
        ## Now pick it up by increasing the position along the z axis by some amount
        ## and keeping the orientation the same.
        #position = Point(objPosiX, objPosiY + objScaleY/2, objPosiZ + objScaleZ)
        #rotatedOrientation = tf.transformations.quaternion_from_euler(0, np.deg2rad(60), 0)
        #rotatedOrientation = Quaternion(*rotatedOrientation)
        #fetchRobot.gripperToPose(Pose(position, rotatedOrientation), verbose=False)
        
        ## Now move the arm to the side by rotating 90 degrees along the z axis.
        #fetchRobot.moveSingleJoint('shoulder_pan_joint', 1.2, verbose=False)

        
        ### Making the gripper vertical to pick the bin from the top.
        ##fetchRobot.gripperToPose(Pose(position, Quaternion(*rotatedOrientation)), verbose=False)
        ####print(fetchRobot.currentArmJointStates)

#########-------------------------------------------------------------------

    #hello = ai.constant("hello TensorFlow!")
    #sess = ai.Session()
    #print(sess.run(hello))

##########--------------------------Recording the images for dataset creation for object detection--------------------------------------------

    ## Show the converted image.
    #i = 72
    #colorImgSaveLoc = './object_images/color_images/'
    #depthImgSaveLoc = './object_images/depth_images/'
    #depthImgColorMapSaveLoc = './object_images/depth_images_color_map/'
    #while not rospy.is_shutdown():
        #colorImg = copy.deepcopy(fetchRobot.colorImg)
        #depthImg = copy.deepcopy(fetchRobot.depthImg)
        #depthImgColorMap = copy.deepcopy(fetchRobot.depthImgColorMap)
        #cv2.imshow('Color Image', colorImg)
        #cv2.imshow('Depth Image', depthImg)
        #cv2.imshow('Depth Color Map', depthImgColorMap)
        #key = cv2.waitKey(1)
        #if key & 0xFF == 27:    break   # Break the loop with esc key.
        #elif key & 0xFF == ord('s'):      # Images will be saved.
            #i += 1
            #timestamp = time.time()
            #colorImgName = 'color_{}.png'.format(i)
            #depthImgName = 'depth_{}.png'.format(i)
            #depthImgColorMapName = 'depthColorMap_{}.png'.format(i)
            #cv2.imwrite(os.path.join(colorImgSaveLoc, colorImgName), colorImg)
            #cv2.imwrite(os.path.join(depthImgSaveLoc, depthImgName), depthImg)
            #cv2.imwrite(os.path.join(depthImgColorMapSaveLoc, depthImgColorMapName), depthImgColorMap)
            #print('Image {} saved...'.format(i))
            
##########--------------------------Testing the neural netowrk--------------------------------------------

    fetchRobot.gripperOpen(verbose=False)
    fetchRobot.gripperToPose(Point(0.3, -0.6, 1.0), verbose=False)
    fetchRobot.moveSingleJoint('torso_lift_joint', 0.4, verbose=False)
    fetchRobot.gripperToPose(Point(0.5, -0.75, 1.1), verbose=False)      # Putting arm to start position.
    #fetchRobot.baseGoForward(0.4, speed=0.1, verbose=False)
    headTiltHori, headPanHori, cameraHeight = 40, 0, 1
    fetchRobot.headPanHoriTilt(0, headTiltHori, verbose=False)
    
    key = ord('`')
    detector = networkDetector()

    # Show the converted image.
    while not rospy.is_shutdown():
        # Prediction from network.
        colorImg = copy.deepcopy(fetchRobot.colorImg)
        depthImg = copy.deepcopy(fetchRobot.depthImg)
        depthImgColorMap = copy.deepcopy(fetchRobot.depthImgColorMap)
        
        img = copy.deepcopy(colorImg)
        
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
                
#-------------------------------------------------------------------------------

        # Draw the detected results now.
        for pdx, p in enumerate(detectedBatchClassNames):
            x, y, w, h = detectedBatchBboxes[pdx].tolist()

            # Only draw the bounding boxes for the non-rbc entities.
            cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 2)
            cv2.putText(img, p, (x+5, y+15), \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)
            cv2.circle(img, (int(x+w/2), int(y+h/2)), 2, (0,0,255), 2)
            cv2.circle(depthImgColorMap, (int(x+w/2), int(y+h/2)), 2, (0,0,255), 2)
            
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
            print(p, center3D)

            score = detectedBatchClassScores[pdx]
            #print(p, score)

            ##cv2.imshow('Color Image', colorImg)
            #cv2.imshow('Predicted Image', img)
            ##cv2.imshow('Depth Image', depthImg)
            #cv2.imshow('Depth Color Map', depthImgColorMap)

            #key = cv2.waitKey(0)
            #if key & 0xFF == 27:    break    # break with esc key.
            #elif key & 0xFF == ord('q'):    break    # break with 'q' key.
        
            #cv2.destroyAllWindows()

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
            
            #print(objPosiX, objPosiY, objPosiZ)
            #print(objOriX, objOriY, objOriZ, objOriW)
            #print(objScaleX, objScaleY, objScaleZ)
            
#-------------------------------------------------------------------------------
                    
            if objectName == 'nuts':                pass
            if objectName == 'coins':                pass
            if objectName == 'washers':                pass
            if objectName == 'gears':                pass
            if objectName == 'emptyBin':                pass
            if objectName == 'crankArmW' or objectName == 'crankArmX' or objectName == 'crankShaft':
                # Initializing the position of the arm before moving it to pickup object.
                jointStateDict = {'shoulder_pan_joint': -1.5, 'shoulder_lift_joint': 0.0, 
                                  'upperarm_roll_joint': 0.0, 'elbow_flex_joint': 0.0, 
                                  'forearm_roll_joint': 0.0, 'wrist_flex_joint': 0.0, 
                                  'wrist_roll_joint': 0.0}
                fetchRobot.moveMultipleJoints(jointStateDict, verbose=False)

                # Setting the position and orientation for the gripper.
                # The gripper is positioned such that it can hold one of the side walls 
                # of the blue bin. Hence there is some offset along the y axis.
                position = Point(objPosiX, objPosiY, objPosiZ + 0.1)
                orientation = Quaternion(objOriX, objOriY, objOriZ, objOriW)
                angles = tf.transformations.euler_from_quaternion([objOriX, objOriY, objOriZ, objOriW])
                print(objOriX, objOriY, objOriZ, objOriW)
                print(np.rad2deg([angles]))
                
                # We want to grab the bin from the top vertically. So fixing an orientation
                # rotated along the y axis and then moving the arm.
                # But sometimes if the rotation along y axis is made 90 degrees, then 
                # the arm is not able to reach the position. Hence we have made it 80 degrees.
                rotatedOrientation = tf.transformations.quaternion_from_euler(0, np.deg2rad(80), 0)
                rotatedOrientation = Quaternion(*rotatedOrientation)
                fetchRobot.gripperOpen(verbose=False)
                fetchRobot.gripperToPose(Pose(position, rotatedOrientation), verbose=False)
                
                # Now rotating only the wrist joint to be across the yaw orientation of the 
                # object, so that its easy to pick up.
                # A value of 3 to the wrist joint rotates it to pi radians.
                fetchRobot.moveSingleJoint('wrist_roll_joint', angles[2]*3/np.pi, verbose=False)
                
                # Now lower the torso_lift_joint and grip the object.
                fetchRobot.moveSingleJoint('torso_lift_joint', 0.25, verbose=False)
                fetchRobot.gripperClose(verbose=False)
                fetchRobot.moveSingleJoint('torso_lift_joint', 0.4, verbose=False)

                break

                ## Tilting the wrist more and grab the bin by the sidewall.
                ## The 0.005 is added so that the gripper does not collide with the table top.
                #position = Point(objPosiX, objPosiY + objScaleY/2, objPosiZ - objScaleZ/2 + 0.005)
                #fetchRobot.gripperToPose(Pose(position, rotatedOrientation), verbose=False)
                #fetchRobot.gripperClose(verbose=False)
                
                ## Now pick it up by increasing the position along the z axis by some amount
                ## and keeping the orientation the same.
                #position = Point(objPosiX, objPosiY + objScaleY/2, objPosiZ + objScaleZ)
                #rotatedOrientation = tf.transformations.quaternion_from_euler(0, np.deg2rad(60), 0)
                #rotatedOrientation = Quaternion(*rotatedOrientation)
                #fetchRobot.gripperToPose(Pose(position, rotatedOrientation), verbose=False)
                
                ## Putting arm to start position.
                #fetchRobot.gripperToPose(Point(0.5, -0.75, 1.1), verbose=False)
                
        # Now move the arm to the side by rotating 90 degrees along the z axis.
        jointStateDict = {'shoulder_pan_joint': -1.5, 'shoulder_lift_joint': 0.0, 
                          'elbow_flex_joint': 0.0, 'wrist_flex_joint': 0.0}
        fetchRobot.moveMultipleJoints(jointStateDict, verbose=False)
        fetchRobot.gripperOpen(verbose=False)
                        
        break
        
#-------------------------------------------------------------------------------

        if key & 0xFF == 27:
            # Go back to initial position and tuck the arm.
            cv2.destroyAllWindows()
            #fetchRobot.baseGoForward(-0.5, speed=0.1, verbose=False)
            #fetchRobot.autoTuckArm(verbose=False)
            break    # break with esc key.

        elif key & 0xFF == ord('q'):
            # Stays in the same position and quits the code.
            cv2.destroyAllWindows()
            break    # break with esc key.
        
        elif len(detectedBatchClassNames) == 0:
            print('\nNo objects detected...\n')
            break

