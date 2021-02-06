#!/usr/bin/env python

# A very basic robot script that moves robot forward.

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np
import tf as tfn    # This is the package called tf (transformation), not tensorflow.

# Namespaces has to be put in the minimal.launch file in this location: /opt/ros/kinetic/share/turtlebot_bringup/launch

class Movement(object):
    '''
    This class defines all the basic movements that the robot needs to do.
    '''
    def __init__(self):
        # Defining the node.
        rospy.init_node('Movement', anonymous=False)

        # This robotName helps to communicate with the correct robot.
        # The robotName is basically the namespace that is there at the beginning
        # of each topic belonging to that robot.
        self.robotName1 = 'fetch'
        self.robotName2 = 'freight'
        
        # Create a publisher which can "talk" to robot and tell it to move
        # Tip: You may need to change cmd_vel_mux/input/navi to /cmd_vel 
        # if you're not using robot2.
        self.pub1 = rospy.Publisher( self.robotName1 + '/base_controller/command', Twist, queue_size=10 )
        self.pub2 = rospy.Publisher( self.robotName2 + '/base_controller/command', Twist, queue_size=10 )

        # Subscribing to the odom topic to see how far the robot has moved.
        rospy.Subscriber( self.robotName1 + '/odom', Odometry, self.callback1 )
        rospy.Subscriber( self.robotName2 + '/odom', Odometry, self.callback2 )

        # Tell user how to stop robot
        rospy.loginfo("To stop robot CTRL + C")

        # robot will stop if we don't keep telling it to move.  
        # How often should we tell it to move? 10 hz.
        self.r = rospy.Rate(10);
        
        # Variables to hold the x, y position and orientation about z of the robot.
        self.posX, self.posY, self.posZ = 0.0, 0.0, 0.0
        self.oriX, self.oriY, self.oriZ, self.oriW = 0.0, 0.0, 0.0, 0.0
        self.roll, self.pitch, self.yaw = 0.0, 0.0, 0.0     # Angles in radians.
        self.rollD, self.pitchD, self.yawD = 0.0, 0.0, 0.0     # Angles in degrees.
        
        # By just calling this wait_for_message function, the Subscriber is 
        # called once and hence the callback function is run once, which updates
        # the self.posX and self.posY variables, otherwise they will always start of
        # with value of 0.0s.
        print('here1')
        msg1 = rospy.wait_for_message( self.robotName1 + '/odom', Odometry )
        print('here2')
        msg2 = rospy.wait_for_message( self.robotName2 + '/odom', Odometry )
        print('here3')

################################################################################

    def shutdown(self):
        # Stop turtlebot.
        rospy.loginfo("Stop robot")
        
        # A default Twist has linear.x of 0 and angular.z of 0. So it'll stop robot.
        self.pub1.publish(Twist())
        self.pub2.publish(Twist())
        
        # Sleep just makes sure robot receives the stop command prior to 
        # shutting down the script.
        rospy.sleep(1)

################################################################################

    def callback1(self, msg1):
        # Callback function of the Subscriber.
        # This updates the x, y coordinates of the robot.
        self.posX1 = msg1.pose.pose.position.x
        self.posY1 = msg1.pose.pose.position.y
        self.posZ1 = msg1.pose.pose.position.z
        self.oriX1 = msg1.pose.pose.orientation.x
        self.oriY1 = msg1.pose.pose.orientation.y
        self.oriZ1 = msg1.pose.pose.orientation.z
        self.oriW1 = msg1.pose.pose.orientation.w
        
        # Calculating the roll, pitch and yaw from the quaternion.
        (self.roll1, self.pitch1, self.yaw1) = \
            tfn.transformations.euler_from_quaternion([self.oriX1, self.oriY1, \
                                                       self.oriZ1, self.oriW1])
        self.rollD1 = np.rad2deg(self.roll1)
        self.pitchD1 = np.rad2deg(self.pitch1)
        self.yawD1 = np.rad2deg(self.yaw1)

################################################################################

    def callback2(self, msg2):
        # Callback function of the Subscriber.
        # This updates the x, y coordinates of the robot.
        self.posX2 = msg2.pose.pose.position.x
        self.posY2 = msg2.pose.pose.position.y
        self.posZ2 = msg2.pose.pose.position.z
        self.oriX2 = msg2.pose.pose.orientation.x
        self.oriY2 = msg2.pose.pose.orientation.y
        self.oriZ2 = msg2.pose.pose.orientation.z
        self.oriW2 = msg2.pose.pose.orientation.w
        
        # Calculating the roll, pitch and yaw from the quaternion.
        (self.roll2, self.pitch2, self.yaw2) = \
            tfn.transformations.euler_from_quaternion([self.oriX2, self.oriY2, \
                                                       self.oriZ2, self.oriW2])
        self.rollD2 = np.rad2deg(self.roll2)
        self.pitchD2 = np.rad2deg(self.pitch2)
        self.yawD2 = np.rad2deg(self.yaw2)

################################################################################

    def forward(self, xDist, xSpeed1=0.1, xSpeed2=0.1):
        # Moves the robot by a distance of xDist from its current position. 
        # Default speed is 0.05 m/s. Twist is a datatype for velocity.
        cmd1 = Twist()
        cmd2 = Twist()
        
        # The targetX is calculated from the current x coordinate.
        targetX1 = self.posX1 + xDist
        targetX2 = self.posX2 + xDist
        
        # Speed is +ve for +ve distance movement, else the speed is made -ve.
        xSpeed1 = xSpeed1 if targetX1 > self.posX1 else -1.0 * xSpeed1
        xSpeed2 = xSpeed2 if targetX2 > self.posX2 else -1.0 * xSpeed2
        
        # Assign the speed to the x command.
        cmd1.linear.x = xSpeed1
        cmd2.linear.x = xSpeed2
        # Let's turn at 0 radians/s.
        cmd1.angular.z = 0
        cmd2.angular.z = 0

        # As long as you haven't ctrl + c keeping doing...
        while not rospy.is_shutdown():
            self.pub1.publish( cmd1 )   # publish the velocity
            self.pub2.publish( cmd2 )   # publish the velocity

            print('\rposX1: {} ; posY1: {} ; oriZ1: {}'.format(self.posX1, self.posY1, self.oriZ1))
            print('\rposX2: {} ; posY2: {} ; oriZ2: {}'.format(self.posX2, self.posY2, self.oriZ2))

            # Stopping condition while moving forward (forward motion indicated 
            # by +ve xSpeed).
            if xSpeed1 > 0 and self.posX1 > targetX1:    
                print( 'here1', targetX1, xSpeed1 )
                break
            # Stopping condition while moving backward (backward motion indicated 
            # by -ve xSpeed).
            elif xSpeed1 < 0 and self.posX1 < targetX1:    
                print( 'here2' )
                break

            # Stopping condition while moving forward (forward motion indicated 
            # by +ve xSpeed).
            if xSpeed2 > 0 and self.posX2 > targetX2:    
                print( 'here3' )
                break
            # Stopping condition while moving backward (backward motion indicated 
            # by -ve xSpeed).
            elif xSpeed2 < 0 and self.posX2 < targetX2:    
                print( 'here4' )
                break

            # Wait for 0.1 seconds (10 HZ) and publish again.
            self.r.sleep()
            #rospy.rostime.wallsleep(0.1)

        # What function to call when you ctrl + c    
        rospy.on_shutdown(self.shutdown)

################################################################################

    def rotate(self, yawAnglD, zSpeed=0.2):
        # Rotates the robot along the z axis by an angle from its current orientation. 
        # Default speed is 0.05 m/s. Twist is a datatype for velocity.
        # The input angle is in degrees.
        
        yawAngl = yawAnglD * np.pi / 180.0
        
        cmd = Twist()
        
        # The targetX is calculated from the current x coordinate.
        # The yawD becomes -ve beyond 180 degrees. So 181 degree is actually 
        # calculated as -179 degrees. So have to deduct 360 from the angle if 
        # it goes beyond 180 degrees.
        targetYawD1 = self.yawD1 + yawAnglD
        if targetYawD1 > 180:    targetYawD1 -= 360
        targetYawD2 = self.yawD2 + yawAnglD
        if targetYawD2 > 180:    targetYawD2 -= 360
        
        # Speed is +ve for anti-clockwise movement, else the speed is made -ve.
        zSpeed = zSpeed if targetYawD1 > self.yawD1 else -1.0 * zSpeed
        
        # Assign the speed to 0.
        cmd.linear.x = 0
        # Let's turn at 0 radians/s.
        cmd.angular.z = zSpeed

        # As long as you haven't ctrl + c keeping doing...
        while not rospy.is_shutdown():
            # publish the velocity
            self.pub1.publish(cmd)
            self.pub2.publish(cmd)

            print('\rposX1: {} ; posY1: {} ; oriZ1: {}'.format(self.posX1, self.posY1, self.yawD1))
            print('\rposX2: {} ; posY2: {} ; oriZ2: {}'.format(self.posX2, self.posY2, self.yawD2))

            # Stopping condition while rotating clockwise (forward motion indicated 
            # by +ve xSpeed).
            if zSpeed > 0 and self.yawD1 > targetYawD1:    break
            # Stopping condition while moving backward (backward motion indicated 
            # by -ve xSpeed).
            elif zSpeed < 0 and self.yawD1 < targetYawD1:    break

            # Stopping condition while rotating clockwise (forward motion indicated 
            # by +ve xSpeed).
            if zSpeed > 0 and self.yawD2 > targetYawD2:    break
            # Stopping condition while moving backward (backward motion indicated 
            # by -ve xSpeed).
            elif zSpeed < 0 and self.yawD2 < targetYawD2:    break

            # Wait for 0.1 seconds (10 HZ) and publish again.
            self.r.sleep()
            #rospy.rostime.wallsleep(0.1)

        # What function to call when you ctrl + c    
        rospy.on_shutdown(self.shutdown)

################################################################################

if __name__ == '__main__':
    
    m = Movement()
    #m.forward(1.2)
    m.rotate(-90)

    











