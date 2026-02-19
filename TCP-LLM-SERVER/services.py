
import rospy
from geometry_msgs.msg import Point

def generateTrajectory(x,y) -> np.ndarray:
    genTraj = rospy.ServiceProxy('add_two_ints', generateTraj)
    
    req = Goal()
    
    req.x = x
    req.y = y
    req.z = 0

    return