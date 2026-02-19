
import enum
from prompts import *
from enums import *
MODIFYTRAJ = "MODIFY TRAJECTORY"
GENTARGETPOINT = "GENERATE TARGET POINT"

class Intention(enum.Enum):
    MODIFYTRAJ = MODIFYTRAJ
    GENTARGETPOINT = GENTARGETPOINT