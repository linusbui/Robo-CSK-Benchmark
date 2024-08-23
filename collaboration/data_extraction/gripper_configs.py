from enum import Enum

# Possible gripper configurations according to:
# Z. Samadikhoshkho, K. Zareinia, and F. Janabi-Sharifi, ‘A Brief Review on Robotic Grippers Classifications’,
# in 2019 IEEE Canadian Conference of Electrical and Computer Engineering (CCECE), Edmonton, AB, Canada, May 2019.
# doi: 10.1109/CCECE.2019.8861780.
class GripperConfig(str, Enum):
    NO = "No specified Gripper",
    TWO_FINGERS = "Robot Grippers with 2 Fingers",
    THREE_FINGERS = "Robot Grippers with 3 Fingers",
    FLEX_FINGERS = "Robot Grippers with Flexible Fingers",
    MULTI_ADAPT = "Multi-Finger and Adaptive Grippers",
    BALL = "Grain-Filled Flexible Ball Grippers",
    BELLOWS = "Bellows Grippers",
    O_RING = "O-ring Grippers"
