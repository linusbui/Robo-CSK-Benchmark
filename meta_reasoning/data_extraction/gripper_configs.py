from enum import Enum

# Possible gripper configurations according to:
# Z. Samadikhoshkho, K. Zareinia, and F. Janabi-Sharifi, ‘A Brief Review on Robotic Grippers Classifications’,
# in 2019 IEEE Canadian Conference of Electrical and Computer Engineering (CCECE), Edmonton, AB, Canada, May 2019.
# doi: 10.1109/CCECE.2019.8861780.
class GripperConfig(str, Enum):
    NO = ("No specified Gripper", 0)
    TWO_FINGERS = ("Robot Grippers with 2 Fingers", 1)
    THREE_FINGERS = ("Robot Grippers with 3 Fingers", 2)
    FLEX_FINGERS = ("Robot Grippers with Flexible Fingers", 3)
    MULTI_ADAPT = ("Multi-Finger and Adaptive Grippers", 4)
    BALL = ("Grain-Filled Flexible Ball Grippers", 5)
    BELLOWS = ("Bellows Grippers", 6)
    O_RING = ("O-ring Grippers", 7)

    def __new__(cls, value, rank):
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.rank = rank
        return obj

    # Comparison based on rank
    def __lt__(self, other):
        if isinstance(other, GripperConfig):
            return self.rank < other.rank
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, GripperConfig):
            return self.rank <= other.rank
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, GripperConfig):
            return self.rank > other.rank
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, GripperConfig):
            return self.rank >= other.rank
        return NotImplemented

    # "Arithmetics" to jump between grippers on different ranks
    @classmethod
    def from_rank(cls, rank: int):
        for member in cls:
            if member.rank == rank:
                return member
        raise ValueError(f"No gripper with rank {rank}")

    def add(self, n: int):
        new_rank = self.rank + n
        if not any(member.rank == new_rank for member in GripperConfig):
            raise IndexError(f"Calculated rank {new_rank} is out of range [0, 7]")
        return GripperConfig.from_rank(new_rank)
