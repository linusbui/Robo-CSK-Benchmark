from enum import Enum

class Utensil(str, Enum):
    KNIFE = "Knife"
    FORK = "Fork"
    SPOON = "Spoon"
    CHOPS = "Chopsticks"
    HAND = "Hand"

    def __new__(cls, value):
        obj = str.__new__(cls, value)
        obj._value_ = value
        return obj

    def __str__(self):
        return self.value

class Plate(str, Enum):
    BOWL = "Bowl"
    SMALL = "Small Plate"
    NORMAL = "Normal Plate"
    SMALL_OR_BOWL = "Small Plate or Bowl"

    def __new__(cls, value):
        obj = str.__new__(cls, value)
        obj._value_ = value
        return obj

    def __str__(self):
        return self.value