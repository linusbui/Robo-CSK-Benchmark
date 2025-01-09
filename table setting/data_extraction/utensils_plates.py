from enum import Enum


class Utensil(str, Enum):
    HANDS = "Hands"
    TONGS = "Tongs"
    KNIFE = "Knife"
    FORK = "Fork"
    SKEWER = "Skewer"
    CHOPS = "Chopsticks"
    SPOON = "Spoon"

    def __new__(cls, value):
        obj = str.__new__(cls, value)
        obj._value_ = value
        return obj

    def __str__(self):
        return self.value


class Plate(str, Enum):
    DINNER = "Dinner Plate"
    DESSERT = "Dessert Plate"
    BOWL = "Bowl"
    COUPE = "Coupe Plate"

    def __new__(cls, value):
        obj = str.__new__(cls, value)
        obj._value_ = value
        return obj

    def __str__(self):
        return self.value
