class MealSetting:
    def __init__(self, meal: str, utensils: [str], plate: str):
        self._meal = meal
        self._utensils = utensils
        self._plate_type = plate

    def __str__(self):
        ut_str = str(self._utensils)
        if len(self._utensils) == 0:
            ut_str = 'Hands'
        return f'To eat a {self._meal} a person needs a {self._plate_type} and the following utensils: {ut_str}'

    def get_meal(self) -> str:
        return self._meal

    def get_utensils(self) -> [str]:
        return self._utensils

    def get_plate(self) -> str:
        return self._plate_type

    def add_utensil(self, utensil: str):
        self._utensils.append(utensil)

    def to_dict(self):
        return {
            'Meal': self.get_meal(),
            'Utensils': self.get_utensils(),
            'Hands?': len(self.get_utensils()) == 0,
            'Plate Type': self.get_plate()
        }
