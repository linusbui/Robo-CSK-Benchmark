from utensils_plates import Utensil, Plate


class MealSetting:
    def __init__(self, meal: str, utensils: [Utensil], plate: Plate):
        self._meal = meal
        self._utensils = utensils
        self._check_for_hands()
        self._plate_type = plate

    def __str__(self):
        return f'To eat a {self._meal} a person needs a {self._plate_type} and the following utensils: {self._get_utensils_as_str()}'

    def _check_for_hands(self):
        if len(self._utensils) == 0:
            self._utensils.append(Utensil.HAND)

    def get_meal(self) -> str:
        return self._meal

    def get_utensils(self) -> [str]:
        return self._utensils

    def get_plate(self) -> str:
        return self._plate_type

    def add_utensil(self, utensil: str):
        self._utensils.append(utensil)
        self._check_for_hands()

    def _get_utensils_as_str(self) -> [str]:
        uts = []
        for u in self._utensils:
            uts.append(str(u))
        return uts

    def to_dict(self):
        return {
            'Meal': self.get_meal(),
            'Utensils': self._get_utensils_as_str(),
            'Plate Type': str(self.get_plate())
        }
