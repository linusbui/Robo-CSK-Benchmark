import re

from utensils_plates import Utensil, Plate


def _get_plate_from_enum(plate: str) -> Plate:
    plate_wo_brackets = re.sub(r'\s*\(.*?\)', '', plate)
    return Plate(plate_wo_brackets.strip())


def _get_utensil_mapping(utensils: str) -> dict:
    mapping = {}
    for utensil in Utensil:
        if utensil.lower() in utensils.lower():
            mapping[utensil] = True
        else:
            mapping[utensil] = False
    return mapping


class PreferredMealSetting:
    def __init__(self,  user_id: str, meal: str, r_id: str):
        self._user_id = user_id
        self._meal = meal
        self._recipe1m_id = r_id
        self._plate = ""
        self._utensils = _get_utensil_mapping("")

    def __str__(self):
        return f'To eat a {self._meal}, person {self._user_id} needs a {self._plate} and the following utensils: {self._get_utensils_as_str_arr()}'

    def add_plate(self, plate: str):
        self._plate = _get_plate_from_enum(plate)

    def add_utensils(self, utensils: str):
        self._utensils = _get_utensil_mapping(utensils)

    def get_meal(self) -> str:
        return self._meal

    def get_recipe_id(self) -> str:
        return self._recipe1m_id

    def get_user_id(self) -> str:
        return self._user_id

    def needs_specific_utensil(self, utl: Utensil) -> bool:
        return self._utensils.get(Utensil(utl))

    def get_plate(self) -> str:
        return self._plate

    def _get_utensils_as_str_arr(self) -> [str]:
        uts = []
        for u in Utensil:
            if self.needs_specific_utensil(u):
                uts.append(u)
        return uts

    def to_dict(self):
        return {
            'Recipe1M+ ID': self.get_recipe_id(),
            'Meal': self.get_meal(),
            'Plate': self.get_plate(),
            'Hands?': self.needs_specific_utensil(Utensil.HANDS),
            'Tongs?': self.needs_specific_utensil(Utensil.TONGS),
            'Knife?': self.needs_specific_utensil(Utensil.KNIFE),
            'Fork?': self.needs_specific_utensil(Utensil.FORK),
            'Skewer?': self.needs_specific_utensil(Utensil.SKEWER),
            'Chopsticks?': self.needs_specific_utensil(Utensil.CHOPS),
            'Spoon?': self.needs_specific_utensil(Utensil.SPOON),
            'Utensils': self._get_utensils_as_str_arr(),
            'User ID': self.get_user_id()
        }
