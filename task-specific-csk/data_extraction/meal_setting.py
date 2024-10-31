class MealSetting:
    def __init__(self, meal: str, fork: bool, spoon: bool, knife: bool, plate: str):
        self._meal = meal
        self._needs_fork = fork
        self._needs_spoon = spoon
        self._needs_knife = knife
        self._plate_type = plate

    def __str__(self):
        utensils = []
        if self._needs_fork:
            utensils.append('Fork')
        if self._needs_spoon:
            utensils.append('Spoon')
        if self._needs_knife:
            utensils.append('Knife')
        if len(utensils) == 0:
            utensils.append('Hands')
        return f'To eat a {self._meal} a person needs a {self._plate_type} and the following utensils: {utensils}'

    def get_meal(self) -> str:
        return self._meal

    def needs_fork(self) -> bool:
        return self._needs_fork

    def needs_spoon(self) -> int:
        return self._needs_spoon

    def needs_knife(self) -> int:
        return self._needs_knife

    def get_plate(self) -> str:
        return self._plate_type

    def to_dict(self):
        return {
            'Meal': self.get_meal(),
            'Fork?': self.needs_fork(),
            'Spoon?': self.needs_spoon(),
            'Knife?': self.needs_knife(),
            'Hands?': not self.needs_fork() and not self.needs_spoon() and not self._needs_knife,
            'Plate Type': self.get_plate()
        }
