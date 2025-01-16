from table_setting.data_extraction.utensils_plates import Plate, Utensil


class TableSettingModelResult:
    def __init__(self, meal: str, correct_plate: Plate, correct_utensils: [Utensil]):
        self._meal = meal
        self._pred_plate = Plate.NONE
        self._pred_utensils = []
        self._corr_plate = correct_plate
        self._corr_utensils = correct_utensils

    def add_predicted_plate(self, predicted_plate: Plate):
        self._pred_plate = predicted_plate

    def add_predicted_utensils(self, predicted_utensils: Utensil):
        self._pred_utensils = predicted_utensils

    def get_meal(self) -> str:
        return self._meal

    def get_predicted_plate(self) -> Plate:
        return self._pred_plate

    def get_predicted_utensils(self) -> [Utensil]:
        return self._pred_utensils

    def get_correct_plate(self) -> Plate:
        return self._corr_plate

    def get_correct_utensils(self) -> [Utensil]:
        return self._corr_utensils

    def get_plate_pred_correctness(self) -> bool:
        return self._pred_plate == self._corr_plate

    def get_jaccard_for_utensils(self) -> float:
        intersect = set(self._pred_utensils) & set(self._corr_utensils)
        union = set(self._pred_utensils + self._corr_utensils)
        return len(intersect) / len(union)

    def to_dict(self):
        return {
            'Meal': self.get_meal(),
            'Predicted Plate': str(self.get_predicted_plate()),
            'Correct Plate': str(self.get_correct_plate()),
            'Predicted Utensils': str(self.get_predicted_utensils()),
            'Correct Utensils': str(self.get_correct_utensils()),
            'Correct Plate?': self.get_plate_pred_correctness(),
            'Jaccard Utensils': self.get_jaccard_for_utensils()
        }
