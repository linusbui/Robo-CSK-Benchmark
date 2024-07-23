import pandas as pd
from metric_calculator import calculate_reciprocal_rank, calculate_average_precision_at_k, calculate_recall_at_k
import ast


class ModelResultTuple:
    def __init__(self, obj: str, locations: [str]):
        self._object = obj
        self._locations = []
        self._locations.extend(locations)
        self._rr = 0.0
        self._ap5 = 0.0
        self._ap3 = 0.0
        self._ap1 = 0.0
        self._rec5 = 0.0
        self._rec3 = 0.0
        self._rec1 = 0.0
        self.calculate_metrics()

    def __str__(self):
        return (f'{self._object} located at/in {self._locations} with RR: {self._rr:.2f}, AP@1: {self._ap1:.2f}, '
                f'AP@3: {self._ap3:.2f}, AP@5: {self._ap5:.2f}, Rec@1: {self._rec1:.2f}, Rec@3: {self._rec3:.2f}, '
                f'Rec@5: {self._rec5:.2f}')

    def get_object(self) -> str:
        return self._object

    def get_locations(self) -> [str]:
        return self._locations

    def calculate_metrics(self):
        gold_standard = self.get_gold_standard_locations()
        if len(gold_standard) == 0:
            return
        self._rr = calculate_reciprocal_rank(self._locations, gold_standard)
        self._ap1 = calculate_average_precision_at_k(1, self._locations, gold_standard)
        self._ap3 = calculate_average_precision_at_k(3, self._locations, gold_standard)
        self._ap5 = calculate_average_precision_at_k(5, self._locations, gold_standard)
        self._rec1 = calculate_recall_at_k(1, self._locations, gold_standard)
        self._rec3 = calculate_recall_at_k(3, self._locations, gold_standard)
        self._rec5 = calculate_recall_at_k(5, self._locations, gold_standard)

    def get_gold_standard_locations(self) -> [str]:
        data = pd.read_csv('../tidy_up_data.csv', delimiter=',', on_bad_lines='skip')
        object_data = data[data['Object'] == self._object]
        gold_standard = []
        if object_data is None:
            return gold_standard

        object_data['Locations'] = object_data['Locations'].apply(ast.literal_eval)
        for location_dict in object_data['Locations']:
            for key, value in location_dict.items():
                gold_standard.append(value[0])
        return gold_standard
