from metric_calculator import calculate_reciprocal_rank, calculate_average_precision_at_k, calculate_recall_at_k


class TidyUpResult:
    def __init__(self, obj: str, pred_locations: [str], corr_locations: [str]):
        self._object = obj
        self._pred_locations = []
        self._pred_locations.extend(pred_locations)
        self._corr_locations = []
        self._corr_locations.extend(corr_locations)
        self._rr = 0.0
        self._ap5 = 0.0
        self._ap3 = 0.0
        self._ap1 = 0.0
        self._rec5 = 0.0
        self._rec3 = 0.0
        self._rec1 = 0.0
        self.calculate_metrics()

    def __str__(self):
        return (f'{self._object} located at/in {self._pred_locations} with RR: {self._rr:.2f}, AP@1: {self._ap1:.2f}, '
                f'AP@3: {self._ap3:.2f}, AP@5: {self._ap5:.2f}, Rec@1: {self._rec1:.2f}, Rec@3: {self._rec3:.2f}, '
                f'Rec@5: {self._rec5:.2f}')

    def get_object(self) -> str:
        return self._object

    def get_predicted_locations(self) -> [str]:
        return self._pred_locations

    def get_correct_locations(self) -> [str]:
        return self._corr_locations

    def get_reciprocal_rank(self) -> float:
        return self._rr

    def get_average_precision_at1(self) -> float:
        return self._ap1

    def get_average_precision_at3(self) -> float:
        return self._ap3

    def get_average_precision_at5(self) -> float:
        return self._ap5

    def get_recall_at1(self) -> float:
        return self._rec1

    def get_recall_at3(self) -> float:
        return self._rec3

    def get_recall_at5(self) -> float:
        return self._rec5

    def calculate_metrics(self):
        if len(self._corr_locations) == 0:
            return
        self._rr = calculate_reciprocal_rank(self._pred_locations, self._corr_locations)
        self._ap1 = calculate_average_precision_at_k(1, self._pred_locations, self._corr_locations)
        self._ap3 = calculate_average_precision_at_k(3, self._pred_locations, self._corr_locations)
        self._ap5 = calculate_average_precision_at_k(5, self._pred_locations, self._corr_locations)
        self._rec1 = calculate_recall_at_k(1, self._pred_locations, self._corr_locations)
        self._rec3 = calculate_recall_at_k(3, self._pred_locations, self._corr_locations)
        self._rec5 = calculate_recall_at_k(5, self._pred_locations, self._corr_locations)

    def to_dict(self):
        return {
            'object': self.get_object(),
            'pred_locations': self.get_predicted_locations(),
            'rr': self.get_reciprocal_rank(),
            'ap@1': self.get_average_precision_at1(),
            'ap@3': self.get_average_precision_at3(),
            'ap@5': self.get_average_precision_at5(),
            'rec@1': self.get_recall_at1(),
            'rec@3': self.get_recall_at3(),
            'rec@5': self.get_recall_at5()
        }
