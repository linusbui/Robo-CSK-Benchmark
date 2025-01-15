class CollaborationModelResult:
    def __init__(self, task: str, hardware: str, pred_answer: bool, corr_answer=True):
        self._task = task
        self._hardware = hardware
        self._pred_answer = pred_answer
        self._corr_answer = corr_answer

    def get_task(self) -> str:
        return self._task

    def get_hardware(self) -> str:
        return self._hardware

    def get_predicted_answer(self) -> bool:
        return self._pred_answer

    def get_correct_answer(self) -> bool:
        return self._corr_answer

    def get_classification_type(self) -> str:
        if self._corr_answer and self._pred_answer:
            return 'tp'
        if self._corr_answer and not self._pred_answer:
            return 'fn'
        if not self._corr_answer and self._pred_answer:
            return 'fp'
        if not self._corr_answer and not self._pred_answer:
            return 'tn'

    def to_dict(self):
        return {
            'Task': self.get_task(),
            'Hardware Config': self.get_hardware(),
            'Model Answer': self.get_predicted_answer(),
            'Correct Answer': self.get_correct_answer()
        }
