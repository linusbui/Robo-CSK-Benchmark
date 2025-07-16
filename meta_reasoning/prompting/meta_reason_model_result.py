from utils.eval_result_super import ModelEvaluationResult


class MetaReasoningBinaryResult(ModelEvaluationResult):
    def __init__(self, task: str, hardware: str, pred_answer: bool, corr_answer: bool):
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
            'task': self.get_task(),
            'hardware_config': self.get_hardware(),
            'model_answer': self.get_predicted_answer(),
            'correct_answer': self.get_correct_answer()
        }


class MetaReasoningMultiChoiceResult(ModelEvaluationResult):
    def __init__(self, tsk: str, corr_conf: str, prediction: str, conf_choices: [str]):
        self._task = tsk
        self._corr_conf = corr_conf.lower()
        self._pred_conf = prediction.lower()
        self._choices = conf_choices

    def get_task(self) -> str:
        return self._task

    def get_correct_configuration(self) -> str:
        return self._corr_conf

    def get_predicted_configuration(self) -> str:
        return self._pred_conf

    def get_choices(self) -> [str]:
        return self._choices

    def get_pred_correctness(self) -> bool:
        return self._corr_conf == self._pred_conf

    def to_dict(self):
        return {
            'task': self.get_task(),
            'correct_configuration': self.get_correct_configuration(),
            'predicted_configuration': self.get_predicted_configuration(),
            'correct_prediction': self.get_pred_correctness(),
            'configuration_choices': self.get_choices()
        }
