from utils.eval_result_super import ModelEvaluationResult


class ToolSubstitutionResult(ModelEvaluationResult):
    def __init__(self, task: str, affordance: str, correct_tool: str, prediction: str, tool_choices: [str]):
        self._task = task
        self._aff = affordance
        self._corr_tool = correct_tool.lower()
        self._pred_tool = prediction.lower()
        self._choices = tool_choices

    def get_task(self) -> str:
        return self._task

    def get_affordance(self) -> str:
        return self._aff

    def get_predicted_tool(self) -> str:
        return self._pred_tool

    def get_correct_tool(self) -> str:
        return self._corr_tool

    def get_choices(self) -> [str]:
        return self._choices

    def get_pred_correctness(self) -> bool:
        return self._pred_tool == self._corr_tool

    def to_dict(self):
        return {
            'task': self.get_task(),
            'affordance': self.get_affordance(),
            'correct_tool': self.get_correct_tool(),
            'predicted_tool': self.get_predicted_tool(),
            'tool_choices': self.get_choices(),
            'correct_prediction': self.get_pred_correctness(),
        }
