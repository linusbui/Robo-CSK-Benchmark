class Prompter:
    def __init__(self, model: str, temp=0.0):
        self.model_name = model
        self.temperature = temp

    def prompt_model(self, system_msg: str, user_msg: str, question: str) -> str:
        pass
