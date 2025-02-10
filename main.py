from meta_reasoning.prompting import meta_reason_prompter
from table_setting.prompting import table_setting_prompter
from tidy_up.prompting import tidy_up_prompter
from tool_usage.prompting import tool_use_prompter
from cooking_procedures.prompting import cooking_procedures_prompter
from utils import GemmaPrompter, OpenAIPrompter, LlamaPrompter

# If the default max_new_token value is used for Gemma, the extraction from the prompt fails for the Tidy Up task
prompters = [OpenAIPrompter(), LlamaPrompter(), GemmaPrompter(max_new_tokens=50)]

if __name__ == "__main__":
    table_setting_prompter.prompt_all_models(prompters)
    print('Finished the Table Setting task')
    meta_reason_prompter.prompt_all_models(prompters)
    print('Finished the Meta-Reasoning task')
    tool_use_prompter.execute_prompting(prompters)
    print('Finished the Tool Usage task')
    tidy_up_prompter.prompt_all_models(prompters)
    print('Finished the Tidy Up task')
    cooking_procedures_prompter.prompt_all_models(prompters)
    print("Finished the Cooking Procedures Knowledge task")

