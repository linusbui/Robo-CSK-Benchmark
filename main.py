from collaboration.prompting import collab_prompter
from table_setting.prompting import table_setting_prompter
from tidy_up.prompting import tidy_up_prompter
from tool_usage.prompting import tool_use_prompter
from utils import GemmaPrompter

#prompters = [OpenAIPrompter(), LlamaPrompter(), GemmaPrompter()]
prompters = [GemmaPrompter()]

if __name__ == "__main__":
    #table_setting_prompter.prompt_all_models(prompters)
    #print('Finished the Table Setting task')
    collab_prompter.prompt_all_models(prompters)
    print('Finished the Meta-Reasoning task')
    #tool_use_prompter.execute_prompting(prompters)
    #print('Finished the Tool Usage task')
    #tidy_up_prompter.prompt_all_models(prompters)
    #print('Finished the Tidy Up task')
