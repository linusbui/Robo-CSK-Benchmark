from collaboration.prompting import collab_prompter
from table_setting.prompting import table_setting_prompter
from tidy_up.prompting import tidy_up_prompter
from tool_usage.prompting import tool_use_prompter

if __name__ == "__main__":
    table_setting_prompter.prompt_all_models()
    collab_prompter.prompt_all_models()
    tool_use_prompter.execute_prompting()
    tidy_up_prompter.prompt_all_models()
