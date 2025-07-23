import argparse

from meta_reasoning.prompting import meta_reason_prompter_binary, meta_reason_prompter_multi_choice
from procedural_knowledge.prompting import cooking_procedures_prompter
from table_setting.prompting import table_setting_prompter
from tidy_up.prompting import tidy_up_prompter_open, tidy_up_prompter_multi_choice
from tool_usage.prompting import tool_use_prompter
from utils import GemmaPrompter, OpenAIPrompter, LlamaPrompter


def main():
    parser = argparse.ArgumentParser(
        description='RoboCSKBench - Please choose the model to evaluate and the tasks to evaluate on')
    parser.add_argument("--models", type=str, choices=["gpt", "llama", "gemma"], nargs="+",
                        help="Choose one or more models to run")
    parser.add_argument("--tasks", type=str, choices=["ts", "mr_b", "mr_mc", "to_us", "ti_up_o", "ti_up_mc", "proc"],
                        nargs="+", help="Choose one or more tasks to run")
    parser.add_argument("--new_tok", type=int, default=1000, help="Maximum number of new tokens for the model output")
    args = parser.parse_args()

    prompters = []
    # Evaluate models:
    if "gpt" in args.models:
        gpt4 = OpenAIPrompter()
        prompters.append(gpt4)
        print(f'Evaluating on the following GPT-4o model: {gpt4.model_name}')
    if "llama" in args.models:
        llama = LlamaPrompter(args.new_tok)
        prompters.append(llama)
        print(f'Evaluating on the following Llama 3.3 model: {llama.model_name}')
    if "gemma" in args.models:
        gemma = GemmaPrompter(args.new_tok)
        prompters.append(gemma)
        print(f'Evaluating on the following Gemma 2 model: {gemma.model_name}')

    # Decide which tasks to evaluate:
    if "ts" in args.tasks:
        table_setting_prompter.prompt_all_models(prompters)
        print('Finished the Table Setting task')
    if "mr_b" in args.tasks:
        meta_reason_prompter_binary.prompt_all_models(prompters)
        print('Finished the Meta-Reasoning task (Binary Questions)')
    if "mr_mc" in args.tasks:
        meta_reason_prompter_multi_choice.prompt_all_models(prompters)
        print('Finished the Meta-Reasoning task (Multi Choice Questions)')
    if "to_us" in args.tasks:
        tool_use_prompter.prompt_all_models(prompters)
        print('Finished the Tool Usage task')
    if "ti_up_o" in args.tasks:
        tidy_up_prompter_open.prompt_all_models(prompters)
        print('Finished the Tidy Up task (Open Questions)')
    if "ti_up_mc" in args.tasks:
        tidy_up_prompter_multi_choice.prompt_all_models(prompters)
        print('Finished the Tidy Up task (Multi Choice Questions)')
    if "proc" in args.tasks:
        cooking_procedures_prompter.prompt_all_models(prompters)
        print("Finished the Cooking Procedures Knowledge task")

if __name__ == "__main__":
    main()
