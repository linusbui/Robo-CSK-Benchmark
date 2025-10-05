import argparse

from meta_reasoning.prompting import meta_reason_prompter_binary, meta_reason_prompter_multi_choice
from procedural_knowledge.prompting import cooking_procedures_prompter
from table_setting.prompting import table_setting_prompter
from tidy_up.prompting import tidy_up_prompter_open, tidy_up_prompter_multi_choice
from tool_usage.prompting import tool_use_prompter
from utils import GemmaPrompter, OpenAIPrompter, LlamaPrompter

it_selfcon = 3
it_selfref = 5
n_ex_sgicl = 8
n_ex_contr = 4

def main():
    parser = argparse.ArgumentParser(
        description='RoboCSKBench - Please choose the model to evaluate and the tasks to evaluate on')
    parser.add_argument("--models", type=str, choices=["gpt", "llama", "gemma"], nargs="+",
                        help="Choose one or more models to run")
    parser.add_argument("--tasks", type=str, choices=["ts", "mr_b", "mr_mc", "to_us", "ti_up_o", "ti_up_mc", "proc_b", "proc_mc"],
                        nargs="+", help="Choose one or more tasks to run")
    parser.add_argument('--techs' ,type=str, choices=['base', 'rar', 'meta', 'contr', 'selfcon', 'selfref', 'sgicl', 'stepback'],
                        nargs="+")
    parser.add_argument("--new_tok", type=int, default=1000, help="Maximum number of new tokens for the model output")
    parser.add_argument("--num_q", type=int, default=None, help="Number of questions to be asked in each task. Mostly for testing.")
    args = parser.parse_args()

    num_runs = args.num_q

    prompters = []
    # List of prompters with higher temperature for Self Constistency
    prompters_selfcon = [] 
    # Evaluate models:
    if "gpt" in args.models:
        if 'selfcon' in args.techs:
            gpt4 = OpenAIPrompter(0.5)
            prompters_selfcon.append(gpt4)
        else:
            gpt4 = OpenAIPrompter()
            prompters.append(gpt4)
        print(f'Evaluating on the following GPT-4o model: {gpt4.model_name}')
    if "llama" in args.models:
        if 'selfcon' in args.techs:
            llama = LlamaPrompter(args.new_tok, 0.5, True)
            prompters_selfcon.append(llama)
        else:
            llama = LlamaPrompter(args.new_tok)
            prompters.append(llama)
        print(f'Evaluating on the following Llama 3.3 model: {llama.model_name}')
    if "gemma" in args.models:
        if 'selfcon' in args.techs:
            gemma = GemmaPrompter(args.new_tok, 0.5, True)
            prompters_selfcon.append(GemmaPrompter(args.new_tok, 0.5, True))
        else:
            gemma = GemmaPrompter(args.new_tok)
            prompters.append(gemma)
        print(f'Evaluating on the following Gemma 2 model: {gemma.model_name}')

    # Decide which tasks to evaluate with which prompting techniques:
    if "ts" in args.tasks:
        if 'base' in args.techs:
            table_setting_prompter.prompt_all_models(prompters, num_runs)
            print('Finished the Table Setting task (Baseline)')
        if 'rar' in args.techs:
            table_setting_prompter.prompt_all_models_rar(prompters, num_runs)
            print('Finished the Table Setting task (Rephrase and Respond)')
        if 'meta' in args.techs:
            table_setting_prompter.prompt_all_models_meta(prompters, num_runs)
            print('Finished the Table Setting task (Metacognitive)')
        if 'selfcon' in args.techs:
            table_setting_prompter.prompt_all_models_selfcon(prompters_selfcon, num_runs, it_selfcon)
            print('Finished the Table Setting task (Self Consistency)')
        if 'selfref' in args.techs:
            table_setting_prompter.prompt_all_models_selfref(prompters, num_runs, it_selfref)
            print('Finished the Table Setting task (Self Refine)')
        if 'stepback' in args.techs:
            table_setting_prompter.prompt_all_models_stepback(prompters, num_runs)
            print('Finished the Table Setting task (Stepback)')
        if 'sgicl' in args.techs:
            table_setting_prompter.prompt_all_models_sgicl(prompters, num_runs, n_ex_sgicl)
            print('Finished the Table Setting task (SG-ICL)')
        if 'contr' in args.techs:
            table_setting_prompter.prompt_all_models_contr(prompters, num_runs, n_ex_contr, it_selfcon)
            print('Finished the Table Setting task (Contrastive CoT)')
    if "mr_mc" in args.tasks:
        if 'base' in args.techs:
            meta_reason_prompter_multi_choice.prompt_all_models(prompters, num_runs)
            print('Finished the Meta-Reasoning task (Multi Choice Questions) (Baseline)')
        if 'rar' in args.techs:
            meta_reason_prompter_multi_choice.prompt_all_models_rar(prompters, num_runs)
            print('Finished the Meta-Reasoning task (Multi Choice Questions) (Rephrase and Respond)')
        if 'meta' in args.techs:
            meta_reason_prompter_multi_choice.prompt_all_models_meta(prompters, num_runs)
            print('Finished the Meta-Reasoning task (Multi Choice Questions) (Metacognitive)')
        if 'selfcon' in args.techs:
            meta_reason_prompter_multi_choice.prompt_all_models_selfcon(prompters_selfcon, num_runs, it_selfcon)
            print('Finished the Meta-Reasoning task (Multi Choice Questions) (Self Consistency)')
        if 'selfref' in args.techs:
            meta_reason_prompter_multi_choice.prompt_all_models_selfref(prompters, num_runs, it_selfref)
            print('Finished the Meta-Reasoning task (Multi Choice Questions) (Self Refine)')
        if 'stepback' in args.techs:
            meta_reason_prompter_multi_choice.prompt_all_models_stepback(prompters, num_runs)
            print('Finished the Meta-Reasoning task (Multi Choice Questions) (Stepback)')
        if 'sgicl' in args.techs:
            meta_reason_prompter_multi_choice.prompt_all_models_sgicl(prompters, num_runs, n_ex_sgicl)
            print('Finished the Meta-Reasoning task (Multi Choice Questions) (SG-ICL)')
        if 'contr' in args.techs:
            meta_reason_prompter_multi_choice.prompt_all_models_contr(prompters, num_runs, n_ex_contr, it_selfcon)
            print('Finished the Meta-Reasoning task (Multi Choice Questions) (Contrastive CoT)')
    if "to_us" in args.tasks:
        if 'base' in args.techs:
            tool_use_prompter.prompt_all_models(prompters, num_runs)
            print('Finished the Tool Usage task (Baseline)')
        if 'rar' in args.techs:
            tool_use_prompter.prompt_all_models_rar(prompters, num_runs)
            print('Finished the Tool Usage task (Rephrase and Respond)')
        if 'meta' in args.techs:
            tool_use_prompter.prompt_all_models_meta(prompters, num_runs)
            print('Finished the Tool Usage task (Metacognitive)')
        if 'selfcon' in args.techs:
            tool_use_prompter.prompt_all_models_selfcon(prompters_selfcon, num_runs, it_selfcon)
            print('Finished the Tool Usage task (Metacognitive)')
        if 'selfref' in args.techs:
            tool_use_prompter.prompt_all_models_selfref(prompters, num_runs, it_selfref)
            print('Finished the Tool Usage task (Self Refine)')
        if 'stepback' in args.techs:
            tool_use_prompter.prompt_all_models_stepback(prompters, num_runs)
            print('Finished the Tool Usage task (Stepback)')
        if 'sgicl' in args.techs:
            tool_use_prompter.prompt_all_models_sgicl(prompters, num_runs, n_ex_sgicl)
            print('Finished the Tool Usage task (SG-ICL)')
        if 'contr' in args.techs:
            tool_use_prompter.prompt_all_models_contr(prompters, num_runs, n_ex_contr, it_selfcon)
            print('Finished the Tool Usage task (Contrastive CoT)')
    if "ti_up_mc" in args.tasks:
        if 'base' in args.techs:
            tidy_up_prompter_multi_choice.prompt_all_models(prompters, num_runs)
            print('Finished the Tidy Up task (Multi Choice Questions) (Baseline)')
        if 'rar' in args.techs:
            tidy_up_prompter_multi_choice.prompt_all_models_rar(prompters, num_runs)
            print('Finished the Tidy Up task (Multi Choice Questions) (Rephrase and Respond)')
        if 'meta' in args.techs:
            tidy_up_prompter_multi_choice.prompt_all_models_meta(prompters, num_runs)
            print('Finished the Tidy Up task (Multi Choice Questions) (Metacognitive)')
        if 'selfcon' in args.techs:
            tidy_up_prompter_multi_choice.prompt_all_models_selfcon(prompters_selfcon, num_runs, it_selfcon)
            print('Finished the Tidy Up task (Multi Choice Questions) (Self Consistency)')
        if 'selfref' in args.techs:
            tidy_up_prompter_multi_choice.prompt_all_models_selfref(prompters, num_runs, it_selfref)
            print('Finished the Tidy Up task (Multi Choice Questions) (Self Refine)')
        if 'stepback' in args.techs:
            tidy_up_prompter_multi_choice.prompt_all_models_stepback(prompters, num_runs)
            print('Finished the Tidy Up task (Multi Choice Questions) (Stepback)')
        if 'sgicl' in args.techs:
            tidy_up_prompter_multi_choice.prompt_all_models_sgicl(prompters, num_runs, n_ex_sgicl)
            print('Finished the Tidy Up task (Multi Choice Questions) (SG-ICL)')
        if 'contr' in args.techs:
            tidy_up_prompter_multi_choice.prompt_all_models_contr(prompters, num_runs, n_ex_contr, it_selfcon)
            print('Finished the Tidy Up task (Multi Choice Questions) (Contrastive CoT)')
    if "proc_mc" in args.tasks:
        if 'base' in args.techs:
            cooking_procedures_prompter.prompt_all_models_multi(prompters, num_runs)
            print("Finished the Cooking Procedures Knowledge task (Multi Choice Questions) (Baseline)")
        if 'rar' in args.techs:
            cooking_procedures_prompter.prompt_all_models_multi_rar(prompters, num_runs)
            print("Finished the Cooking Procedures Knowledge task (Multi Choice Questions) (Rephrase and Respond)")
        if 'meta' in args.techs:
            cooking_procedures_prompter.prompt_all_models_multi_meta(prompters, num_runs)
            print("Finished the Cooking Procedures Knowledge task (Multi Choice Questions) (Metacognitive)")
        if 'selfcon' in args.techs:
            cooking_procedures_prompter.prompt_all_models_multi_selfcon(prompters_selfcon, num_runs, it_selfcon)
            print("Finished the Cooking Procedures Knowledge task (Multi Choice Questions) (Self Consistency)")
        if 'selfref' in args.techs:
            cooking_procedures_prompter.prompt_all_models_multi_selfref(prompters, num_runs, it_selfref)
            print("Finished the Cooking Procedures Knowledge task (Multi Choice Questions) (Self Refine)")
        if 'stepback' in args.techs:
            cooking_procedures_prompter.prompt_all_models_multi_stepback(prompters, num_runs)
            print("Finished the Cooking Procedures Knowledge task (Multi Choice Questions) (Stepback)")
        if 'sgicl' in args.techs:
            cooking_procedures_prompter.prompt_all_models_multi_sgicl(prompters, num_runs, n_ex_sgicl)
            print("Finished the Cooking Procedures Knowledge task (Multi Choice Questions) (SG-ICL)")
        if 'contr' in args.techs:
            cooking_procedures_prompter.prompt_all_models_multi_contr(prompters, num_runs, n_ex_contr, it_selfcon)
            print("Finished the Cooking Procedures Knowledge task (Multi Choice Questions) (Contrastive CoT)")

if __name__ == "__main__":
    main()
