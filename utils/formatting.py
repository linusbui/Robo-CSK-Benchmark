# Takes LLM response from Meta Reasoning prompting technique and list of possible answers
# Returns first match of possible answers in the LLM response
# Also works for rephrase and respond
def transform_prediction_meta_single(pred: str, choices: [str]) -> str:
    # Extract last sentence from LLM response
    answ = pred.splitlines()[-1]

    # Check if last sentence contains viable answer
    for choice in choices:
        if choice.lower() in answ.lower():
            return choice
    print(f'{answ} does not contain viable answer')
    return pred


# Takes LLM response from Meta Reasoning prompting technique and list of possible answers
# Returns all matches of possible answers in the LLM response
# Also works for rephrase and respond
def transform_prediction_meta_mult(pred: str, choices: [str]) -> str:
    # Extract last sentence from LLM response
    answ = pred.splitlines()[-1]

    # Check if last sentecne contains viable answer
    res = []
    for choice in choices:
        if choice.lower() in answ.lower():
            res.append(choice)
    if len(res) == 0:
        print(f'{answ} does not contain viable answer')
    return res


def transform_prediction_selfcon_single(pred: str, choices: [str]) -> str:
    # Extract last three sentence from LLM response
    split = pred.splitlines()
    split.reverse()

    # Scan back to front for viable answer, abort after three lines
    l = 0
    for line in split:
        if l > 2: break
        for choice in choices:
            if choice.lower() in line.lower():
                return choice
        l+=1
    print(f'{pred} does not contain viable answer')
    return pred


# Majority vote over LLM answers
def majority_vote(answ: [object]) -> object:
    counts = []
    for a in answ:
        counts.append(answ.count(a))
    ind = counts.index(max(counts))
    return answ[ind]