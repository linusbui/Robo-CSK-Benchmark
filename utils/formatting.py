import re

# Primary method of extracting LLM response from answer
# Assumptions:
# Exact answer is contained in the last 5 lines of the response
# First occurence of possible answer is the intended answer
def transform_prediction(pred: str, choices: [str]) -> str:
    # LLM response line by line
    split = pred.splitlines()
    split.reverse()

    # For replacement of non-alphabetical chars
    regex = re.compile('[^a-zA-Z]')

    # Scan for possible answer
    for i in range(min(len(split), 5)):
        for choice in choices:
            # for single word answers
            if not ' ' in choice:
                line = regex.sub(' ', split[i])
                for word in line.split():
                    if word.lower() == choice.lower():
                        return choice
            # for multiple word answers
            elif choice.lower() in split[i].lower():
                return choice
    print('\nNo viable answer found!')
    return 'None'


# Majority vote over LLM answers
def majority_vote(answ: [object]) -> object:
    counts = []
    for a in answ:
        counts.append(answ.count(a))
    ind = counts.index(max(counts))
    return answ[ind]