def calculate_average_precision_at_k(k: int, retrieved: [str], gold_standard: [str]) -> float:
    assert k >= 1
    r = 0
    for i in retrieved:
        if i in gold_standard:
            r += 1

    if r == 0:
        return 0.0

    ap_k = 0
    for i in range(1, k+1):
        if retrieved[i-1] in gold_standard:
            ap_k += calculate_precision_at_k(i, retrieved, gold_standard)
    return ap_k / r


def calculate_precision_at_k(k: int, retrieved: [str], gold_standard: [str]) -> float:
    assert k >= 1
    assert k <= len(gold_standard)
    c = 0
    for r in retrieved[0:k]:
        if r in gold_standard:
            c += 1
    return c / k
