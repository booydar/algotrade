import numpy as np, random

MIN = "[MIN"
MAX = "[MAX"
MED = "[MED"
SUM_MOD = "[SM"
END = "]"

OPERATORS = [MIN, MAX, MED, SUM_MOD]
VALUES = range(10)

VALUE_P = 0.75
MAX_ARGS = 3
MAX_DEPTH = 4

def generate_tree(depth):
    if depth < MAX_DEPTH:
        r = random.random()
    else:
        r = 1

    if r > VALUE_P:
        value = random.choice(VALUES)
        return value
    else:
        num_values = random.randint(2, MAX_ARGS)
        values = []
        for _ in range(num_values):
            values.append(generate_tree(depth + 1))

        op = random.choice(OPERATORS)
        t = (op, values[0])
        for value in values[1:]:
            t = (t, value)
        t = (t, END)
    return t


def to_string(t, parens=False):
    if isinstance(t, str):
        return t
    elif isinstance(t, int):
        return str(t)
    else:
        if parens:
            return '( ' + to_string(t[0], parens) + ' ' + to_string(t[1], parens) + ' )'
        else:
            return to_string(t[0], parens) + ' ' + to_string(t[1], parens)# + ' '


op2token = dict(zip(list(np.arange(10).astype(str)) + OPERATORS + [END], range(2, 15+2)))

def to_tokens(t):
    string = to_string(t)
    tokens = list(map(lambda x: op2token[x], string.split(' ')))
    return tokens


def to_value(t):
    if not isinstance(t, tuple):
        return t
    l = to_value(t[0])
    r = to_value(t[1])
    if l in OPERATORS:  # Create an unsaturated function.
        return (l, [r])
    elif r == END:  # l must be an unsaturated function.
        if l[0] == MIN:
            return min(l[1])
        elif l[0] == MAX:
            return max(l[1])
        # elif l[0] == FIRST:
        #     return l[1][0]
        # elif l[0] == LAST:
        #     return l[1][-1]
        elif l[0] == MED:
            return int(np.median(l[1]))
        elif l[0] == SUM_MOD:
            return (np.sum(l[1]) % 10)
    elif isinstance(l, tuple):  # We've hit an unsaturated function and an argument.
        return (l[0], l[1] + [r])