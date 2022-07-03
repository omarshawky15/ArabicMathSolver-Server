from collections import deque

all_labels = {0: 'alif', 1: 'ba', 2: 'jeem', 3: 'dal', 4: 'dot', 5: 'sen', 6: 'sad', 7: 'ta', 8: 'ayn', 9: 'qaf',
              10: 'lam', 11: 'mim', 12: 'nun', 13: 'ha', 14: 'waw', 15: 'ya', 16: '-', 17: '(', 18: ')', 19: '+',
              20: 'div', 21: 'csc', 22: 'int', 23: 'phi', 24: 'sqrt', 25: 'theta', 26: 'times', 27: 'cos', 28: 'log',
              29: 'sin', 30: 'tan', 31: '0', 32: '1', 33: '2', 34: '3', 35: '4', 36: '5', 37: '6', 38: '7', 39: '8',
              40: '9', 41: 'cot', 42: 'sec'}
variables = ['alif', 'ba', 'jeem', 'dal', 'ra', 'sen', 'sad', 'za', 'ayn', 'fa', 'qaf', 'kaf', 'lam', 'mim', 'nun',
             'waw', 'ya']
nums = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'dot']
math_func = ['csc', 'sqrt', 'cos', 'log', 'sin', 'tan', 'cot', 'sec', 'frac']
constants = {'ta': 'pi', 'ha': 'E'}
right_to_left_brackets = {'(': ')', ')': '(', '[': ']', ']': '[', '{': '}', '}': '{'}


class Symbol_Info:
    def __init__(self, label, x1, y1, x2, y2):
        self.label = label
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2


def educated_parse(symbol_list):
    remove_symbols_indexes = []

    for i in range(len(symbol_list)):
        if i in remove_symbols_indexes:  # Skip updated symbols
            continue

        symbol = symbol_list[i]

        if symbol.label == '-':  # Handle line symbols (equal, minus, division, fraction)
            parseLine(symbol, symbol_list, i, remove_symbols_indexes)

    for index in remove_symbols_indexes:
        symbol_list.pop(index)
    return symbol_list


def parseLine(symbol, symbol_list, i, remove_symbols_indexes):
    if i < len(symbol_list) - 1:  # Handle equal mark
        s1 = symbol_list[i + 1]
        if s1.label == '-' and abs(s1.x1 - symbol.x1) < 30 and abs(s1.x2 - symbol.x2) < 30:
            updateSymbol([symbol, s1], symbol_list, i, '=')
            remove_symbols_indexes.append(i + 1)
            return

    if i < len(symbol_list) - 2:  # Handle division mark
        s1 = symbol_list[i + 1]
        s2 = symbol_list[i + 2]
        if s1.label == '0' and s2.label == '0' and \
                symbol.x1 < s1.x1 < symbol.x2 and symbol.x1 < s2.x1 < symbol.x2 and \
                s1.y1 < symbol.y1 and s2.y1 > symbol.y2:
            updateSymbol([symbol, s1, s2], symbol_list, i, 'div')
            remove_symbols_indexes.append(i + 1)
            remove_symbols_indexes.append(i + 2)
            return

    # Handle fraction mark
    # Assumption: the symbol that belongs to a fraction has to be between the fraction x-axis bounds
    if i >= 2:
        frac = symbol
        j = i - 1
        numerator = 0
        denominator = 0
        while j >= 0:
            s = symbol_list[j]
            if isUpperFrac(s, frac):
                numerator += 1
            if isUnderFrac(s, frac):
                denominator += 1
            j -= 1
        if numerator > 0 and denominator > 0:
            updateSymbol([symbol], symbol_list, i, 'frac')


def updateSymbol(update_list, symbol_list, j, new_label):
    new_x1 = 9223372036854775807
    new_y1 = 9223372036854775807
    new_x2 = -9223372036854775807
    new_y2 = -9223372036854775807
    for i in update_list:
        new_x1 = min(i.x1, new_x1)
        new_y1 = min(i.y1, new_y1)
        new_x2 = max(i.x2, new_x2)
        new_y2 = max(i.y2, new_y2)
    symbol_list[j] = Symbol_Info(new_label, new_x1, new_y1, new_x2, new_y2)


# Convert to mathematical expression
def toExpr(symbol_list, mapped_symbols):
    equation = ""

    i = len(symbol_list) - 1
    while i >= 0:
        symbol = symbol_list[i]

        # Handle power
        if (i < len(symbol_list) - 1) and isProperBase(symbol_list[i + 1]) and isProperExp(symbol_list[i]) and\
                isUpperPow(symbol, symbol_list[i + 1]):
            base = symbol_list[i + 1]
            exp = symbol
            exponent_list = deque()

            exponent_list.appendleft(exp)
            i -= 1
            exp = symbol_list[i]
            while i >= 0 and isUpperPow(exp, base):
                exponent_list.appendleft(exp)
                i -= 1
                exp = symbol_list[i]

            exp_eqn, exp_mapping = toExpr(exponent_list, mapped_symbols)
            if len(exp_eqn) > 1:
                exp_eqn = '(' + exp_eqn + ')'
            equation += '^' + exp_eqn
            continue

        # Add multiplication sign
        if isMultiplication(i, symbol_list):
            equation += '*'
        elif symbol.label == 'times':
            equation += '*'
            i -= 1
            continue

        # Handle fraction
        if symbol.label == 'frac':
            frac = symbol
            i -= 1
            upper_list = deque()
            under_list = deque()

            while i >= 0 and (isUpperFrac(symbol_list[i], frac) or isUnderFrac(symbol_list[i], frac)):
                if isUpperFrac(symbol_list[i], frac):
                    upper_list.appendleft(symbol_list[i])
                elif isUnderFrac(symbol_list[i], frac):
                    under_list.appendleft(symbol_list[i])
                else:
                    break
                i -= 1

            upper_eqn, upper_mapping = toExpr(upper_list, mapped_symbols)
            under_eqn, under_mapping = toExpr(under_list, mapped_symbols)
            mapped_symbols.update(upper_mapping)
            mapped_symbols.update(under_mapping)
            if len(upper_eqn) > 1:
                upper_eqn = '(' + upper_eqn + ')'
            if len(under_eqn) > 1:
                under_eqn = '(' + under_eqn + ')'
            equation += upper_eqn + '/' + under_eqn
            continue

        # Handle square root
        if symbol.label == 'sqrt':
            sqrt = symbol
            i -= 1
            inner_list = deque()

            while i >= 0 and isInner(symbol_list[i], sqrt):
                inner_list.appendleft(symbol_list[i])
                i -= 1

            inner_eqn, inner_mapping = toExpr(inner_list, mapped_symbols)
            mapped_symbols.update(inner_mapping)
            equation += 'sqrt(' + inner_eqn + ')'
            continue

        # Handle log
        # Assumptions:
        # a) log arguments must be between brackets.
        # b) log base is simple expression and doesn't contain any brackets.
        if symbol.label == 'log':
            i -= 1
            log_base = deque()
            log_arg = deque()
            while i >= 0 and symbol_list[i].label != ')':  # Parse log base
                log_base.appendleft(symbol_list[i])
                i -= 1

            open_close_brackets = 0
            if symbol_list[i].label == ')':
                open_close_brackets += 1
            else:   # Assumption (a) violated. Fail-safe: add the next symbol as log argument and continue
                log_arg.appendleft(symbol_list[i])
            i -= 1

            while i >= 0 and open_close_brackets > 0:   # Parse log arguments
                if symbol_list[i].label == ')':
                    open_close_brackets += 1
                elif symbol_list[i].label == '(':
                    open_close_brackets -= 1

                if open_close_brackets > 0:     # Add all symbols except log brackets' opening and closing
                    log_arg.appendleft(symbol_list[i])
                i -= 1

            base_eqn = '10'
            if len(log_base) > 0:
                base_eqn, base_mapping = toExpr(log_base, mapped_symbols)
                mapped_symbols.update(base_mapping)
            log_eqn, log_mapping = toExpr(log_arg, mapped_symbols)
            equation += 'log(' + log_eqn + ',' + base_eqn + ')'
            continue

        if symbol.label in variables:  # Normal variable, for ex: sen
            if symbol.label not in mapped_symbols:
                mapped_symbols[symbol.label] = get_next_eng_symbol(mapped_symbols)
            equation += mapped_symbols[symbol.label]
            i -= 1
        elif symbol.label in constants:  # Like pi or e
            equation += constants[symbol.label]
            i -= 1
        elif symbol.label in right_to_left_brackets:
            equation += right_to_left_brackets[symbol.label]
            i -= 1
        elif symbol.label.isnumeric():
            if symbol.label == 'dot':
                curr_num = '.'
            else:
                curr_num = symbol.label
            i -= 1
            while i >= 0 and symbol_list[i].label in nums:
                if symbol_list[i].label == 'dot':
                    curr_num += '.'
                elif isUpperPow(symbol_list[i], symbol_list[i + 1]) and symbol_list[i + 1].label != 'dot':
                    break
                else:
                    curr_num += symbol_list[i].label
                i -= 1
            equation += curr_num[::-1]  # Reversed number, ex: "21" --> "12"
        else:
            equation += symbol.label
            i -= 1

    return equation, mapped_symbols


# Checks that the base of a power is not any miscellaneous symbol
def isProperBase(symbol):
    return symbol.label in variables or symbol.label in nums or symbol.label in constants or symbol.label == '('


# Checks that the exponent of a power is not any miscellaneous symbol
def isProperExp(symbol):
    return symbol.label in variables or symbol.label in nums or symbol.label in constants or symbol.label in math_func \
           or symbol.label in [')', '-']


# The exponent borders should be to the left of 0.4 of the base rectangle width and above 0.4 of its length
def isUpperPow(exponent, base):
    base_height_limit = base.y1 + (base.y2 - base.y1) * 0.4
    base_width_limit = base.x1 + (base.x2 - base.x1) * 0.4
    return exponent.y2 < base_height_limit and exponent.x2 < base_width_limit and exponent.y1 < base.y1


# The symbol is considered above the fraction line if the symbol center is between the fraction line's ends and the
# symbol's lowest point is above the fraction line's lowest point.
# Assumption: the fraction line is mostly horizontal.
def isUpperFrac(up, frac):
    up_x_center = up.x1 + (up.x2 - up.x1) / 2
    return up.y2 < frac.y2 and frac.x1 < up_x_center < frac.x2


# The symbol is considered below the fraction line if the symbol center is between the fraction line's ends and the
# symbol's highest point is below the fraction line's highest point.
# Assumption: the fraction line is mostly horizontal.
def isUnderFrac(down, frac):
    down_x_center = down.x1 + (down.x2 - down.x1) / 2
    return down.y1 > frac.y1 and frac.x1 < down_x_center < frac.x2


def isInner(symbol, sqrt):
    x_center = symbol.x1 + (symbol.x2 - symbol.x1) / 2
    y_center = symbol.y1 + (symbol.y2 - symbol.y1) / 2
    return sqrt.x1 < x_center < sqrt.x2 and sqrt.y1 < y_center < sqrt.y2


def isMultiplication(i, symbol_list):
    if i >= len(symbol_list) - 1:
        return False
    prev_label = symbol_list[i + 1].label
    curr_label = symbol_list[i].label
    return ((prev_label in nums and prev_label != 'dot') or
            prev_label in variables or prev_label in constants or prev_label == '(') and \
           ((curr_label in nums and curr_label != 'dot') or
            curr_label in variables or curr_label in constants or curr_label == ')' or curr_label in math_func)


def get_next_eng_symbol(mapped_symbols):
    next_eng_symbol = 'x'
    if mapped_symbols is None or len(mapped_symbols) == 0:
        return next_eng_symbol

    curr_eng_symbols = {v for k, v in mapped_symbols.items()}
    for dummy in range(26):
        if next_eng_symbol not in curr_eng_symbols:
            return next_eng_symbol
        else:
            next_eng_symbol = chr(ord('a') + (ord(next_eng_symbol) - ord('a') + 1) % 26)
            if next_eng_symbol == 'e':  # Exclude euler's number from variables
                next_eng_symbol = 'f'
    # Assumption: number of variables won't exceed 26 in one equation
    return next_eng_symbol
