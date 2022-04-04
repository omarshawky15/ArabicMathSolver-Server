from collections import deque

all_labels = {39: 'times', 28: 'csc', 9: 'ayn', 17: '-', 48: '2', 19: ')', 8: 'ta', 31: 'leq', 14: 'ha', 43: 'log',
              16: 'ya', 52: '6', 50: '4', 56: 'cot', 36: 'sqrt', 3: 'dal', 55: '9', 37: 'sum', 53: '7', 27: 'gt',
              0: 'alif', 38: 'theta', 15: 'waw', 54: '8', 29: 'infty', 34: 'phi', 32: 'lt', 1: 'ba', 20: '[', 30: 'int',
              45: 'tan', 21: ']', 6: 'sen', 51: '5', 57: 'Sec', 25: 'div', 49: '3', 44: 'sin', 47: '1', 33: 'neq',
              7: 'sad', 35: 'sigma', 10: 'qaf', 5: 'dot', 41: 'cos', 2: 'jeem', 23: '}', 40: 'Larr', 11: 'lam', 18: '(',
              42: 'lim', 22: '{', 13: 'nun', 4: 'ra', 24: '+', 26: 'geq', 46: '0', 12: 'mim'}


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

        if symbol.label == '0':  # Handle dot symbols (baa, taa, thaa, ..)
            parseDot(symbol, symbol_list, i, remove_symbols_indexes)

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


def parseDot(symbol, symbol_list, i, remove_symbols_indexes):
    # Todo
    pass


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
    variables = ['alif', 'ba', 'ta', 'tha', 'jeem', 'ha', 'kha', 'dal', 'dhal', 'ra', 'zay', 'sen', 'shin', 'sad',
                 'dad', 'ta', 'za', 'ayn', 'ghayn', 'fa', 'qaf', 'kaf', 'lam', 'mim', 'nun', 'ha', 'waw', 'ya']
    nums = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'dot']
    equation = ""

    i = len(symbol_list) - 1
    while i >= 0:
        symbol = symbol_list[i]
        if (i < len(symbol_list) - 1) and isUpperPow(symbol, symbol_list[i + 1]) and \
                (symbol_list[i].label in variables or symbol_list[i].label in nums) and \
                (symbol_list[i + 1].label in variables or symbol_list[i + 1].label in nums):  # Handle power

            equation += '^' + symbol_list[i].label
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
            equation += '(' + upper_eqn + ')' + '/' + '(' + under_eqn + ')'
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
            equation += ' sqrt(' + inner_eqn + ') '
            continue

        if symbol.label in variables:  # Normal variable, for ex: sen
            if symbol.label not in mapped_symbols:
                mapped_symbols[symbol.label] = get_next_eng_symbol(mapped_symbols)

            if i < len(symbol_list) - 1 and symbol_list[i + 1].label in nums:
                equation += '*'
            equation += mapped_symbols[symbol.label]
            i -= 1
            continue
        elif symbol.label.isnumeric():
            if symbol.label == 'dot':
                curr_num = '.'
            else:
                curr_num = symbol.label
            i -= 1
            while i >= 0 and symbol_list[i].label in nums:
                if symbol_list[i].label == 'dot':
                    curr_num += '.'
                else:
                    curr_num += symbol_list[i].label
                i -= 1
            equation += curr_num[::-1]  # Reversed number, ex: "21" --> "12"
        else:
            equation += symbol.label
            i -= 1

    return equation, mapped_symbols


def isUpperPow(power, s1):
    s1_y_center = s1.y1 + (s1.y2 - s1.y1) / 2
    s1_x_center = s1.x1 + (s1.x2 - s1.x1) / 2
    return power.y2 < s1_y_center and power.x2 < s1_x_center


def isUpperFrac(up, frac):
    up_x_center = up.x1 + (up.x2 - up.x1) / 2
    return up.y2 < frac.y1 and frac.x1 < up_x_center < frac.x2


def isUnderFrac(down, frac):
    down_x_center = down.x1 + (down.x2 - down.x1) / 2
    return down.y1 > frac.y2 and frac.x1 < down_x_center < frac.x2


def isInner(symbol, sqrt):
    x_center = symbol.x1 + (symbol.x2 - symbol.x1) / 2
    y_center = symbol.y1 + (symbol.y2 - symbol.y1) / 2
    return sqrt.x1 < x_center < sqrt.x2 and sqrt.y1 < y_center < sqrt.y2


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
    # Assumption: number of variables won't exceed 26 in one equation
    return next_eng_symbol
