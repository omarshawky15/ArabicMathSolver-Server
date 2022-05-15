dictionary = {"alif": "أ",
              "ba": "ب",
              "jeem": "ت",
              "dal": "د",
              "ra": "ر",
              ".": ",",
              "sen": "س",
              "sad": "ص",
              "pi": "ط",
              "ayn": "ع",
              "qaf": "ق",
              "lam": "ل",
              "mim": "م",
              "nun": "ن",
              "ha": "هـ",
              "waw": "و",
              "ya": "ى",
              "E": "هـ",
              "I": "ت",
              "-": "-",
              "(": "(",
              ")": ")",
              "[": "[",
              "]": "]",
              "{": "{",
              "}": "}",
              "+": "+",
              "div": "/",
              "/": "/",
              "geq": "≥",
              "gt": ">",
              "csc": "قتا",
              "infty": "∞",
              "int": "∫",
              "leq": "≤",
              "lt": "<",
              "neq": "≠",
              "phi": "Φ",
              "sigma": "Σ",
              "sqrt": "√",
              "sum": "Σ",
              "theta": "θ",
              "times": "×",
              "larr": "←",
              "cos": "جتا",
              "lim": "نها",
              "log": "لـو",
              "sin": "جا",
              "tan": "ظا",
              "0": "٠",
              "1": "١",
              "2": "٢",
              "3": "٣",
              "4": "٤",
              "5": "٥",
              "6": "٦",
              "7": "٧",
              "8": "٨",
              "9": "٩",
              "cot": "ظتا",
              "sec": "قا",
              " ": "",
              "*": "*",
              "=": "=",
              }

stop_symbols = ["(", ")", "[", "]", "*", "+", "-", "^", "="]


def toArabicExpr(equation, mapped_symbols):
    arabic_expr = ''

    i = 0
    while i < len(equation):
        curr_symbol = ''

        if str.isalpha(equation[i]) and equation[i] not in ['I', 'E']:
            while i < len(equation) and equation[i] not in stop_symbols:
                curr_symbol += equation[i]
                i += 1
            if curr_symbol in mapped_symbols:
                arabic_expr += dictionary[mapped_symbols[curr_symbol]]
            elif curr_symbol == 'log':
                log_parameter = ''
                base = ''
                if equation[i] == '(':
                    i += 1
                    while i < len(equation) and equation[i] != ',':
                        log_parameter += equation[i]
                        i += 1
                    i += 1
                    while i < len(equation) and equation[i] != ')':
                        base += equation[i]
                        i += 1
                    i += 1
                    arabic_expr += dictionary[curr_symbol] + "<sub>" + toArabicExpr(base, mapped_symbols) + "</sub>" + \
                                   '(' + toArabicExpr(log_parameter, mapped_symbols) + ')'
                else:
                    print("Error in log parsing")
            elif curr_symbol in dictionary:
                arabic_expr += dictionary[curr_symbol]
            else:
                print("General error")
            i -= 1
        elif equation[i] == '^':
            i += 1
            exp_eqn = ''
            if equation[i] == '(':
                i += 1
                # Assuming only 1 pair of brackets exist in the exponent
                while i < len(equation) and equation[i] != ')':
                    exp_eqn += equation[i]
                    i += 1
                arabic_expr += "<sup>" + toArabicExpr(exp_eqn, mapped_symbols) + "</sup>"
            else:
                exp_eqn += equation[i]
                arabic_expr += "<sup>" + toArabicExpr(exp_eqn, mapped_symbols) + "</sup>"
        else:
            arabic_expr += dictionary[equation[i]]

        i += 1

    if arabic_expr[0] in ['+', '-']:
        arabic_expr = arabic_expr[1:] + arabic_expr[0]
    return arabic_expr


def translate_to_arabic_html(expression, solution, mapping):
    reversed_mapping = {v: k for k, v in mapping.items()}
    arabic_expr = toHtml(toArabicExpr(expression, reversed_mapping))

    arabic_sol = []
    if not isinstance(solution, list):
        curr_arabic_sol = toArabicExpr(str(solution), reversed_mapping)
        arabic_sol.append(toHtml(curr_arabic_sol))
    else:
        for sol in solution:
            curr_arabic_sol = toArabicExpr(str(sol), reversed_mapping)
            arabic_sol.append(toHtml(curr_arabic_sol))
    return arabic_expr, arabic_sol


def toHtml(expression):
    return "<html>\n<body>\n<p>" + expression + "</p>\n</body>\n</html>"
