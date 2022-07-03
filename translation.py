dictionary = {"alif": "أ",
              "ba": "ب",
              "jeem": "جـ",
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
              "-": " -",  # To handle issue with html when converting from english ltr to arabic rtl
              "(": "(",
              ")": ")",
              "[": "[",
              "]": "]",
              "{": "{",
              "}": "}",
              "+": "+",
              "div": "\\",
              "/": "\\",
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
              ",": ","
              }

stop_symbols = ["(", ")", "[", "]", "*", "+", "-", "^", "/", "=", ",", " "]


def toArabicExpr(equation, mapped_symbols):
    arabic_expr = ''

    i = 0
    while i < len(equation):
        curr_symbol = ''

        if str.isalpha(equation[i]) and equation[i] not in ['I', 'E']:
            while i < len(equation) and equation[i] not in stop_symbols:
                curr_symbol += equation[i]
                i += 1
            if curr_symbol in mapped_symbols and mapped_symbols[curr_symbol] in dictionary:
                arabic_expr += dictionary[mapped_symbols[curr_symbol]]
            elif curr_symbol == 'log':
                log_arg = ''
                base = ''
                if equation[i] == '(':
                    i += 1
                    while i < len(equation) and equation[i] not in [',', ')']:
                        log_arg += equation[i]
                        i += 1
                    if i < len(equation) and equation[i] == ')':
                        arabic_expr += dictionary[curr_symbol] + '(' + toArabicExpr(log_arg, mapped_symbols) + ')'
                        i += 1
                    else:
                        i += 1
                        while i < len(equation) and equation[i] != ')':
                            base += equation[i]
                            i += 1
                        i += 1
                        arabic_expr += dictionary[curr_symbol] + "<sub>" + toArabicExpr(base,
                                                                                        mapped_symbols) + "</sub>" + \
                                       '(' + toArabicExpr(log_arg, mapped_symbols) + ')'

                else:
                    print("Error in log parsing")
            elif curr_symbol in dictionary:
                arabic_expr += dictionary[curr_symbol]
            else:
                arabic_expr += curr_symbol
            i -= 1
        elif equation[i] == '^':
            i += 1
            exp_eqn = ''
            if equation[i] == '(':
                i += 1
                open_close_brackets = 1

                # Assumption: brackets in exponent are valid.
                while i < len(equation) and open_close_brackets > 0:
                    if equation[i] == '(':
                        open_close_brackets += 1
                    elif equation[i] == ')':
                        open_close_brackets -= 1

                    if open_close_brackets > 0:
                        exp_eqn += equation[i]
                    i += 1
                i -= 1  # Cancel effect of outer loop

                arabic_expr += " <sup>" + toArabicExpr(exp_eqn, mapped_symbols) + "</sup>"
            else:
                exp_eqn += equation[i]
                arabic_expr += " <sup>" + toArabicExpr(exp_eqn, mapped_symbols) + "</sup>"
        elif equation[i] in dictionary:
            arabic_expr += dictionary[equation[i]]
        else:
            arabic_expr += equation[i]

        i += 1

    return arabic_expr


def refine_expr(expression):
    return expression.replace("**", "^")


def translate_to_arabic_html(expression, solution, mapping):
    expression = refine_expr(str(expression))
    reversed_mapping = {v: k for k, v in mapping.items()}
    arabic_expr = toHtml(toArabicExpr(expression, reversed_mapping))

    arabic_sol = []
    if not isinstance(solution, list):
        curr_arabic_sol = refine_expr(str(solution))
        curr_arabic_sol = toArabicExpr(curr_arabic_sol, reversed_mapping)
        arabic_sol.append(toHtml(curr_arabic_sol))
    else:
        for sol in solution:
            curr_arabic_sol = refine_expr(str(sol))
            curr_arabic_sol = toArabicExpr(curr_arabic_sol, reversed_mapping)
            arabic_sol.append(toHtml(curr_arabic_sol))
    return arabic_expr, arabic_sol


def toHtml(expression):
    return "<html><body>" + expression + "</body></html>"
