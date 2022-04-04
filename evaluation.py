import sympy as sp


def calculate(equation, mapping):
    solution, error = '', ''
    try:
        sympy_expr = sp.simplify(sp.sympify(equation, evaluate=True))
    except:
        error = "Sympifying equation failed"
        return solution, error

    variables = []
    for k, v in mapping.items():
        variables.append(sp.Symbol(v))

    return sympy_expr, error


def polynomial(expression, mapping):
    solution, error = '', ''
    if '=' not in expression:
        equation = expression
    else:
        splitted_expr = expression.split('=')
        if len(splitted_expr) != 2 or splitted_expr[0] == '' or splitted_expr[1] == '':
            error = 'Missing left or right side'
            return solution, error

        lhs, rhs = splitted_expr[0], splitted_expr[1]
        # Move the right-hand side to the left-hand side with negative sign
        equation = lhs + '- (' + rhs + ')'

    try:
        sympy_eqn = sp.simplify(sp.sympify(equation))
    except:
        error = "Sympifying equation failed"
        return solution, error
    if len(mapping) == 0:
        error = 'No variables in the equation'
        return solution, error

    variables = []
    for k, v in mapping.items():
        variables.append(sp.Symbol(v))

    return sp.solve(sympy_eqn, variables), error


def differentiate(expression, mapping):
    solution, error = '', ''
    try:
        sympy_eqn = sp.sympify(expression)
    except:
        error = "Sympifying equation failed"
        return solution, error

    if len(mapping) != 1:
        error = 'Wrong number of variables to differentiate with respect to'
        return solution, error

    variable = ''
    for k, v in mapping.items():
        variable = sp.Symbol(v)

    return sp.diff(sympy_eqn, variable), error


def integrate(expression, mapping):
    solution, error = '', ''
    try:
        sympy_eqn = sp.sympify(expression)
    except:
        error = "Sympifying equation failed"
        return solution, error

    if len(mapping) != 1:
        error = 'Wrong number of variables to integrate with respect to'
        return solution, error

    variable = ''
    for k, v in mapping.items():
        variable = sp.Symbol(v)

    return sp.integrate(sympy_eqn, variable), error
