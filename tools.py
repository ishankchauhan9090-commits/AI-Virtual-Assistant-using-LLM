import ast
import math
import operator


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _evaluate(node):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponents must be between -100 and 100.")
        return _BINARY_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_evaluate(node.operand))
    raise ValueError("Use numbers and basic arithmetic operators only.")


def calculator(expression: str):
    """Evaluate a small arithmetic expression without executing Python code."""
    try:
        if not expression or len(expression) > 128:
            raise ValueError("Enter an expression of at most 128 characters.")
        tree = ast.parse(expression, mode="eval")
        result = _evaluate(tree.body)
        if not math.isfinite(float(result)):
            raise ValueError("The result is outside the supported numeric range.")
        return {"success": True, "expression": expression, "result": result}
    except (SyntaxError, TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
        return {"success": False, "expression": expression, "error": str(exc)}
