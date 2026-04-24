"""
Calculator tool -- evaluates math expressions safely.

Examples the LLM might send:
  - "2 + 2"
  - "sqrt(144) * 3"
  - "(500 * 0.15) + 42"
"""

import math
from tools.base import BaseTool


class CalculatorTool(BaseTool):
    name = "calculator"
    description = (
        "Evaluate a mathematical expression. Supports basic arithmetic, "
        "exponents (**), and math functions like sqrt(), sin(), cos(), log(). "
        "Use this whenever you need to compute a numerical result."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": 'The math expression to evaluate, e.g. "sqrt(144) * 3"',
            }
        },
        "required": ["expression"],
    }

    # Whitelist of safe names the expression can reference
    _SAFE_NAMES = {
        name: getattr(math, name)
        for name in dir(math)
        if not name.startswith("_")
    }
    _SAFE_NAMES["abs"] = abs
    _SAFE_NAMES["round"] = round
    _SAFE_NAMES["min"] = min
    _SAFE_NAMES["max"] = max

    def run(self, expression: str) -> str:
        try:
            # eval with restricted builtins for safety
            result = eval(expression, {"__builtins__": {}}, self._SAFE_NAMES)
            return str(result)
        except Exception as e:
            return f"Error evaluating '{expression}': {e}"
