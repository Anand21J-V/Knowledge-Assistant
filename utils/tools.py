import math

def calculator_tool(expression: str) -> str:
    """Safely evaluate a math expression using Python's math module."""
    try:
        safe_globals = {"__builtins__": None}
        safe_globals.update(math.__dict__)
        result = eval(expression, safe_globals)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

def dictionary_tool(word: str) -> str:
    """Return a mock dictionary definition."""
    return f"Definition of '{word}': This is a simulated dictionary entry for the word."
