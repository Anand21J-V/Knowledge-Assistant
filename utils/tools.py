# utils/tools.py

import re

def simple_calculator(query: str) -> str:
    try:
        expression = re.findall(r"[-+*/().\\d\\s]+", query)
        if expression:
            return f"The result is: {eval(expression[0])}"
        return "No valid expression found."
    except:
        return "Could not evaluate the expression."

def simple_define_tool(word: str) -> str:
    dictionary = {
        "rag": "Retrieval-Augmented Generation, enhancing LLMs with external documents.",
        "llm": "Large Language Model trained on big text datasets.",
        "embedding": "Numerical vector representation of text.",
        "agent": "A logic-based system that routes tasks to tools or chains."
    }
    return dictionary.get(word.lower(), f"No definition found for: {word}")
