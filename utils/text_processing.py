import re

def extract_question_answer(text: str):
    # Define regex patterns
    user_pattern = r'user\s+(.*)\s+assistant'
    assistant_pattern = r'assistant\s+(.*)'

    # Find user question
    user_match = re.search(user_pattern, text, re.DOTALL)
    question = user_match.group(1).strip() if user_match else None

    # Find assistant answer
    assistant_match = re.search(assistant_pattern, text, re.DOTALL)
    answer = assistant_match.group(1).strip() if assistant_match else None

    return question, answer