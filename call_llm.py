import os
import openai

def call_openai_gpt(messages: list, model: str = "gpt-4o-mini") -> str:
    """
    Calls an OpenAI GPT model with the given messages.

    Args:
        messages (list): A list of messages, each being a dictionary with 'role' and 'content'.
                        Example: [{"role": "system", "content": "You are a helpful assistant."},
                                  {"role": "user", "content": "What is the capital of France?"}]
        model (str): The model name (default is 'gpt-4o-mini').

    Returns:
        str: The model's response.
    """
    if not isinstance(messages, list) or not all(isinstance(msg, dict) and "role" in msg and "content" in msg for msg in messages):
        raise ValueError("Messages must be a list of dictionaries with 'role' and 'content' keys.")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key not found in environment variables.")

    client = openai.OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=256
    )

    return response.choices[0].message.content.strip()

def call_openai_gpt_few_shot(system_msg: str, few_shot_set: list, user_question: str, prompt_template: str, model: str = "gpt-4o-mini") -> str:
    """
    Calls an OpenAI GPT model with a system message, few-shot examples, and a user question.

    Args:
        system_msg (str): The system's role message.
        few_shot_set (list): A list of tuples [(example_data, example_label as 0 or 1)].
        user_question (str): The user's actual question.
        prompt_template (str): Template for formatting the examples.
        model (str): The model name (default is 'gpt-4o-mini').

    Returns:
        str: The model's response.
    """
    messages = [{"role": "system", "content": system_msg}]

    for code, label in few_shot_set:
        # Convert 0 → "NO" and 1 → "YES"
        label_str = "YES" if label == 1 else "NO"
        messages.append({"role": "user", "content": prompt_template.format(code=code)})
        messages.append({"role": "assistant", "content": label_str})

    messages.append({"role": "user", "content": user_question})

    return call_openai_gpt(messages, model)

