import random
from prompts import * 
from call_llm import *


def lfm_learn_from_mistakes(training_data, few_shot_set, max_few_shot_size, prompt_template):
    """
    Implements the Learn-from-Mistakes (LFM) algorithm for updating the few-shot set.

    Args:
        training_data (list): List of tuples [(code_snippet, label)], where label is 0 (NO) or 1 (YES).
        few_shot_set (list): Initial few-shot examples [(code_snippet, label)], with labels as 0 or 1.
        max_few_shot_size (int): Maximum size of the few-shot set.
        prompt_template (str): Template for formatting the examples.

    Returns:
        list: The updated few-shot set.
    """
    # Make a copy to modify
    few_shot_set = few_shot_set.copy()
    
    # Iterate over the remaining training data
    for code_snippet, label in training_data:
        if (code_snippet, label) in few_shot_set:
            continue  # Skip if already in few-shot set
        
        # Call the model with few-shot examples
        predicted_label = call_openai_gpt_few_shot(
            system_msg=lfm_SYSTEM_MSG,
            few_shot_set=[(code, "YES" if lbl == 1 else "NO") for code, lbl in few_shot_set],  # Convert labels
            user_question=prompt_template.format(code=code_snippet),
            prompt_template=prompt_template  # Now passed explicitly
        ).strip().upper()  # Normalize case for robustness

        # Convert LLM response to numerical label (1 for YES, 0 for NO)
        if "YES" in predicted_label:
            predicted_label = 1
        elif "NO" in predicted_label:
            predicted_label = 0
        else:
            # If the response is invalid, add the example anyway and mark as incorrect prediction
            few_shot_set.append((code_snippet, label))
            continue  # Skip further checking
        
        # If the prediction is incorrect, add the example to the few-shot set
        if predicted_label != label:
            few_shot_set.append((code_snippet, label))
        
        # Stop if the few-shot set reaches the max size
        if len(few_shot_set) >= max_few_shot_size:
            break

    return few_shot_set



