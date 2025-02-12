import random
import numpy as np
from scipy.special import kl_div
from prompts import * 
from call_llm import *
import torch
from transformers import GPT2Tokenizer, GPT2Model


class GPT2XLEncoder:
    def __init__(self, model_name="gpt2-xl"):
        """
        Initializes the GPT-2 XL model for embeddings.
        """
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2Model.from_pretrained(model_name)
        self.model.eval()  # Set model to evaluation mode

    def encode(self, demonstration_set, instance):
        """
        Generates embeddings for the given input using GPT-2 XL.

        Args:
            demonstration_set (list): Not used in this case, kept for compatibility.
            instance (str): The input text to encode.

        Returns:
            np.array: Embedding vector.
        """
        with torch.no_grad():
            inputs = self.tokenizer(instance, return_tensors="pt", truncation=True, max_length=512)
            outputs = self.model(**inputs)
            last_hidden_state = outputs.last_hidden_state  # Shape: [batch_size, seq_len, hidden_dim]
            embedding = last_hidden_state.mean(dim=1).squeeze().numpy()  # Mean pooling
        
        return embedding

class LFNN:
    def __init__(self, training_data, t, encoder_model=GPT2XLEncoder()):
        """
        Initializes the LFNN model by precomputing embeddings for the training dataset.

        Args:
            training_data (list): List of tuples [(x_i, y_i)].
            t (int): Size of the demonstration set.
            encoder_model: The model used to encode the data.
        """
        self.training_data = training_data
        self.t = t
        self.encoder_model = encoder_model

        # Step 1: Precompute embeddings
        self.demonstration_set = random.sample(training_data, t)
        self.anchor_set = [item for item in training_data if item not in self.demonstration_set]

        # Compute and store key vectors for each (a, b) in A
        self.key_vectors = {
            (a, b): self.encode_data(a)
            for (a, b) in self.anchor_set
        }

    def encode_data(self, instance):
        """
        Encodes a given instance using the encoder model and demonstration set.

        Args:
            instance: The data point to encode.

        Returns:
            np.array: Encoded vector representation.
        """
        return self.encoder_model.encode(self.demonstration_set, instance)

    def get_nearest_neighbors(self, query_instance, k):
        """
        Finds the k nearest neighbors of a given query instance using KL divergence.

        Args:
            query_instance: The query data point x.
            k (int): Number of nearest neighbors.

        Returns:
            list: k nearest neighbors [(x_i, y_i)].
        """
        # Compute query embedding
        query_vector = self.encode_data(query_instance)

        # Compute KL divergence for each key vector
        kl_values = {
            (a, b): np.sum(kl_div(query_vector, key_vector))
            for (a, b), key_vector in self.key_vectors.items()
        }

        # Select k pairs with the smallest KL divergence
        nearest_neighbors = sorted(kl_values, key=kl_values.get)[:k]

        return nearest_neighbors

def lfnn_predict(nearest_neighbors, query_instance, prompt_template):
    """
    Uses the nearest neighbor set to generate a few-shot prompt and predict using GPT.

    Args:
        nearest_neighbors (list): The selected k-nearest neighbors [(x_i, y_i)].
        query_instance: The query function to classify.
        prompt_template (str): Template for formatting the few-shot prompts.

    Returns:
        str: The LLM prediction (YES or NO).
    """
    # Format nearest neighbors as few-shot examples
    few_shot_set = [(x, "YES" if y == 1 else "NO") for x, y in nearest_neighbors]

    # Use call_openai_gpt_few_shot with selected few-shot examples
    prediction = call_openai_gpt_few_shot(
        system_msg=lfm_SYSTEM_MSG,
        few_shot_set=few_shot_set,
        user_question=prompt_template.format(code=query_instance),
        prompt_template=prompt_template
    )

    return prediction
