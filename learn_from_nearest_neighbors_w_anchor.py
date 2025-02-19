import random
import numpy as np
from scipy.special import kl_div
from prompts import * 
from call_llm import *
import torch
from torch import nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2Model

class AnchorStore(nn.Module):

    def __init__(self, K=1024, dim=50257, knn=1, n_class=2):
        super(AnchorStore, self).__init__()
        self.register_buffer("queue_anchor", torch.randn(K, dim))
        self.register_buffer("queue_label", torch.zeros(K, dtype=torch.long))
        self.queue_label.fill_(-1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.knn = knn
        self.n_class = n_class

    def enqueue(self, anchors, labels):
        ptr = int(self.queue_ptr)
        bs = anchors.shape[0]
        self.queue_anchor[ptr:ptr + bs, :] = anchors
        self.queue_label[ptr:ptr + bs] = labels
        self.queue_ptr[0] = ptr + bs

    def knn_infer(self, query):
        kl_distance = torch.mean(self.queue_anchor[:, None, :] * (self.queue_anchor[:, None, :].log() - query.log()), dim=2).transpose(1, 0)
        if self.knn == 1:
            _, indices = torch.min(kl_distance, dim=1)
            return self.queue_label[indices]
        else:
            _, indices = torch.topk(kl_distance, self.knn, largest=False)
            knn_labels = self.queue_label[indices]
            return torch.mode(knn_labels, dim=1).values


class LearnFromNN:
    def __init__(self, encoder_func, anchor_store):
        """
        Initializes the LearnFromNN class with a specified encoder function and precomputes encoded keys.

        Args:
            encoder_func (callable): A function that takes (anchor_store, instance) and returns an embedding.
            anchor_store (AnchorStore): The anchor store containing precomputed embeddings.
        """
        self.encoder_func = encoder_func
        self.anchor_store = anchor_store
        self.precompute_embeddings()

    def precompute_embeddings(self):
        """
        Computes and stores embeddings in advance for efficient querying.
        """
        encoded_keys = []
        for i in range(len(self.anchor_store.queue_label)):
            if self.anchor_store.queue_label[i] != -1:
                encoded_keys.append(self.encoder_func(self.anchor_store, str(self.anchor_store.queue_label[i].item())))
        self.encoded_keys = torch.tensor(encoded_keys)

    def retrieve_nearest_neighbors(self, instance, k=5):
        """
        Retrieves the k-nearest neighbors from the anchor set based on KL divergence.

        Args:
            instance (str): The input text to encode.
            k (int): Number of nearest neighbors to retrieve.

        Returns:
            list: k nearest examples.
        """
        instance_embedding = self.encoder_func(self.anchor_store, instance)
        neighbors = self.anchor_store.knn_infer(torch.tensor(instance_embedding).unsqueeze(0))
        return neighbors

    def learn_from_nn(self, instance, k=5):
        """
        Learns from nearest neighbors using LLM prompting.

        Args:
            instance (str): The input text.
            k (int): Number of nearest neighbors to retrieve.

        Returns:
            str: LLM-generated response.
        """
        neighbors = self.retrieve_nearest_neighbors(instance, k)
        prompt = PROMPT_INST.format(func="\n".join(map(str, neighbors)) + "\n" + instance)
        response = call_openai_gpt([{"role": "system", "content": SYS_INST}, {"role": "user", "content": prompt}])
        return response


def gpt2xl_encoder(anchor_store, instance):
    """
    Encoder function for GPT-2 XL.

    Args:
        anchor_store (AnchorStore): Not used in this case, kept for compatibility.
        instance (str): The input text to encode.

    Returns:
        np.array: Embedding vector.
    """
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
    model = GPT2Model.from_pretrained("gpt2-xl")
    model.eval()

    with torch.no_grad():
        inputs = tokenizer(instance, return_tensors="pt", truncation=True, max_length=512)
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().detach().cpu()
    return embedding

# Example usage:
if __name__ == "__main__":
    anchor_store = AnchorStore()
    lfnn = LearnFromNN(gpt2xl_encoder, anchor_store)
    
    # Example dataset with anchor (demonstration) and query sets
    dataset = [
        "def add(a, b):\n    return a + b",
        "def subtract(a, b):\n    return a - b",
        "def multiply(a, b):\n    return a * b",
        "def divide(a, b):\n    return a / b"
    ]
    
    anchor_set = dataset[:3]  # First three examples are the anchor set
    query_text = dataset[3]    # Last example is the query
    
    # Store embeddings in the anchor store
    for text in anchor_set:
        embedding = gpt2xl_encoder(anchor_store, text)
        anchor_store.enqueue(torch.tensor([embedding]), torch.tensor([0]))
    
    print("Query Input:", query_text)
    response = lfnn.learn_from_nn(query_text, k=1)
    print("LLM Response:", response)
