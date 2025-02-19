import os
import pickle
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2Model
from scipy.stats import entropy
import heapq


class GPT2XLEncoder:
    """
    An encoder that uses GPT-2 XL to generate an embedding from text.
    The embedding is pooled and transformed into a probability distribution.
    """
    def __init__(self, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
        self.model = GPT2Model.from_pretrained('gpt2-xl')
        self.model.to(self.device)
        self.model.eval()
    
    def __call__(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state  # shape: [1, seq_len, hidden_size]
        embedding = last_hidden_state.mean(dim=1).squeeze(0)  # shape: [hidden_size]
        prob_distribution = F.softmax(embedding, dim=0)
        return prob_distribution.cpu().numpy()


class NearestNeighborLearner:
    """
    Retrieves data from a labeled dataset based on the k nearest neighbors (by KL divergence)
    between encoded representations. Two encoding strategies are available:
    
      - Plain: Uses the raw instance text.
      - Demonstration-based: A preselected demonstration set (data and label) is concatenated
        with the instance text.
    
    """
    def __init__(self, encoder, dataset, cache_file=None, demo_size=1):
        self.encoder = encoder
        self.cache_file = cache_file
        
        # Disable demonstration-set strategy if demo_size is 0.
        self.use_demo = demo_size > 0
        
        if self.use_demo:
            if len(dataset) <= demo_size:
                raise ValueError(f"Demo size {demo_size} must be less than the dataset size {len(dataset)}.")

            indices = list(range(len(dataset)))
            demo_indices = random.sample(indices, demo_size)
            self.demo_set = [dataset[i] for i in demo_indices]
            self.anchor_set = [dataset[i] for i in indices if i not in demo_indices]

            self.demo_text = "\n".join([f"Data: {data}\nLabel: {label}" 
                                        for data, label in self.demo_set]) + "\n"
            
        else:
            self.anchor_set = dataset
            self.demo_set = None
            self.demo_text = ""
        
        # Load cached anchor encodings if available.
        if self.cache_file is not None and os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                cache = pickle.load(f)
            if isinstance(cache, dict):
                self.encoded_data = cache.get('encoded_data', [])
                if self.use_demo:
                    self.demo_set = cache.get('demo_set', self.demo_set)
                    self.demo_text = cache.get('demo_text', self.demo_text)
            elif isinstance(cache, list):
                self.encoded_data = cache
            else:
                raise ValueError("Cache file has an unrecognized format.")
            print(f"Loaded precomputed encodings from {self.cache_file}")
        else:
            # Precompute encodings for each anchor instance.
            self.encoded_data = []  # Each element: (instance, encoding, label)
            for instance, label in self.anchor_set:
                encoding = self.encode_text(instance)
                encoding = self.normalize_encode(encoding)  # Normalize to make it a probability distribution.
                self.encoded_data.append((instance, encoding, label))
            # Save the computed encodings for future runs.
            if self.cache_file is not None:
                cache = {'encoded_data': self.encoded_data}
                if self.use_demo:
                    cache['demo_set'] = self.demo_set
                    cache['demo_text'] = self.demo_text
                with open(self.cache_file, 'wb') as f:
                    pickle.dump(cache, f)
                print(f"Saved precomputed encodings to {self.cache_file}")
    
    def encode_text(self, text):
        """
        Encodes a text string (used for both anchors and queries).
        If the demonstration-set strategy is enabled, the demonstration text is concatenated
        with the input text.
        """
        if self.use_demo and self.demo_text:
            text_to_encode = self.demo_text + f"Data: {text}"
        else:
            text_to_encode = text
        return self.encoder(text_to_encode)
    
    def normalize_encode(self, encoded_text):
        encoded_text = np.array(encoded_text, dtype=np.float64)
        epsilon = 1e-10
        encoded_text += epsilon
        encoded_text /= np.sum(encoded_text)
        return encoded_text

    
    def compute_distance(self, query_encoding, target_encoding):
        query_encoding = np.array(query_encoding, dtype=np.float64)
        epsilon = 1e-10
        query_encoding += epsilon
        query_encoding /= np.sum(query_encoding)
        return entropy(query_encoding, target_encoding)
    
    def retrieve_slow(self, query, k):
        query_encoding = self.encode_text(query)
        distances = []
        for instance, encoding, label in self.encoded_data:
            distance = self.compute_distance(query_encoding, encoding)
            distances.append((distance, instance, label))
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:k]
        return [(instance, label) for (_, instance, label) in k_nearest]
    


    def retrieve(self, query, k):
        query_encoding = self.encode_text(query)
        distances = []

        for instance, encoding, label in self.encoded_data:
            distance = self.compute_distance(query_encoding, encoding)
            distances.append((distance, instance, label))
        
        k_nearest = heapq.nsmallest(k, distances, key=lambda x: x[0])  # O(N log k)
        
        return [(instance, label) for _, instance, label in k_nearest]



def main(dataset, query, cache_file="encoded_data_cache.pkl", demo_size=1, k=5, verbose=False):


    # Automatically generate a cache file name if the default is used.
    if cache_file == "encoded_data_cache.pkl":
        if demo_size > 0:
            cache_file = f"encoded_data_cache_demo_{demo_size}.pkl"
        else:
            cache_file = "encoded_data_cache_plain.pkl"
        print(f"Using cache file: {cache_file}")


    gpt2xl_encoder = GPT2XLEncoder()
    
    nn_learner = NearestNeighborLearner(
        gpt2xl_encoder,
        dataset,
        cache_file=cache_file,
        demo_size=demo_size)
    
    neighbors = nn_learner.retrieve(query, k)

    if verbose:
        print("Nearest neighbors (from anchor set):")
        for instance, label in neighbors:
            print(f"Instance: {instance} | Label: {label}")
    return neighbors

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Nearest Neighbor Learner with GPT-2 XL encoder and demonstration-set encoding strategy."
    )
    parser.add_argument(
        "--cache_file", type=str, default="cache_data/encoded_data_cache.pkl",
        help="Path to a file to load/save precomputed encodings (default: %(default)s)."
    )
    parser.add_argument(
        "--demo_size", type=int, default=1,
        help="Number of demonstration examples to use if use_demo is enabled. Set to 0 for plain encoding (default: %(default)s)."
    )

    parser.add_argument(
        "--k", type=int, default=2,
        help="Number of nearest neighbors to retrieve (default: %(default)s)."
    )
    args = parser.parse_args()
    training_data = [
        ("int x = 0;", 0),
        ("char *ptr; free(ptr);", 1),
        ("int *p; p = malloc(10);", 0),
        ("char *s; strcpy(s, 'Hello');", 1),
        ("int x = 5; x++;", 0),
        ("free(NULL);", 0),
        ("int *arr = malloc(10 * sizeof(int)); arr[10] = 5;", 1)
    ]
    new_code_snippet = "memcpy(dest, src, strlen(src));"  # Buffer overflow risk, should be YES
    main(training_data, new_code_snippet, cache_file=args.cache_file, demo_size=args.demo_size, k=args.k)

