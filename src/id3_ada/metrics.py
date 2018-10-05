import numpy as np

def entropy(x):
    event_counts = np.unique(x, return_counts = True)[1]
    event_probs = event_counts / event_counts.sum()
    code_lengths = -1 * np.log2(event_probs)
    return (event_probs * code_lengths).sum()
