import numpy as np

embeddings = {
    "fever":     np.array([0.9, 0.0, 0.3]),
    "pain":      np.array([0.8, 0.1, 0.4]),
    "discharge": np.array([0.1, 0.9, 0.1]),
    "billing":   np.array([0.0, 0.9, 0.1]),
    "admit":     np.array([0.2, 0.8, 0.2]),
    "cough":     np.array([0.8,0.1,0.3])
}

def cosine_similarity(word1, word2):
    vec1, vec2 = embeddings[word1], embeddings[word2]
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


print("--- Context - Disambiguation Via Cosine Similarity")
print(f"cough vs fever:   {cosine_similarity('cough', 'fever'):.4f}")
print(f"cough vs billing: {cosine_similarity('cough', 'billing'):.4f}")