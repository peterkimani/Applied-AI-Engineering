import numpy as np

def simple_numpy_tokenizer(text): 
    #1. Convert to lower case and removing simple punctuation
    clean_text = text.lower().replace(".","").replace("?","")
    tokens = clean_text.split()

    #2. Convert to Numpy array
    token_array = np.array(tokens)

    #3. Create a vocubulary (unique tokens)
    #this represents the internal dictionary our system recognises
    vocab, inverse_indices= np.unique(token_array,return_inverse=True)

    return token_array, vocab, inverse_indices

#Real patient case from the AfyaPlus Model 
patient_query = "My Chest hurts!!!. Is it a heart attack?"

tokens, vocabulary, token_id = simple_numpy_tokenizer(patient_query)

print("---  Tokenisation Results ---")
print(f'Original Text: {patient_query}')
print(f"Tokens: {tokens}")
print(f"Vocabulary: {vocabulary}")
print(f"Token Ids (Numerical Representation):{token_id}")