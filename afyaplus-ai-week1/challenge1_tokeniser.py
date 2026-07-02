import numpy as np

def emergency_tokeniser (text): 
    """ Split text into tokens while preserving '!' as seperate tokens"""
    text = text.lower()
    # Todo 1: Padding ! with spaces so split treats them as seperate tokens. 
    text = text.replace("!"," ! ").replace(".","")
    # Spliting the text
    text = text.split()

    # wrapping the function in np array
    tokens = np.array(text)

    return tokens

# Test Case
samples = [ "Help!!!", "I cannot breathe!", "My chest hurts."]

for s in samples: 
    print(s, "->", emergency_tokeniser(s))