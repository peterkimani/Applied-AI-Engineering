import numpy as np

def simple_numpy_tokenizer(text):
    clean_text = text.lower().replace(".","").replace("?","").replace(",","")
    tokens = clean_text.split()
    return np.array(tokens)

original_query = (
      "Hujambo, it is Juma. I am feeling very hot, my head hurts since yesterday, "
    "and I am coughing. I cannot go to the clinic because of the rain. What should I do?" 
)

summary_query ="patient has fever headache cough since yesterday seeking advice"

#Tokenizing the original query 
original_token = len(simple_numpy_tokenizer(original_query))
summary_token = len(simple_numpy_tokenizer(summary_query))
savings = (original_token - summary_token)/original_token * 100
monthly_savings = (original_token - summary_token)*1000 *30

print(f"Original tokens: {original_token}")
print(f"Summary tokens:  {summary_token}")
print(f"Per-query savings: {savings:.1f}%")
print(f"Monthly token savings (1000 users/day): {monthly_savings:,}")

