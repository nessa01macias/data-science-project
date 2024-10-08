from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the Gemma model and tokenizer once during app initialization
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b-it",
    torch_dtype=torch.float32,  # Switch to float32 for better compatibility
    device_map=None  # Set to None to avoid offloading to GPU
)

def query_llm(prompt):
    try:
        # Use CPU explicitly
        device = "cpu"
        
        # Tokenize the prompt
        input_ids = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Debugging: Print input_ids to ensure it's correctly generated
        print(f"Input IDs: {input_ids}")

        # Check if the tokenized input is empty or invalid
        if len(input_ids['input_ids'][0]) == 0:
            raise ValueError("Tokenized input is empty or invalid.")
        
        # Safeguard for short input: minimum 20 tokens for text generation
        max_new_tokens = min(len(input_ids['input_ids'][0]) * 2, 100)

        # Generate the response using **input_ids to unpack input_ids and attention_mask
        outputs = model.generate(
            **input_ids,  # This unpacks input_ids and attention_mask as needed
            max_new_tokens=max_new_tokens
        )
        
                # Decode the output tokens back into text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Print and return the decoded text
        print(f"Generated Text: {generated_text}")
        
        return generated_text
    except Exception as e:
        print(f"Error querying LLM: {e}")
        return "An error occurred while processing your request."

