# GPT-Rx: AI-Powered Medicine Description Checker
# This script uses GPT-Neo 2.7B to generate a concise, detailed description 
# of a given medicine, including uses, dosage, and potential side effects.
#
# Note: This tool is for educational purposes only and does not replace professional medical advice.
# Requirements: transformers, torch
# Install via: pip install transformers torch

import sys
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import torch

# ------------------------------------------------------------------------------
# Model Setup:
# Load GPT-Neo 2.7B, a free advanced model from EleutherAI.
# ------------------------------------------------------------------------------
model_name = "EleutherAI/gpt-neo-2.7B"  # GPT-Neo 2.7B model
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPTNeoForCausalLM.from_pretrained(model_name)

# ------------------------------------------------------------------------------
# Function: get_medicine_description
# Generates a medicine description using the LLM.
# ------------------------------------------------------------------------------
def get_medicine_description(medicine_name):
    """
    Generates a detailed description for the given medicine.

    Args:
        medicine_name (str): The generic name of the medicine.

    Returns:
        str: A description including uses, dosage, and side effects.
    """
    prompt = (
        f"Provide a detailed description for the medicine '{medicine_name}', "
        "including its generic name, primary uses, dosage recommendations, "
        "and potential side effects. Answer concisely."
    )
    
    # Encode the prompt for the model.
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    # Generate a response using GPT-Neo.
    outputs = model.generate(
        inputs,
        max_length=150,              # Maximum output length
        num_return_sequences=1,      # Generate one response
        no_repeat_ngram_size=2,      # Avoid repeated phrases
        early_stopping=True          # Stop early if needed
    )
    
    # Decode the output tokens into text.
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated response by removing the prompt.
    description = generated_text[len(prompt):].strip()
    
    return description

# ------------------------------------------------------------------------------
# Main function: Runs the application.
# ------------------------------------------------------------------------------
def main():
    print("GPT-Rx: AI Medicine Description Checker")
    print("-----------------------------------------")
    
    # Prompt user to enter the medicine name.
    medicine_name = input("Enter the generic name of the medicine: ").strip()
    if not medicine_name:
        print("Error: No medicine name provided. Exiting...")
        sys.exit(1)
    
    print("\nGenerating description. Please wait...\n")
    
    # Generate and display the medicine description.
    description = get_medicine_description(medicine_name)
    print("Medicine Description:")
    print(description)
    
    print("\nDisclaimer: This tool is for educational purposes only and is not a substitute for professional medical advice.")

# ------------------------------------------------------------------------------
# Run the application if executed as a script.
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
