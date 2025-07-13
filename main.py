
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
#import scann
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

from transformers import BitsAndBytesConfig
#import bitsandbytes as bnb

import utils
from smart_search import SmartSearch_FAISS


def clean_text(txt, EOS_TOKEN):
    """Clean text by removing specific tokens and redundant spaces"""
    txt = (txt
           .replace(EOS_TOKEN, "") # Replace the end-of-sentence token with an empty string
           .replace("**", "")      # Replace double asterisks with an empty string
           .replace("<pad>", "")   # Replace "<pad>" with an empty string
           .replace("  ", " ")     # Replace double spaces with single spaces
          ).strip()                # Strip leading and trailing spaces from the text
    return txt


def add_indefinite_article(role_name):
    """Check if a role name has a determinative adjective before it, and if not, add the correct one"""
    
    # Check if the first word is a determinative adjective
    determinative_adjectives = ["a", "an", "the"]
    words = role_name.split()
    if words[0].lower() not in determinative_adjectives:
        # Use "a" or "an" based on the first letter of the role name
        determinative_adjective = "an" if words[0][0].lower() in "aeiou" else "a"
        role_name = f"{determinative_adjective} {role_name}"

    return role_name


class GemmaHF():
    """Wrapper for the Transformers implementation of Gemma"""
    
    def __init__(self, model_name, max_seq_length=2048):
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        
        # Initialize the model and tokenizer
        print("\nInitializing model:")
        self.device = utils.define_device()
        self.model, self.tokenizer = self.initialize_model(self.model_name, self.device, self.max_seq_length)
        
    def initialize_model(self, model_name, device, max_seq_length):
        """Initialize a 4-bit quantized causal language model (LLM) and tokenizer with specified settings"""
        # Load the pre-trained model with quantization configuration
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.bfloat16,
        )

        # Load the tokenizer with specified device and max_seq_length
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            device_map=device,
            max_seq_length=max_seq_length
        )
        
        # Return the initialized model and tokenizer
        return model, tokenizer


    def max_position_embeddings(self):
        return self.model.config.max_position_embeddings


    def generate_text(self, prompt, max_new_tokens=2048, temperature=0.0):
        """Generate text using the instantiated tokenizer and model with specified settings"""
    
        # Encode the prompt and convert to PyTorch tensor
        input_ids = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)

        # Determine if sampling should be performed based on temperature
        do_sample = True if temperature > 0 else False

        # Generate text based on the input prompt
        outputs = self.model.generate(**input_ids, 
                                      max_new_tokens=max_new_tokens, 
                                      do_sample=do_sample, 
                                      temperature=temperature
                                     )

        # Decode the generated output into text
        results = [self.tokenizer.decode(output) for output in outputs]

        # Return the list of generated text results
        return results


def generate_summary_and_answer(question, data, searcher, model,
                                max_new_tokens=2048, temperature=0.4, role="expert"):
    """Generate an answer for a given question using context from a dataset"""
    
    # Find similar contexts in the dataset based on the embedded question
    neighbors, distances = searcher.search_batched(question)
    
    # Extract context from the dataset based on the indices of similar contexts
    context = " ".join([data[pos] for pos in np.ravel(neighbors)])
    
    # Get the end-of-sentence token from the tokenizer
    try:
        EOS_TOKEN = model.tokenizer.eos_token
    except:
        EOS_TOKEN = "<eos>"

    
    # Add a determinative adjective to the role
    role = add_indefinite_article(role)
    
    # Generate a prompt for summarizing the context
    prompt = f"""
        <start_of_turn>system
        You are {role}
        <end_of_turn>
        <start_of_turn>user
        in order to answer the question: "{question}", summarize this: "{context}"
        <end_of_turn>
        <start_of_turn>model
        SUMMARY:
        """.strip()

    # Generate a summary based on the prompt
    results = model.generate_text(prompt, model.max_position_embeddings(), temperature)[0]
    
    # Clean the generated summary
    summary = clean_text(results.split("SUMMARY:")[-1], EOS_TOKEN)

    print("SUMMARY:", results, ">>>>>")

    # Generate a prompt for providing an answer
    prompt = f"""
            Here is the context: {summary}
            Using the relevant information from the context 
            and integrating it with your knowledge,
            provide an answer as {role} to the question: {question}.
            If the context doesn't provide
            any relevant information answer with 
            [I couldn't find a good match in my
            knowledge base for your question, 
            hence I answer based on my own knowledge] \
            ANSWER:
            """.strip() + EOS_TOKEN

    # Generate an answer based on the prompt
    results = model.generate_text(prompt, max_new_tokens, temperature)[0]
    
    # Clean the generated answer
    answer = clean_text(results.split("ANSWER:")[-1], EOS_TOKEN)

    # Return the cleaned answer
    return answer


class AIAssistant():

    """An AI assistant that interacts with users by providing answers based on a provided knowledge base"""
    # embeddings_name="thenlper/gte-large"
    def __init__(self, gemma_model, embeddings_name, temperature=0.4, role="expert"):
        """Initialize the AI assistant."""
        # Initialize attributes
        self.searcher = SmartSearch_FAISS(embeddings_name)
        self.embeddings_name = embeddings_name
        self.knowledge_base = []
        self.temperature = temperature
        self.role = role
        
        # Initialize Gemma model (it can be transformer-based or any other)
        self.gemma_model = gemma_model

        
    def store_knowledge_base(self, knowledge_base):
        """Store the knowledge base"""
        self.knowledge_base=knowledge_base
        
    def learn_knowledge_base(self, knowledge_base):
        """Store and index the knowledge based to be used by the assistant"""
        # Storing the knowledge base
        self.store_knowledge_base(knowledge_base)
        result = False
        
        if knowledge_base is not None:
            # Load and index the knowledge base
            print("Indexing and mapping the knowledge base:")

            # Instantiate the searcher for similarity search
            result = self.searcher.add_texts_to_index(knowledge_base, utils.define_device())
        return result

        
    def query(self, query):
        """Query the knowledge base of the AI assistant."""
        # Generate and print an answer to the query
        answer = generate_summary_and_answer(query, 
                                             self.knowledge_base, 
                                             self.searcher, 
                                             self.gemma_model,
                                             temperature=self.temperature,
                                             role=self.role)
        print(answer)
        
    def set_temperature(self, temperature):
        """Set the temperature (creativity) of the AI assistant."""
        self.temperature = temperature
        
    def set_role(self, role):
        """Define the answering style of the AI assistant."""
        self.role = role
        
    def save_embeddings(self, filename="embeddings.bin"):
        """Save the embeddings to disk"""
        self.searcher.save_index(filename)
        
    def load_embeddings(self, filename="embeddings.bin"):
        """Load the embeddings from disk and index them"""
        return self.searcher.open_file(filename)


def filtering(texts_data: list):
    filtered = []
    filter = [
        "machine learning", "data science", "deep learning", "artificial intelligence", "linear regression", "decission tree",
        "cross-validation", "matrix regularization",]
    for sent in texts_data:
        for f in filter:
            if str(sent).lower().find(f.lower()) >= 0:
                filtered.append(sent)
                break
    return filtered


if __name__ == '__main__':

    extracted_texts = []

    csv_file = "wikipedia_data_science_kb.csv"

    if Path(csv_file).exists():
        df = pd.read_csv(csv_file)
        wikipedia_text = df['wikipedia_text']

        extracted_texts = wikipedia_text.tolist()
    else:
        categories = ["Machine_learning", "Data_science", "Statistics", "Deep_learning", "Artificial_intelligence"]
        extracted_texts = utils.get_wikipedia_pages(categories)

        if extracted_texts is not None:
            wikipedia_data_science_kb = pd.DataFrame(extracted_texts, columns=["wikipedia_text"])
            wikipedia_data_science_kb.to_csv(csv_file, index=False)
            wikipedia_data_science_kb.head()

    filtered = filtering(extracted_texts)
    print(f"wikipedia_text.sz={len(extracted_texts)}, filtered.sz={len(filtered)}")

    # restore original by filtered texts_data
    extracted_texts = filtered
    #######################################################################################################################

    # Initialize the name of the embeddings and model
    embeddings_name = "./gte-large"
    model_name = "./gemma-2b-it"

    # Create an instance of AIAssistant with specified parameters
    assistant = AIAssistant(gemma_model=GemmaHF(model_name), embeddings_name=embeddings_name)

    if assistant.load_embeddings():
        print("AIAssistant::loaded_embeddings OK.")
        assistant.store_knowledge_base(extracted_texts)
    else:
        # Map the intended knowledge base to embeddings and index it
        assistant.learn_knowledge_base(extracted_texts)

        # Save the embeddings to disk (for later use)
        assistant.save_embeddings()


    # Set the temperature (creativity) of the AI assistant and set the role
    assistant.set_temperature(0.0)
    assistant.set_role("data science expert whose explanations are useful, clear and complete")


    #######################################################################################################################
    # Run and test
    assistant.query("What is the difference between data science, machine learning, and artificial intelligence?")

    exit(0)

    assistant.query("Explain how linear regression works")

    assistant.query("What are decision trees, and how do they work in machine learning?")

    assistant.query("What is cross-validation, and why is it used in machine learning?")

    assistant.query("Explain the concept of regularization and its importance in preventing overfitting in machine learning models")
