
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
from smart_search import SmartSearch


def define_device():
    """Define the device to be used by PyTorch"""

    # Get the PyTorch version
    torch_version = torch.__version__

    # Print the PyTorch version
    print(f"PyTorch version: {torch_version}", end=" -- ")

    # Check if MPS (Multi-Process Service) device is available on MacOS
    if torch.backends.mps.is_available():
        # If MPS is available, print a message indicating its usage
        print("using MPS device on MacOS")
        # Define the device as MPS
        defined_device = torch.device("mps")
    else:
        # If MPS is not available, determine the device based on GPU availability
        defined_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Print a message indicating the selected device
        print(f"using {defined_device}")

    # Return the defined device
    return defined_device


def get_embedding(text, embedding_model):
    """Get embeddings for a given text using the provided embedding model"""
    
    # Encode the text to obtain embeddings using the provided embedding model
    embedding = embedding_model.encode(text, show_progress_bar=False)
    
    # Convert the embeddings to a list of floats and return
    return embedding.tolist()


def map2embeddings(data, embedding_model):
    """Map a list of texts to their embeddings using the provided embedding model"""
    
    # Initialize an empty list to store embeddings
    embeddings = []

    # Iterate over each text in the input data list
    no_texts = len(data)
    print(f"Mapping {no_texts} pieces of information")
    for i in tqdm(range(no_texts)):
        # Get embeddings for the current text using the provided embedding model
        embeddings.append(get_embedding(data[i], embedding_model))
    
    # Return the list of embeddings
    return embeddings


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
        self.device = define_device()
        self.model, self.tokenizer = self.initialize_model(self.model_name, self.device, self.max_seq_length)
        
    def initialize_model(self, model_name, device, max_seq_length):
        """Initialize a 4-bit quantized causal language model (LLM) and tokenizer with specified settings"""

        # Define the data type for computation
        compute_dtype = getattr(torch, "float16")

        # Define the configuration for quantization
        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=compute_dtype,
        # )

        # Load the pre-trained model with quantization configuration
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.bfloat16
        )

        # Load the tokenizer with specified device and max_seq_length
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            device_map=device,
            max_seq_length=max_seq_length
        )
        
        # Return the initialized model and tokenizer
        return model, tokenizer


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


def generate_summary_and_answer(question, data, searcher, embedding_model, model,
                                max_new_tokens=2048, temperature=0.4, role="expert"):
    """Generate an answer for a given question using context from a dataset"""
    
    # Embed the input question using the provided embedding model
    embeded_question = np.array(get_embedding(question, embedding_model)).reshape(1, -1)
    
    # Find similar contexts in the dataset based on the embedded question
    neighbors, distances = searcher.search_batched(embeded_question)
    
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
             Summarize this context: "{context}" in order to answer the question "{question}" as {role}\
             SUMMARY:
             """.strip() + EOS_TOKEN
    
    # Generate a summary based on the prompt
    results = model.generate_text(prompt, max_new_tokens, temperature)
    
    # Clean the generated summary
    summary = clean_text(results[0].split("SUMMARY:")[-1], EOS_TOKEN)

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
    results = model.generate_text(prompt, max_new_tokens, temperature)
    
    # Clean the generated answer
    answer = clean_text(results[0].split("ANSWER:")[-1], EOS_TOKEN)

    # Return the cleaned answer
    return answer


class AIAssistant():

    """An AI assistant that interacts with users by providing answers based on a provided knowledge base"""
    # embeddings_name="thenlper/gte-large"
    def __init__(self, gemma_model, embeddings_name, temperature=0.4, role="expert"):
        """Initialize the AI assistant."""
        # Initialize attributes
        self.searcher = None
        self.embeddings_name = embeddings_name
        self.knowledge_base = []
        self.temperature = temperature
        self.role = role
        
        # Initialize Gemma model (it can be transformer-based or any other)
        self.gemma_model = gemma_model
        
        # Load the embedding model
        self.embedding_model = SentenceTransformer(self.embeddings_name)
        
    def store_knowledge_base(self, knowledge_base):
        """Store the knowledge base"""
        self.knowledge_base=knowledge_base
        
    def learn_knowledge_base(self, knowledge_base):
        """Store and index the knowledge based to be used by the assistant"""
        # Storing the knowledge base
        self.store_knowledge_base(knowledge_base)
        
        if knowledge_base is not None:
            # Load and index the knowledge base
            print("Indexing and mapping the knowledge base:")
            embeddings = map2embeddings(self.knowledge_base, self.embedding_model)
            self.embeddings = np.array(embeddings).astype(np.float32)

            # Instantiate the searcher for similarity search
            self.index_embeddings()
        else:
            self.embeddings = None
        
    def index_embeddings(self):
        if self.embeddings is not None:
            """Index the embeddings using ScaNN """
            self.searcher = (scann.scann_ops_pybind.builder(db=self.embeddings, num_neighbors=10, distance_measure="dot_product")
                    .tree(num_leaves=min(self.embeddings.shape[0] // 2, 1000), 
                        num_leaves_to_search=100, 
                        training_sample_size=self.embeddings.shape[0])
                    .score_ah(2, anisotropic_quantization_threshold=0.2)
                    .reorder(100)
                    .build()
            )
        else:
            self.searcher = None
        
    def query(self, query):
        """Query the knowledge base of the AI assistant."""
        # Generate and print an answer to the query
        answer = generate_summary_and_answer(query, 
                                             self.knowledge_base, 
                                             self.searcher, 
                                             self.embedding_model, 
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
        
    def save_embeddings(self, filename="embeddings.npy"):
        """Save the embeddings to disk"""
        np.save(filename, self.embeddings)
        
    def load_embeddings(self, filename="embeddings.npy"):
        """Load the embeddings from disk and index them"""
        self.embeddings = np.load(filename)
        # Re-instantiate the searcher
        self.index_embeddings()



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

    print(f"wikipedia_text.sz={len(extracted_texts)}")
    #######################################################################################################################

    # Initialize the name of the embeddings and model
    embeddings_name = "./gte-large"
    model_name = "./gemma-2b-it"

    # Create an instance of AIAssistant with specified parameters
    gemma_ai_assistant = AIAssistant(gemma_model=GemmaHF(model_name), embeddings_name=embeddings_name)

    # Map the intended knowledge base to embeddings and index it
    gemma_ai_assistant.learn_knowledge_base(knowledge_base=extracted_texts)

    # Save the embeddings to disk (for later use)
    gemma_ai_assistant.save_embeddings()

    # Set the temperature (creativity) of the AI assistant and set the role
    gemma_ai_assistant.set_temperature(0.0)
    gemma_ai_assistant.set_role("data science expert whose explanations are useful, clear and complete")


    #######################################################################################################################
    # Run and test
    gemma_ai_assistant.query("What is the difference between data science, machine learning, and artificial intelligence?")

    gemma_ai_assistant.query("Explain how linear regression works")

    gemma_ai_assistant.query("What are decision trees, and how do they work in machine learning?")

    gemma_ai_assistant.query("What is cross-validation, and why is it used in machine learning?")

    gemma_ai_assistant.query("Explain the concept of regularization and its importance in preventing overfitting in machine learning models")
