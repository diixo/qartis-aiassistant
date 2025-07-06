import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore")

import re
import numpy as np
import pandas as pd
from tqdm import tqdm
#import scann        #pip install scann
import wikipediaapi #pip install wikipedia-api

import torch

import transformers
from transformers import (AutoModelForCausalLM, 
                          AutoTokenizer, 
                          BitsAndBytesConfig,
                         )
from sentence_transformers import SentenceTransformer
import bitsandbytes as bnb # pip install -q -i https://pypi.org/simple/ bitsandbytes


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
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )

        # Load the pre-trained model with quantization configuration
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            quantization_config=bnb_config,
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
    
    def __init__(self, gemma_model, embeddings_name="thenlper/gte-large", temperature=0.4, role="expert"):
        """Initialize the AI assistant."""
        # Initialize attributes
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
        
        # Load and index the knowledge base
        print("Indexing and mapping the knowledge base:")
        embeddings = map2embeddings(self.knowledge_base, self.embedding_model)
        self.embeddings = np.array(embeddings).astype(np.float32)
        
        # Instantiate the searcher for similarity search
        self.index_embeddings()
        
    def index_embeddings(self):
        """Index the embeddings using ScaNN """
        self.searcher = (scann.scann_ops_pybind.builder(db=self.embeddings, num_neighbors=10, distance_measure="dot_product")
                 .tree(num_leaves=min(self.embeddings.shape[0] // 2, 1000), 
                       num_leaves_to_search=100, 
                       training_sample_size=self.embeddings.shape[0])
                 .score_ah(2, anisotropic_quantization_threshold=0.2)
                 .reorder(100)
                 .build()
           )
        
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



# Pre-compile the regular expression pattern for better performance
BRACES_PATTERN = re.compile(r'\{.*?\}|\}')

def remove_braces_and_content(text):
    """Remove all occurrences of curly braces and their content from the given text"""
    return BRACES_PATTERN.sub('', text)

def clean_string(input_string):
    """Clean the input string."""
    
    # Remove extra spaces by splitting the string by spaces and joining back together
    cleaned_string = ' '.join(input_string.split())
    
    # Remove consecutive carriage return characters until there are no more consecutive occurrences
    cleaned_string = re.sub(r'\r+', '\r', cleaned_string)
    
    # Remove all occurrences of curly braces and their content from the cleaned string
    cleaned_string = remove_braces_and_content(cleaned_string)
    
    # Return the cleaned string
    return cleaned_string


def extract_wikipedia_pages(wiki_wiki, category_name):
    """Extract all references from a category on Wikipedia"""
    
    # Get the Wikipedia page corresponding to the provided category name
    category = wiki_wiki.page("Category:" + category_name)
    
    # Initialize an empty list to store page titles
    pages = []
    
    # Check if the category exists
    if category.exists():
        # Iterate through each article in the category and append its title to the list
        for article in category.categorymembers.values():
            pages.append(article.title)
    
    # Return the list of page titles
    return pages


def get_wikipedia_pages(categories):
    """Retrieve Wikipedia pages from a list of categories and extract their content"""
    
    # Create a Wikipedia object
    wiki_wiki = wikipediaapi.Wikipedia('Gemma AI Assistant (gemma@example.com)', 'en')
    
    # Initialize lists to store explored categories and Wikipedia pages
    explored_categories = []
    wikipedia_pages = []

    # Iterate through each category
    print("- Processing Wikipedia categories:")
    for category_name in categories:
        print(f"\tExploring {category_name} on Wikipedia")
        
        # Get the Wikipedia page corresponding to the category
        category = wiki_wiki.page("Category:" + category_name)
        
        # Extract Wikipedia pages from the category and extend the list
        wikipedia_pages.extend(extract_wikipedia_pages(wiki_wiki, category_name))
        
        # Add the explored category to the list
        explored_categories.append(category_name)

    # Extract subcategories and remove duplicate categories
    categories_to_explore = [item.replace("Category:", "") for item in wikipedia_pages if "Category:" in item]
    wikipedia_pages = list(set([item for item in wikipedia_pages if "Category:" not in item]))
    
    # Explore subcategories recursively
    while categories_to_explore:
        category_name = categories_to_explore.pop()
        print(f"\tExploring {category_name} on Wikipedia")
        
        # Extract more references from the subcategory
        more_refs = extract_wikipedia_pages(wiki_wiki, category_name)

        # Iterate through the references
        for ref in more_refs:
            # Check if the reference is a category
            if "Category:" in ref:
                new_category = ref.replace("Category:", "")
                # Add the new category to the explored categories list
                if new_category not in explored_categories:
                    explored_categories.append(new_category)
            else:
                # Add the reference to the Wikipedia pages list
                if ref not in wikipedia_pages:
                    wikipedia_pages.append(ref)

    # Initialize a list to store extracted texts
    extracted_texts = []
    
    # Iterate through each Wikipedia page
    print("- Processing Wikipedia pages:")
    for page_title in tqdm(wikipedia_pages):
        try:
            # Make a request to the Wikipedia page
            page = wiki_wiki.page(page_title)

            # Check if the page summary does not contain certain keywords
            if "Biden" not in page.summary and "Trump" not in page.summary:
                # Append the page title and summary to the extracted texts list
                if len(page.summary) > len(page.title):
                    extracted_texts.append(page.title + " : " + clean_string(page.summary))

                # Iterate through the sections in the page
                for section in page.sections:
                    # Append the page title and section text to the extracted texts list
                    if len(section.text) > len(page.title):
                        extracted_texts.append(page.title + " : " + clean_string(section.text))
                        
        except Exception as e:
            print(f"Error processing page {page.title}: {e}")
                    
    # Return the extracted texts
    return extracted_texts


categories = ["Machine_learning", "Data_science", "Statistics", "Deep_learning", "Artificial_intelligence"]
extracted_texts = get_wikipedia_pages(categories)
print("Found", len(extracted_texts), "Wikipedia pages")


wikipedia_data_science_kb = pd.DataFrame(extracted_texts, columns=["wikipedia_text"])
wikipedia_data_science_kb.to_csv("wikipedia_data_science_kb.csv", index=False)
wikipedia_data_science_kb.head()

################# 6

# Initialize the name of the embeddings and model
embeddings_name = "thenlper/gte-large"
model_name = "/kaggle/input/gemma/transformers/2b-it/1"

# Create an instance of AIAssistant with specified parameters
gemma_ai_assistant = AIAssistant(gemma_model=GemmaHF(model_name), embeddings_name=embeddings_name)

# Map the intended knowledge base to embeddings and index it
gemma_ai_assistant.learn_knowledge_base(knowledge_base=extracted_texts)

# Save the embeddings to disk (for later use)
gemma_ai_assistant.save_embeddings()

# Set the temperature (creativity) of the AI assistant and set the role
gemma_ai_assistant.set_temperature(0.0)
gemma_ai_assistant.set_role("data science expert whose explanations are useful, clear and complete")

