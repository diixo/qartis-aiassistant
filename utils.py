import torch
import re
import wikipediaapi #pip install wikipedia-api
from tqdm import tqdm


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
