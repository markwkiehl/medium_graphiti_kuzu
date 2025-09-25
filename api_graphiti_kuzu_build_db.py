#   Written by:  Mark W Kiehl
#   http://mechatronicsolutionsllc.com/
#   http://www.savvysolutions.info/savvycodesolutions/

# Copyright (C) Mechatroinc Solutions LLC
# License:  MIT


"""
Graphiti graph framework with Kuzu graph database.  

Implements two simple techniques to optimize the content before saving it to the database:

1) The facts were enhanced to maximize the clarity of the relationships in the content, and to extrapolate and state any temporal relationships.
2) A high quality 'description' from each fact is generated and then included with each fact saved to the Graph database.



Graphiti is a framework for building and querying temporally-aware knowledge graphs, specifically tailored for AI agents operating in dynamic environments.
Combine it with a free graph database such as Kuzu for a complete graph system.

Kuzu graph database is an open-source, embedded graph database. It is free to use and released under an MIT license, which is a highly permissive open-source license
https://kuzudb.com/https://github.com/kuzudb/kuzuhttps://docs.kuzudb.com/get-started/




Graphiti Examples:
https://github.com/getzep/graphiti
https://medium.com/coding-nexus/building-dynamic-ai-agents-with-graphiti-the-future-of-real-time-knowledge-graphs-6f912c9a5ad0
https://help.getzep.com/graphiti/core-concepts/custom-entity-and-edge-types
https://help.getzep.com/graphiti/core-concepts/communities
https://help.getzep.com/graphiti/working-with-data/searching
https://medium.com/@saeedhajebi/a-production-ready-api-for-graphitis-powerful-but-flawed-memory-15f17a9c1b41


Kuzu Examples:

https://docs.kuzudb.com/get-started/


PIP INSTALL:

pip install graphiti-core
pip install graphiti-core[kuzu]
pip install dotenv
pip install openai
pip install pydantic
pip install requests
pip install openai


OTHER REQUIREMENTS:

If using Ollama, must use LiteLLM to act as a API adapter or proxy to normalize the API calls Graphiti makes.
https://github.com/BerriAI/litellm

pip install litellm[proxy]


With the Python venv activated, execute the following from the Windows OS command prompt:
    litellm --model ollama/mistral:7b --api_base http://localhost:11434


"""

# Define the script version in terms of Semantic Versioning (SemVer)
# when Git or other versioning systems are not employed.
__version__ = "0.0.0"
from pathlib import Path
print("'" + Path(__file__).stem + ".py'  v" + __version__)


# Imports

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from logging import INFO

from dotenv import load_dotenv

from graphiti_core import Graphiti
from graphiti_core.driver.kuzu_driver import KuzuDriver
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient, LLMConfig

# Using Ollama with Graphiti
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient

from typing import List, Dict, Union, Iterable, Any
import os
import requests
import json
import openai
import pydantic
import re


# Configure logging
logging.basicConfig(
    level=INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


# Set the level of the httpx logger to WARNING.  This hides the Graphiti HTTP request output.
logging.getLogger("httpx").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# API Keys

# IMPORTANT:
# Create a .env file in the same folder as this script and populate it with the API key for OpenAI.
# OPENAI_API_KEY=your_164_char_open_api_key

load_dotenv()

# Set the LLM source to use from:  OPENROUTER.AI, OPENAI, OLLAMA
LLM_SOURCE = "OPENROUTER.AI"


# Your OpenRouter API key will be read from an environment variable.
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# Choose a LLM:
#OPENROUTER_MODEL = "google/gemini-2.5-flash"           # High quality results and very fast.  BEST CHOICE
OPENROUTER_MODEL = "deepseek/deepseek-chat-v3.1"        # FREE and friendly to Graphiti (no response validation errors)
#OPENROUTER_MODEL = "google/gemini-2.0-flash-001"       # Inexpensive LLM
#OPENROUTER_MODEL = "x-ai/grok-4-fast:free"             # FREE
#OPENROUTER_MODEL = "deepseek/deepseek-chat-v3.1:free"  # FREE, but slow.  
#OPENROUTER_MODEL = "google/gemma-3-12b-it:free"        # Model doesn't follow prompt and what it returns generates excessive Graphiti errors:  pydantic_core._pydantic_core.ValidationError: 1 validation error for ExtractedEntities
#OPENROUTER_MODEL = "nvidia/nemotron-nano-9b-v2:free"   # Excessive Graphiti errors:  pydantic_core._pydantic_core.ValidationError: 1 validation error for ExtractedEntities
#OPENROUTER_MODEL = "openai/gpt-4o-mini"                # Excessive Graphiti errors:  pydantic_core._pydantic_core.ValidationError: 4 validation errors for NodeResolutions
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


# Your OpenAI API key will be read from an environment variable.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4.1-mini"     # NO ERRORS | 21 nodes, 28 relationships | Fast
#OPENAI_MODEL = "gpt-3.5-turbo"  # ERROR: max_tokes is too large: 8192
#OPENAI_MODEL = "gpt-4-turbo"    # ERROR: max_tokens is too large: 8192
#OPENAI_MODEL = "gpt-4"          # Excessive Graphiti validation errors from the LLM response.  openai.BadRequestError: Error code: 400 - {'error': {'message': "Invalid parameter: 'response_format' of type 'json_object' is not supported with this model.", 'type': 'invalid_request_error', 'param': 'response_format', 'code': None}}
#OPENAI_MODEL = "gpt-4o"         # Frequent Graphiti validation errors due to the response received.
OPENAI_BASE_URL = "https://api.openai.com/v1"


# Configure variables for using an Ollama LLM.
# https://github.com/getzep/graphiti?tab=readme-ov-file#using-graphiti-with-ollama-local-llm
# The 404 page not found error is happening because Graphiti’s OpenAIGenericClient is trying to call the OpenAI API format (/v1/chat/completions), but you’ve configured it to point at your local Ollama server:
OLLAMA_API_KEY = "ollama"  # Ollama doesn't require a real API key, but some placeholder is needed
# See file litellm_config.yaml
OLLAMA_MODEL = "chat-model-1"
# Point the base URL to the LiteLLM proxy server
OLLAMA_BASE_URL = "http://localhost:4000"


# In order to use Ollama, you must run the LiteLLM proxy server.
# From the Windows command prompt, execute: 
#   litellm --config litellm_config.yaml
# This will start a proxy server (default at http://localhost:4000), accept OpenAI style requests at /v1/chat/completions and /v1/embeddings, and translate them into Ollama's /api/generate and /api/embeddings calls.

# ---------------------------------------------------------------------------


def test_ollama_server_for_graphiti_embeddings(verbose:bool=False):
    """
    Test HTTP POST to the Ollama server for access to an embedding model. 

    Returns TRUE if successful, FALSE otherwise.
    """

    result = False

    # The URL for the LiteLLM proxy's embeddings endpoint
    url = "http://localhost:4000/v1/embeddings"

    # Headers to specify the content type
    headers = {
        "Content-Type": "application/json"
    }

    # The payload (request body) for an embedding call
    # The 'input' field contains the text you want to embed
    payload = {
        "model": "embedding-model-1",     # should match the "model_name" in litellm_config.yml
        "input": "How can I programmatically check that output for the correct content?"
    }

    print("Sending request to LiteLLM for embeddings...")

    try:
        # Send the POST request
        response = requests.post(url, headers=headers, data=json.dumps(payload))

        # Check for a successful response (status code 200)
        if response.status_code == 200:
            if verbose: print("Success! Response from Ollama via LiteLLM:")
            # The response will be a JSON object with the embedding vector
            response_data = response.json()
            
            # Access the embedding vector and print a portion of it
            embedding = response_data['data'][0]['embedding']
            result = True
            if verbose: 
                print(f"Embedding vector (first 5 values): {embedding[:5]}...")
                print(f"Vector dimension: {len(embedding)}")
            
        else:
            print(f"Error! Status code: {response.status_code}")
            print(response.text)

    except requests.exceptions.RequestException as e:
        # Handle any connection errors
        print(f"An error occurred: {e}")

    return result


def test_ollama_server_for_graphiti(verbose:bool=False):
    """
    Validates that calls to Ollama from Graphiti will be successful.
    Change the endpoint 'url' below to refelect if the connection is 
    directly with Ollama, or via the LiteLLM proxy.

    Returns TRUE if successful, FALSE if errors occurred. 

    Usage:
        ollama_is_good_for_graphiti = test_ollama_server_for_graphiti()
        print(f"ollama_is_good_for_graphiti: {ollama_is_good_for_graphiti}")    
    """

    #import requests
    #import json
    #import pydantic
    #from typing import List, Dict, Any

    test_result = False

    # Define a Pydantic model to represent the expected JSON structure
    class ExtractedEntity(pydantic.BaseModel):
        name: str
        type_id: Any

    class ExtractedEntities(pydantic.BaseModel):
        extracted_entities: List[ExtractedEntity]


    def check_ollama_output(output_dict, verbose:bool=False):
        """
        Checks the output from Ollama for the correct content and structure.

        Returns a "check_count" of the number of passed tests.  
        good: check_count = 2 
        perfect: check_count = 3

        Usage:
            check_count = check_ollama_output(

        """
        check_count = 0
        if verbose: print("\n--- Running Output Validation ---")
        try:
            # Access the content string from the nested dictionary
            response_content_str = output_dict['choices'][0]['message']['content']


            # Parse the content string into a Python dictionary
            response_data = json.loads(response_content_str)
            if verbose: 
                print("Successfully parsed response content to a dictionary.")
                print("Parsed Data:", response_data)
            check_count += 1

            # Validate the data using Pydantic
            validated_data = ExtractedEntities(**response_data)
            if verbose: print("Successfully validated JSON schema with Pydantic!")
            check_count += 1

            # Perform a content check
            if validated_data.extracted_entities:
                if verbose: print(f"Success! Found {len(validated_data.extracted_entities)} extracted entities.")
                # Example: Check if a specific entity name exists
                entity_names = [entity.name for entity in validated_data.extracted_entities]
                if "Sir Lancelot" in entity_names:
                    check_count += 1
                    if verbose: print("✓ 'Sir Lancelot' found in the extracted entities.")
                else:
                    if verbose: print("✗ 'Sir Lancelot' was not found.")
            else:
                if verbose: print("Warning: The 'extracted_entities' list is empty.")

        except (KeyError, IndexError) as e:
            print(f"Error: A required key or index was missing from the response dictionary: {e}")
            print("The response structure is not as expected.")
        except json.JSONDecodeError:
            print("Error: The content is not a valid JSON string.")
        except pydantic.ValidationError as e:
            print(f"Error: The JSON structure does not match the Pydantic model.")
            print(e)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

        return check_count

    # The URL for the Ollama OpenAI-compatible API endpoint
    #url = "http://localhost:11434/v1/chat/completions"
    
    # Endpoint below is for using LiteLLM
    url = "http://localhost:4000/v1/chat/completions"

    # Headers to specify the content type
    headers = {
        "Content-Type": "application/json"
    }

    # The payload (request body) in the format of OpenAI's chat completions
    # Make sure you have the specified model (e.g., 'llama2') pulled in Ollama
    payload = {
        "model": "chat-model-1",        # This must match the "model_name" in litellm_config.yaml
        "messages": [
            {
                "role": "system",
                "content": "You are a highly skilled data extraction AI. Your sole task is to analyze the provided text and extract a list of entities and their relationships. Return the result as a single JSON object. The JSON must contain a key named 'extracted_entities' which is an array of objects. Each object must have a 'name' and 'type_id'."
            },
            {
                "role": "user",
                "content": "Here is the text to analyze: 'Sir Lancelot was a brave knight who rescued a prince from a fire-breathing dragon.'"
            }
        ]
    }

    if verbose: print("Sending request to Ollama...")

    try:
        # Send the POST request
        response = requests.post(url, headers=headers, data=json.dumps(payload))

        # Check for a successful response (status code 200)
        if response.status_code == 200:
            if verbose: 
                print("Success! Response from Ollama:")
                print(response.json())

            # Call the validation function with the dictionary
            check_count = check_ollama_output(response.json(), verbose=verbose)
            if check_count > 1: test_result = True
            if verbose: print(f"check_count: {check_count}")
        else:
            print(f"Error! Status code: {response.status_code}")
            print(response.text)

    except requests.exceptions.RequestException as e:
        # Handle any connection errors
        print(f"An error occurred: {e}")

    return test_result


def docs_fake_christopher_diaz(return_chunks=True):
    """
    Returns 6 fake facts, fake metadata, and then QA about them.

    from api_llm_rag_data import docs_fake_christopher_diaz
    chunks, metadata, qna = docs_fake_christopher_diaz()

    Conversion:

    chunks, metadata, qna = docs_fake_christopher_diaz()
    docs = []
    for chunk in chunks:
        docs.append(chunk['content'])

    """

    from typing import List

    # Also make a simple textual document.
    document = """
    Christopher Diaz was a white man born in Ryandbury, Rhode Island.
    He worked for Willis Group LLC as a clinical scientist in Lewischester, MI.
    Christopher married Cindy Lopez in 1965.
    Cindy gave him two sons and one daughter.
    Christopher enjoyed being a sports coach for the public school Griffin, where his kids attended.
    He died at the age of 82 in 2014.
    """
                            

    chunks: List[str] = [
        {"content": "Christopher Diaz was a white man born in Ryandbury, Rhode Island.",
         "section": "life",
        },
        {"content": "He worked for Willis Group LLC as a clinical scientist in Lewischester, MI.",
         "section": "employment",
        },
        {"content": "Christopher married Cindy Lopez in 1965.",
         "section": "family",
        },
        {"content": "Cindy gave him two sons and one daughter.",
         "section": "family",
        },
        {"content": "Christopher enjoyed being a sports coach for the public school Griffin, where his kids attended.",
         "section": "family",
        },
        {"content": "He died at the age of 82 in 2014.",
         "section": "life",
        },
    ]

    metadata: List[str] = [
        {"source": "FakeWikiBio"},
        {"language": "en - English"},
        {"summary": "Biography of Christopher Diaz"},
    ]

    qna: List[str] = [
                    {"Q": "How many children did Christopher Diaz have?", "A": "Three, two boys and one girl"},
                    {"Q": "Who did Christopher Diaz marry and in what year?", "A": "Cindy Lopez in 1965"},
                    {"Q": "Did the children of Christopher Diaz attend a private or public school?", "A": "His children attended a public school named Griffin."},
                    #{"Q": "In what year did Christopher Diaz die?", "A": "2014"},       # I cannot provide information on a private citizen's death.

                    {"Q": "Was Christopher Diaz employed as a sports coach for Willis Group LLC?", "A": "No, he was a clinical scientist for Willis Group LLC"},
                    {"Q": "What is the name of the state where the children of Christopher Diaz attended public school?", "A": "Michigan (likely because that is the state he is employed"},
                    {"Q": "In what year was Christopher Diaz born?", "A": "1932"},
                    {"Q": "What public school did Christopher Dias attend?", "A": "Unknown"},
                    {"Q": "Was Christopher Diaz's first child born in the year 1930?", "A": "No, that year was before Christopher was born."},
                    {"Q": "How old was Christopher Diaz when he married Cindy?", "A": "33 years old (yr_born=yr_death-age:2014-82=1932; age_married=yr_married-yr_born:1965-1932=33)"},
                    ]
    
    return chunks, metadata, qna


def call_llm_openrouter(api_key: str, prompt: str, model: str) -> str:
    """
    Sends a prompt and facts to the OpenRouter API for analysis.

    Args:
        api_key: The OpenRouter API key.
        prompt: The full prompt string to send to the model.
        model: The ID of the model to use.

    Returns:
        The text response from the model, or an error message.
    """
    if not api_key:
        return "Error: OPENROUTER_API_KEY environment variable is not set."

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    # The OpenRouter API endpoint for chat completions.
    API_URL = "https://openrouter.ai/api/v1/chat/completions"

    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
        response_json = response.json()
        
        # Extract and return the model's generated text
        message_content = response_json["choices"][0]["message"]["content"]
        return message_content

    except requests.exceptions.RequestException as e:
        return f"An error occurred: {e}"
    except (KeyError, IndexError) as e:
        return f"Error parsing response from API: {e}\nResponse: {response.text}"


def call_llm_openai(api_key: str, prompt: str, model: str) -> str:
    """
    Sends a prompt and query to the OpenAI API for analysis.

    Args:
        api_key: The OpenAI API key.
        prompt: The full prompt string to send to the model.
        model: The ID of the model to use.

    Returns:
        The text response from the model, or an error message.
    """
    if not api_key:
        return "Error: OPENROUTER_API_KEY environment variable is not set."
    
    # Initialize the async OpenAI client
    #client = openai.AsyncOpenAI(api_key=api_key)
    client = openai.OpenAI(api_key=api_key)

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        
        # Extract and return the model's generated text
        message_content = completion.choices[0].message.content
        return message_content

    except openai.APIStatusError as e:
        return f"OpenAI API error occurred: {e.status_code}\n{e.response}"
    except openai.APIError as e:
        return f"An OpenAI API error occurred: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"


def call_llm_ollama(api_key: str, prompt: str, model: str) -> str:
    """
    Sends a prompt to a local Ollama server for chat completion.

    Local Ollama installation typically runs on localhost and does not require an API key.

    IMPORTANT:  A temperature value of 0.1 rather than default of 0.8 improves robustness of the content returned from 50% to 100%. 

    Args:
        prompt: The full prompt string to send to the model.
        model: The ID of the model to use (e.g., "llama3").

    Returns:
        The text response from the model, or an error message.

    Usage:
        llm_response = call_llm_ollama(OLLAMA_API_KEY, prompt, OLLAMA_MODEL)
    """

    #import requests
    #import json
    #from typing import List, Dict, Any

    # The URL for the Ollama OpenAI-compatible API endpoint
    #url = "http://localhost:11434/v1/chat/completions"
    
    # Endpoint below is for using LiteLLM
    url = "http://localhost:4000/v1/chat/completions"

    # Headers to specify the content type
    headers = {
        "Content-Type": "application/json"
    }

    # The payload (request body) in the format of OpenAI's chat completions
    # Make sure you have the specified model (e.g., 'llama2') pulled in Ollama
    payload = {
        "model": model,        # This must match the "model_name" in litellm_config.yaml model_list
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.1      # 0.0 to 1.0  (default 0.8)
    }

    try:
        # Send the POST request
        response = requests.post(url, headers=headers, data=json.dumps(payload))

        # Check for a successful response (status code 200)
        if response.status_code == 200:
            #print(response.json())
            # {'id': 'chatcmpl-326301cd-3c88-420f-a69e-0af4ff07a1e5', 'created': 1758623426, 'model': 'ollama/llama3.1:8b', 'object': 'chat.completion', 'choices': [{'finish_reason': 'stop', 'index': 0, 'message': {'content': '{}', 'role': 'assistant'}}], 'usage': {'completion_tokens': 4, 'prompt_tokens': 317, 'total_tokens': 321}}
            json_str = response.json()
            content = json_str['choices'][0]['message']['content']
            if len(content) == 0: print(f"\nThe Ollama model '{model}' didn't return anything for the prompt.  The prompt may be too complex for the model.\n {prompt}\n")
            return content

        else:
            print(f"Error! Status code: {response.status_code}")
            print(response.text)
            return response.text

    except requests.exceptions.RequestException as e:
        # Handle any connection errors
        print(f"An error occurred: {e}")
        return None


def optimize_facts_with_llm(facts:str=None):
    """
    Use an LLM to analyze each fact and rewrite it to maximize the clarity of the relationships and to state all temporal relationships.
    This prompt has been extensively tested and engineered to generate the best results with any LLM. 
    """

    print("--- Running LLM Relationship & Temporal Analysis ---")

    # NOTE: If LLM returns a null/empty response because the prompt is too complicated, the following prompt may work:
    '''
    PROMPT = """
        Using the following facts, write a single coherent paragraph formatted in the same way as the input. 
        In your response, replace all pronouns that refer to a person with that person's full name, calculate and add any logically derived dates or ages, and do not invent new facts.
        
    """ + facts 
    '''

    PROMPT = """Revise the following facts into one single, coherent paragraph. 
            Requirements:
            1) Replace EVERY pronoun that refers to a person with that person's FULL NAME, including both first and last name, exactly as it appears in the facts (e.g., he → John Smith, him → John Smith, his → John Smith's, she → Jane Smith, her → Jane Smith's, they → John Smith and Jane Smith). Do not ever shorten a name to only the first name or last name.
            2) Do not use any pronouns that refer to people anywhere in the paragraph.
            3) Calculate and add any dates or ages that can be logically derived. If a derived date is ambiguous by ±1 year because the birthday relative to the event is unknown, state the ambiguity explicitly.
            4) Do not invent facts that are not logically derived from the given facts. Keep all original facts.
            5) The output must be one paragraph formatted in the same way as the input (no bullets, no extra text).
    
    Facts:

    """ + facts    

    match LLM_SOURCE:
        case "OPENROUTER.AI":
            facts = call_llm_openrouter(OPENROUTER_API_KEY, PROMPT, OPENROUTER_MODEL)
        case "OPENAI":
            facts = call_llm_openai(OPENAI_API_KEY, PROMPT, OPENAI_MODEL)
        case "OLLAMA":
            # For 8B or lower parameter Ollama models, the prompt below produces a more robust result.
            PROMPT = """Using the following facts, write a single coherent paragraph formatted in the same way as the input (no bullets, no extra text).  In your response, replace ALL pronouns that refer to a person with that person's FULL NAME, calculate and add any logically derived dates or ages, and do not invent new facts.  
                
            """ + facts 
            facts = call_llm_ollama(OLLAMA_API_KEY, PROMPT, OLLAMA_MODEL)
        case _:
            raise Exception(f"Unknown LLM_SOURCE: {LLM_SOURCE}")
    
    #print("\n--- Analysis Result ---")
    #print(facts)

    # Christopher Diaz, a white man, was born in Ryandbury, Rhode Island, in 1932. Professionally, Christopher Diaz worked as a clinical scientist for Willis Group LLC, an employer located in Lewischester, Michigan. In 1965, at the age of 33, Christopher Diaz married Cindy Lopez. Following the marriage, Cindy Lopez and Christopher Diaz had two sons and one daughter, all born after 1965. Christopher Diaz's children later attended the Griffin public school, where Christopher Diaz served as a sports coach. Christopher Diaz died in 2014 at the age of 82, at which point Christopher Diaz and Cindy Lopez had been married for 49 years.

    return facts


def get_description_for_sentence(fact:str=None):
    """
    Creates a "source_description" that is passed to the Graphiti .add_episode() method.

    The description field is a key part of the data bundling process. 
    The graph database by itself does a good job of defining the actions (e.g., create_node, create_relationship), but the description provides the "why."

    This function employs a hybrid approach where a summary is derived from the fact, and metadata is extracted and presented in a structured way. 
    """
    
    #prompt = f"Generate a single, concise description for a graph episode that would add the following fact: '{fact}'. The description should explain the purpose of the data, not just restate it."

    # Hybrid prompt:
    prompt = f"""
        You are generating a source_description for a graph episode.

        Given the following episode_body:
        {fact}

        Instructions:
        - Write a single line only, no bullets or explanations.  
        - Use exactly this format: [Purpose: ...] [Entities: ...] [Date/Time: ...] [Location: ...] [Domain: ...]  
        - If a field has no data, omit that bracket.  
        - Do not add extra words outside the brackets.  

        Output only the final formatted source_description.
    """

    match LLM_SOURCE:
        case "OPENROUTER.AI":
            description = call_llm_openrouter(OPENROUTER_API_KEY, prompt, OPENROUTER_MODEL)
        case "OPENAI":
            description = call_llm_openai(OPENAI_API_KEY, prompt, OPENAI_MODEL)
        case "OLLAMA":
            description = call_llm_ollama(OLLAMA_API_KEY, prompt, OLLAMA_MODEL)
        case _:
            raise Exception(f"Unknown LLM_SOURCE: {LLM_SOURCE}")

    #print(f"\nfact: {fact}")
    #print(f"description: {description}")
    """
    fact: Christopher Diaz, a white man, was born in Ryandbury, Rhode Island, in 1932
    description: Establishes a personal profile for Christopher Diaz by linking the individual to foundational demographic data and his specific birth event.
    """

    return description


def write_facts_as_json_to_file(path_file:Path, facts:list, verbose:bool=False):
    """
    Write the facts (a list of dictionaries) to a local file specified by path_file.
    """
    # Write JSON file
    #import json
    #from pathlib import Path

    if path_file.is_file(): path_file.unlink()

    with open(path_file, 'w', encoding='utf-8') as f:
        json.dump(facts, f, indent=2)

    if not path_file.is_file(): raise Exception(f"ERROR - file not saved successfully.  {path_file}")


def read_facts_from_json_file(path_file:Path, verbose:bool=False):
    """
    Reads the facts from a local JSON file and return them as a list of dictionaries.
    """

    if not path_file.is_file(): raise Exception(f"ERROR - file not found:  {path_file}")

    with open(path_file, 'r', encoding='utf=8') as f:
        json_str = json.load(f)
    
    return json_str


async def build_db(graphiti, driver, facts:str=None, group_id:str="", verbose:bool=False):
    """
    Build a graph database based on the facts & descriptions in 'facts'.
    group_id is a unique id for the facts.  
    Use the group_id to group facts.  
    """    

    # The group_id field allows you go add a unique identifier to a group of related episodes. 


    # Build the graph episodes from the facts & descriptions.
    # Note: the group_id parameter allows you to create isolated graph namespaces within the same database. 
    episodes = []    
    for fact in facts:
        episodes.append({'content': fact['content'],
                        'type': EpisodeType.text,
                        'description': fact['description'],
                        'group_id': group_id,
        })

    #print(f"\nEpisodes:\n{episodes}\n")
    # [{'content': 'Christopher Diaz, a white man, was born in Ryandbury, Rhode Island, in 1932 (since Christopher Diaz died at the age of 82 in 2014, 2014 - 82 = 1932)', 'type': <EpisodeType.text: 'text'>, 'description': '**The purpose of this graph episode is to establish the birth details of Christopher Diaz, a white man, specifically his birthdate (1932) and birthplace (Ryandbury, Rhode Island), derived from his death in 2014 at age 82.** \n'}, {'content': 'Christopher Diaz worked as a clinical scientist for Willis Group LLC in Lewischester, MI', 'type': <EpisodeType.text: 'text'>, 'description': 'This graph episode links Christopher Diaz to Willis Group LLC, clarifying his professional role and location as a clinical scientist in Lewischester, MI, data crucial for understanding his career trajectory and potential industry connections. \n'}, {'content': 'In 1965, when Christopher Diaz was 33 years old (1965 - 1932 = 33), Christopher Diaz married Cindy Lopez', 'type': <EpisodeType.text: 'text'>, 'description': "This episode establishes the 1965 marital union of Christopher Diaz and Cindy Lopez, specifically highlighting Christopher's age at the time of the marriage to help differentiate this event from potential future or past relationships. \n"}, {'content': 'Christopher Diaz and Cindy Lopez had three children together: two sons and one daughter', 'type': <EpisodeType.text: 'text'>, 'description': 'This graph episode introduces the family structure of Christopher Diaz and Cindy Lopez, establishing their three children (two sons, one daughter) to contextualize future demographic or genealogical analyses involving this couple.'}, {'content': 'Christopher Diaz enjoyed being a sports coach for the public school Griffin, the same school his children attended', 'type': <EpisodeType.text: 'text'>, 'description': "This episode reveals Christopher Diaz's dual role as a dedicated public school coach at Griffin, actively involved in the same community his children attend, highlighting his personal investment in their school environment. \n"}, {'content': 'Christopher Diaz died in 2014 at the age of 82', 'type': <EpisodeType.text: 'text'>, 'description': "This episode adds Christopher Diaz's death (2014, age 82) to ensure the graph's biographical data is accurate and up-to-date for researchers tracking his historical impact. \n"}]

    # Initialize the graph database with graphiti's indices. This only needs to be done once.
    # NOTE: The method below installs and loads the FTS extension for that particular database connection.
    await graphiti.build_indices_and_constraints()
    
    #################################################
    # ADDING EPISODES
    #################################################
    # Episodes are the primary units of information
    # in Graphiti. They can be text or structured JSON
    # and are automatically processed to extract entities
    # and relationships.
    #################################################
    # Add episodes to the graph
    add_episode_failures = 0
    for i, episode in enumerate(episodes):
        # Create a timezone-aware datetime for the reference time.
        reference_time = datetime.now(timezone.utc)
        # Creates a timezone-NAIVE datetime object (because Graphiti doesn't use timezone-aware datetimes
        #reference_time = datetime.now()
        
        success = False
        for attempt in range(3):
            try:
                episode_body = episode['content'] if isinstance(episode['content'], str) else json.dumps(episode['content'])
                await graphiti.add_episode(
                    name=f'biography {i}',
                    episode_body=episode_body,
                    source=episode['type'],
                    source_description=episode['description'],
                    reference_time=reference_time,
                    group_id=episode['group_id'],
                )
                # If successful, break the attempt loop.
                print(f"✅ Successfully added episode {i+1} on attempt {attempt + 1}.")
                success = True
                break
            except json.JSONDecodeError as e:
                print(f"❌ Attempt {attempt + 1} failed due to JSON format error: {e}")
                await asyncio.sleep(2 ** attempt) 
            except Exception as e:
                print(f"❌ Attempt {attempt + 1} failed due to an unexpected error: {e}")
                await asyncio.sleep(2 ** attempt) 
                # NOTE: The error: "1 validation error for EdgeDuplicate duplicate_facts" means that you tried to add duplicate content to the graph database. 
        if not success:
            add_episode_failures += 1
            print(f"🛑 Episode {i+1} NOT added due to errors experienced after {attempt+1} tries.\n{episode_body}")
    if add_episode_failures > 0: print(f"\n🛑 Total add episode failures: {add_episode_failures} out of {episodes} episodes.\n")

def sanitize_string_group_id(input_string):
    """
    Replaces any character in a string that is not alphanumeric, a dash, or an
    underscore with an underscore.

    Args:
        input_string (str): The string to be sanitized.

    Returns:
        str: The sanitized string.
    """
    # The regular expression `[^a-zA-Z0-9_-]` matches any character
    # that is NOT (^) in the specified set of characters.
    # The set includes lowercase letters (a-z), uppercase letters (A-Z),
    # digits (0-9), a hyphen (-), and an underscore (_).
    return re.sub(r'[^a-zA-Z0-9_-]', '_', input_string)


async def main():

    # Optional test of Olamma server.
    #ollama_is_good_for_graphiti = test_ollama_server_for_graphiti()
    #print(f"ollama_is_good_for_graphiti: {ollama_is_good_for_graphiti}")

    #ollama_embeddings_are_good_for_graphiti = test_ollama_server_for_graphiti_embeddings(verbose=False)
    #print(f"ollama_embeddings_are_good_for_graphiti: {ollama_embeddings_are_good_for_graphiti}")

    # Look for dangling database files and delete them if they exist
    if Path(Path.cwd()).joinpath("db_graphiti_kuzu/christopher_diaz.wal").is_file(): raise Exception(f"Delete any *.kuzu and *.wal files found in {Path(Path.cwd()).joinpath("db_graphiti_kuzu/christopher_diaz.wal").parent}")
    if Path(Path.cwd()).joinpath("db_graphiti_kuzu/christopher_diaz.kuzu.wal").is_file(): raise Exception(f"Delete any *.kuzu and *.wal files found in {Path(Path.cwd()).joinpath("db_graphiti_kuzu/christopher_diaz.wal").parent}")

    # Define a database file for the Kuzu graph database.
    path_file_db = Path(Path.cwd()).joinpath("db_graphiti_kuzu/christopher_diaz.kuzu")
    # Create the subfolders if they don't exist.
    if not path_file_db.parent.is_dir(): path_file_db.parent.mkdir(parents=True)

    # Delete the graph database file if it already exists.
    if path_file_db.is_file(): path_file_db.unlink()    

    # Define a local text file to hold the facts compiled in case writing to the graph database fails. 
    path_file_facts = Path(Path.cwd()).joinpath("facts.json")

    # Delete path_file_facts if it exists.  This disables the data recovery option.  Use for development only. 
    if path_file_facts.is_file(): path_file_facts.unlink()

    # Create a Kuzu driver
    print(f"Initializing Kuzu database file {str(path_file_db)} ..")
    driver = KuzuDriver(db=str(path_file_db))

    # Configure the LLM client for Graphiti
    match LLM_SOURCE:
        case "OPENROUTER.AI":
            print(f"Configuring Graphiti LLM to OpenRouter.ai and model: {OPENROUTER_MODEL}")
            # Configure the LLM client for Graphiti to OpenRouter
            llm_config = LLMConfig(
                api_key=OPENROUTER_API_KEY,
                base_url=OPENROUTER_BASE_URL,
                model=OPENROUTER_MODEL
            )
        case "OPENAI":
            print(f"Configuring Graphiti LLM to OpenAI and model: {OPENAI_MODEL}")
            # Configure the LLM client for Graphiti to OpenAI (the default)
            llm_config = LLMConfig(
                api_key=OPENAI_API_KEY,
                base_url=OPENAI_BASE_URL,
                model=OPENAI_MODEL
            )
        case "OLLAMA":
            print(f"Configuring Graphiti LLM to Ollama and model: {OLLAMA_MODEL}")
            # Configure the LLM client for Graphiti to Ollama
            llm_config = LLMConfig(
                api_key="ollama",
                base_url="http://localhost:4000",
                model="chat-model-1",          # chat model
                #small_model=OLLAMA_MODEL_SMALL
            )
            embedder = OpenAIEmbedder(
                config=OpenAIEmbedderConfig(
                    api_key="ollama",
                    embedding_model="embedding-model-1",  # embedding model
                    embedding_dim=768,
                    base_url="http://localhost:4000",
                )
            )

        case _:
            raise Exception(f"Unknown LLM_SOURCE: {LLM_SOURCE}")

    # Create the LLM client instance
    llm_client = OpenAIGenericClient(config=llm_config)
    
    if LLM_SOURCE == "OLLAMA":
        # Initialize Graphiti with the Kuzu driver and the custom LLM client.
        graphiti = Graphiti(graph_driver=driver, 
                            llm_client=llm_client,
                            embedder=embedder,
                            cross_encoder=OpenAIRerankerClient(client=llm_client, config=llm_config),
                            )
    else:
        graphiti = Graphiti(graph_driver=driver, llm_client=llm_client)

    # NOTE: To use a different embedder for vector embeddings with Graphiti, you need to use the OpenAIEmbedder class and pass it a configuration that points to an OpenAI-compatible embeddings API endpoint,

    if not path_file_facts.is_file():

        # Optimize the facts returned from docs_fake_christopher_diaz() and then
        # save it to the Graphiti graph framework with Kuzu graph database.  
        docs, metadata, qna = docs_fake_christopher_diaz()
        facts = ""
        print(f"\nFacts:")
        for doc in docs:
            facts += doc['content'] + "  "
            print(f"{doc['content']}")
        #print(f"\nFacts:\n{facts}")
        # Christopher Diaz was a white man born in Ryandbury, Rhode Island.  He worked for Willis Group LLC as a clinical scientist in Lewischester, MI.  Christopher married Cindy Lopez in 1965.  Cindy gave him two sons and one daughter.  Christopher enjoyed being a sports coach for the public school Griffin, where his kids attended.  He died at the age of 82 in 2014.
        
        # Optimize the relationships using an LLM.
        facts = optimize_facts_with_llm(facts)
        print(f"\nOptimized Facts:")
        for fact in facts.split(". "): print(fact)
        #print(f"\nOptimized Facts:\n{facts}")
        #facts = "Christopher Diaz, a white man, was born in Ryandbury, Rhode Island, in 1932. Professionally, Christopher Diaz worked as a clinical scientist for Willis Group LLC, an employer located in Lewischester, Michigan. In 1965, at the age of 33, Christopher Diaz married Cindy Lopez. Following the marriage, Cindy Lopez and Christopher Diaz had two sons and one daughter, all born after 1965. Christopher Diaz's children later attended the Griffin public school, where Christopher Diaz served as a sports coach. Christopher Diaz died in 2014 at the age of 82, at which point Christopher Diaz and Cindy Lopez had been married for 49 years."
        
        # Build a list of the sentences from the facts. 
        sentences = facts.split('.')
        sentences = [s.strip() for s in sentences if s]
        #print(sentences)
        # ['Christopher Diaz, a white man, was born in Ryandbury, Rhode Island, in 1932', 'Professionally, Christopher Diaz worked as a clinical scientist for Willis Group LLC, an employer located in Lewischester, Michigan', 'In 1965, at the age of 33, Christopher Diaz married Cindy Lopez', 'Following the marriage, Cindy Lopez and Christopher Diaz had two sons and one daughter, all born after 1965', "Christopher Diaz's children later attended the Griffin public school, where Christopher Diaz served as a sports coach", 'Christopher Diaz died in 2014 at the age of 82, at which point Christopher Diaz and Cindy Lopez had been married for 49 years']
        #sentences = ['Christopher Diaz, a white man, was born in Ryandbury, Rhode Island, in 1932', 'Professionally, Christopher Diaz worked as a clinical scientist for Willis Group LLC, an employer located in Lewischester, Michigan', 'In 1965, at the age of 33, Christopher Diaz married Cindy Lopez', 'Following the marriage, Cindy Lopez and Christopher Diaz had two sons and one daughter, all born after 1965', "Christopher Diaz's children later attended the Griffin public school, where Christopher Diaz served as a sports coach", 'Christopher Diaz died in 2014 at the age of 82, at which point Christopher Diaz and Cindy Lopez had been married for 49 years']

        # Get a description for each sentence and then rebuild facts with sentences and the description.
        facts = []
        for sentence in sentences:
            description = get_description_for_sentence(sentence)
            facts.append({'content': sentence,
                            'description': description
            })
        print(f"\nOptimized Facts + Descriptions:")
        for fact in facts:
            print(f"\n{fact['content']}\n{fact['description']}")
        print()
        #facts = [{'content': 'Christopher Diaz was born in 1932 in Ryandbury, Rhode Island, and throughout his adult life worked as a clinical scientist for Willis Group LLC in Lewischester, Michigan', 'description': '“Augment Christopher Diaz’s biographical node with his 1932 birth in Ryandbury, Rhode Island, and his lifelong tenure as a clinical scientist at Willis Group LLC in Lewischester, Michigan to support demographic and career-path analyses.”'}, {'content': 'At the age of 33, Christopher Diaz married Cindy Lopez in 1965; Cindy Lopez and Christopher Diaz went on to have two sons and one daughter', 'description': 'Add Christopher Diaz’s 1965 marriage at age 33 to Cindy Lopez—and their two sons and one daughter—to enrich the family relationship graph for comprehensive genealogical and demographic insights.'}, {'content': 'During the years when the two sons and one daughter of Cindy Lopez and Christopher Diaz attended Griffin Public School, Christopher Diaz served as a sports coach at that institution', 'description': 'Add a time-qualified edge linking Christopher Diaz to a sports-coach role at Griffin Public School over the exact period his two sons and one daughter attended, enabling analysis of parental involvement in school life concurrent with student enrollment.'}, {'content': 'Christopher Diaz died in 2014 at the age of 82', 'description': 'Ingest Christopher Diaz’s 2014 death at age 82 to enrich the graph’s lifespan data for demographic and mortality trend analysis.'}]
        #print(f"\n{facts}")
        # [{'content': 'Christopher Diaz, a white man, was born in Ryandbury, Rhode Island, in 1932 (since Christopher Diaz died at the age of 82 in 2014, 2014 - 82 = 1932)', 'description': '**The purpose of this graph episode is to establish the birth details of Christopher Diaz, a white man, specifically his birthdate (1932) and birthplace (Ryandbury, Rhode Island), derived from his death in 2014 at age 82.** \n'}, {'content': 'Christopher Diaz worked as a clinical scientist for Willis Group LLC in Lewischester, MI', 'description': 'This graph episode links Christopher Diaz to Willis Group LLC, clarifying his professional role and location as a clinical scientist in Lewischester, MI, data crucial for understanding his career trajectory and potential industry connections. \n'}, {'content': 'In 1965, when Christopher Diaz was 33 years old (1965 - 1932 = 33), Christopher Diaz married Cindy Lopez', 'description': "This episode establishes the 1965 marital union of Christopher Diaz and Cindy Lopez, specifically highlighting Christopher's age at the time of the marriage to help differentiate this event from potential future or past relationships. \n"}, {'content': 'Christopher Diaz and Cindy Lopez had three children together: two sons and one daughter', 'description': 'This graph episode introduces the family structure of Christopher Diaz and Cindy Lopez, establishing their three children (two sons, one daughter) to contextualize future demographic or genealogical analyses involving this couple.'}, {'content': 'Christopher Diaz enjoyed being a sports coach for the public school Griffin, the same school his children attended', 'description': "This episode reveals Christopher Diaz's dual role as a dedicated public school coach at Griffin, actively involved in the same community his children attend, highlighting his personal investment in their school environment. \n"}, {'content': 'Christopher Diaz died in 2014 at the age of 82', 'description': "This episode adds Christopher Diaz's death (2014, age 82) to ensure the graph's biographical data is accurate and up-to-date for researchers tracking his historical impact. \n"}]

        # Save the facts to a local JSON file just in case we experience an error later.
        path_file_facts = Path(Path.cwd()).joinpath("facts.json")
        write_facts_as_json_to_file(path_file_facts, facts)

    else:
        # path_file_facts exists.  Recover the data
        # Retrieve the local JSON file with the facts.
        #path_file_facts = Path(Path.cwd()).joinpath("facts.json")
        print(f"Recovering the facts saved to {path_file_facts} ..")
        facts = read_facts_from_json_file(path_file_facts)
        for fact in facts:
            print(f"\n{fact['content']}\n{fact['description']}")
        print()

    try: 

        # Define a unique group_id for the set of facts.
        # group_id must contain only alphanumeric characters, dashes, or underscores
        group_id = "Biography of Christopher Diaz"
        group_id = sanitize_string_group_id(group_id)
        print(f"group_id: '{group_id}'")

        # Save the sentences & descriptions to the Graphiti graph framework with Kuzu graph database.  
        await build_db(graphiti, driver, facts, group_id)

        # Count all nodes
        query_nodes = "MATCH (n) RETURN count(n) AS total_nodes;"
        result_nodes = await driver.execute_query(query_nodes)
        node_count = result_nodes[0][0]
        print(f"\nTotal nodes: {node_count}")

        # Count all relationships
        query_relationships = "MATCH ()-[r]->() RETURN count(r) AS total_relationships;"
        result_relationships = await driver.execute_query(query_relationships)
        relationship_count = result_relationships[0][0]
        print(f"Total relationships: {relationship_count}")

        # Show all relationships
        query_all_rels = "MATCH (n)-[r]->(m) RETURN n.name, label(r), m.name;"
        all_relationships = await driver.execute_query(query_all_rels)
        print("\nAll relationships in the graph:")
        for rel in all_relationships:
            if rel: # Make sure the row is not empty
                for r in rel:
                    print(" -> ".join(r.values()))
        """
        All relationships in the graph:
        Christopher Diaz -> RELATES_TO -> BORN_IN
        Christopher Diaz -> RELATES_TO -> HAS_RACE
        Christopher Diaz -> RELATES_TO -> WORKED_AS
        Christopher Diaz -> RELATES_TO -> WORKED_FOR
        Christopher Diaz -> RELATES_TO -> WORKED_IN_LOCATION
        Willis Group LLC -> RELATES_TO -> EMPLOYER_LOCATION
        Christopher Diaz -> RELATES_TO -> MARRIED
        Christopher Diaz -> RELATES_TO -> PARENT_OF
        Cindy Lopez -> RELATES_TO -> PARENT_OF
        Christopher Diaz -> RELATES_TO -> PARENT_OF
        Cindy Lopez -> RELATES_TO -> PARENT_OF
        Christopher Diaz -> RELATES_TO -> PARENT_OF
        Cindy Lopez -> RELATES_TO -> PARENT_OF
        Christopher Diaz -> RELATES_TO -> ENJOYED_BEING_COACH_AT
        their three children -> RELATES_TO -> CHILDRENS_SCHOOL
        BORN_IN -> RELATES_TO -> Ryandbury, Rhode Island
        HAS_RACE -> RELATES_TO -> white man
        WORKED_FOR -> RELATES_TO -> Willis Group LLC
        WORKED_IN_LOCATION -> RELATES_TO -> Lewischester, MI
        EMPLOYER_LOCATION -> RELATES_TO -> Lewischester, MI
        WORKED_AS -> RELATES_TO -> clinical scientist
        MARRIED -> RELATES_TO -> Cindy Lopez
        PARENT_OF -> RELATES_TO -> their three children
        PARENT_OF -> RELATES_TO -> their three children
        PARENT_OF -> RELATES_TO -> two sons
        PARENT_OF -> RELATES_TO -> two sons
        PARENT_OF -> RELATES_TO -> one daughter
        PARENT_OF -> RELATES_TO -> one daughter
        ENJOYED_BEING_COACH_AT -> RELATES_TO -> public school Griffin
        CHILDRENS_SCHOOL -> RELATES_TO -> public school Griffin
        biography 0 -> MENTIONS -> Christopher Diaz
        biography 1 -> MENTIONS -> Christopher Diaz
        biography 2 -> MENTIONS -> Christopher Diaz
        biography 3 -> MENTIONS -> Christopher Diaz
        biography 4 -> MENTIONS -> Christopher Diaz
        biography 5 -> MENTIONS -> Christopher Diaz
        biography 0 -> MENTIONS -> Ryandbury, Rhode Island
        biography 0 -> MENTIONS -> white man
        biography 1 -> MENTIONS -> Willis Group LLC
        biography 1 -> MENTIONS -> Lewischester, MI
        biography 1 -> MENTIONS -> clinical scientist
        biography 2 -> MENTIONS -> Cindy Lopez
        biography 3 -> MENTIONS -> Cindy Lopez
        biography 3 -> MENTIONS -> their three children
        biography 4 -> MENTIONS -> their three children
        biography 3 -> MENTIONS -> two sons
        biography 3 -> MENTIONS -> one daughter
        biography 4 -> MENTIONS -> public school Griffin
        """

        # If we got this far, no problems, so delete the JSON file that stored the facts. 
        if path_file_facts.is_file(): path_file_facts.unlink()

    finally:

        # Close the database connection
        await graphiti.close()
        print("\nDatabase connection closed")


if __name__ == '__main__':
    pass

    asyncio.run(main())


    