#   Written by:  Mark W Kiehl
#   http://mechatronicsolutionsllc.com/
#   http://www.savvysolutions.info/savvycodesolutions/

# Copyright (C) Mechatroinc Solutions LLC
# License:  MIT


"""
Graphiti graph framework with Kuzu graph database.  

Implements two simple techniques to optimize the content before saving it to the database:

1) The facts were enhanced to maximize the clarity of the relationships in the content, and to extrapolate and state any temporal relationships.
2) A high quality 'description' from each fact was generated and then included with each fact saved to the Graph database.

Uses OpenRouter.ai for the LLM. 



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

"""

# Define the script version in terms of Semantic Versioning (SemVer)
# when Git or other versioning systems are not employed.
__version__ = "0.0.0"
from pathlib import Path
print("'" + Path(__file__).stem + ".py'  v" + __version__)

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

from typing import List, Dict, Union, Iterable, Any
import os
import requests
import json
import openai



# Configure logging
logging.basicConfig(
    level=INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


# Set the level of the httpx logger to WARNING
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
#OPENROUTER_MODEL = "google/gemini-2.5-flash"    # 5/9
#OPENROUTER_MODEL = "deepseek/deepseek-chat-v3.1"    # 6/9
#OPENROUTER_MODEL = "openai/gpt-4o-mini"         # 6/9
#OPENROUTER_MODEL = "openai/gpt-4.1-mini"    # 6/9
#OPENROUTER_MODEL = "openai/gpt-oss-120b"        # 7/9
OPENROUTER_MODEL = "x-ai/grok-4-fast:free"      # 7/9  Fast and free with good high quality analysis of the data. 
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"


# Your OpenAI API key will be read from an environment variable.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4-turbo"
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
                    {"Q": "What is the name of the state where the children of Christopher Diaz attended public school?", "A": "Michigan (likely because that is the state he is employed)."},
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


async def show_all_graphiti_kuzu_relationships(graphiti, driver, verbose:bool=False):
    """
    Perform a Cypher query to see all the relationships that were actually created in the graph.
    """

    #################################################
    # Count All Nodes and Relationships
    #################################################
    # Execute a Cypher query to get a total count of all nodes and relationships in the database.
    # MATCH (n) RETURN count(n) AS total_nodes;
    # MATCH ()-[r]->() RETURN count(r) AS total_relationships;
    # MATCH (n:Episode) RETURN count(n) AS episode_count;
    # The generic Cypher query MATCH (n) RETURN count(n) is not the intended way to query a Graphiti graph

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

                #print(r)
                # {'n.name': 'Christopher Diaz', 'LABEL(r._ID,[,,,,RELATES_TO,RELATES_TO,,MENTIONS,,HAS_MEMBER,HAS_MEMBER])': 'RELATES_TO', 'm.name': 'BORN_IN'}
                #..

    # print output: 
    """
    All relationships in the graph:
    BORN_IN -> RELATES_TO -> Ryandbury, Rhode Island
    HAS_RACE -> RELATES_TO -> white man
    WORKED_AS -> RELATES_TO -> Willis Group LLC
    EMPLOYER_LOCATED_IN -> RELATES_TO -> Lewischester, Michigan
    MARRIED_TO -> RELATES_TO -> Cindy Lopez
    MARRIED_TO -> RELATES_TO -> Cindy Lopez
    DIED_AT_AGE -> RELATES_TO -> Cindy Lopez
    PARENT_OF -> RELATES_TO -> two sons
    PARENT_OF -> RELATES_TO -> two sons
    PARENT_OF -> RELATES_TO -> one daughter
    PARENT_OF -> RELATES_TO -> one daughter
    ATTENDED -> RELATES_TO -> Griffin public school
    COACHED_AT -> RELATES_TO -> Griffin public school
    biography 0 -> MENTIONS -> Christopher Diaz
    biography 1 -> MENTIONS -> Christopher Diaz
    biography 2 -> MENTIONS -> Christopher Diaz
    biography 3 -> MENTIONS -> Christopher Diaz
    biography 4 -> MENTIONS -> Christopher Diaz
    biography 5 -> MENTIONS -> Christopher Diaz
    biography 0 -> MENTIONS -> Ryandbury, Rhode Island
    biography 0 -> MENTIONS -> white man
    biography 1 -> MENTIONS -> Willis Group LLC
    biography 1 -> MENTIONS -> Lewischester, Michigan
    biography 2 -> MENTIONS -> Cindy Lopez
    biography 3 -> MENTIONS -> Cindy Lopez
    biography 5 -> MENTIONS -> Cindy Lopez
    biography 3 -> MENTIONS -> two sons
    biography 3 -> MENTIONS -> one daughter
    biography 4 -> MENTIONS -> Christopher Diaz's children
    biography 4 -> MENTIONS -> Griffin public school
    Christopher Diaz -> RELATES_TO -> BORN_IN
    Christopher Diaz -> RELATES_TO -> HAS_RACE
    Christopher Diaz -> RELATES_TO -> WORKED_AS
    Willis Group LLC -> RELATES_TO -> EMPLOYER_LOCATED_IN
    Christopher Diaz -> RELATES_TO -> MARRIED_TO
    Cindy Lopez -> RELATES_TO -> PARENT_OF
    Christopher Diaz -> RELATES_TO -> PARENT_OF
    Cindy Lopez -> RELATES_TO -> PARENT_OF
    Christopher Diaz -> RELATES_TO -> PARENT_OF
    Christopher Diaz's children -> RELATES_TO -> ATTENDED
    Christopher Diaz -> RELATES_TO -> COACHED_AT
    Christopher Diaz -> RELATES_TO -> MARRIED_TO
    Christopher Diaz -> RELATES_TO -> DIED_AT_AGE
    """


async def graphiti_kuzu_db_query_examples(graphiti, driver, query:str=None, ground_truth:str=None, verbose:bool=False):
    """
    Demonstrate various methods to query the Graphiti graph database built on Kuzu.

    1) Retrieve relationships (edges) from Graphiti using the .search() method.  This is a hybrid search that combines sematic similarity and BM25 text retrieval. 

    2) Center node search can return more contextually relevant results by reranking the search results based on their graph distance to a specific node.

    3) Node search using search recipes to perform a NODE_HYBRID_SEARCH_RRF and retrieve nodes directly instead of edges.

    """

    #################################################
    # BASIC SEARCH
    #################################################
    # The simplest way to retrieve relationships (edges)
    # from Graphiti is using the search method, which
    # performs a hybrid search combining semantic
    # similarity and BM25 text retrieval.
    #################################################

    # Perform a hybrid search combining semantic similarity and BM25 retrieval
    print(f"\nHybrid search for: '{query}'")
    results = await graphiti.search(query)

    print(f"Ground Truth: {ground_truth}")

    print('\nSearch Results:')
    for result in results:
        print(f'UUID: {result.uuid}')
        print(f'Fact: {result.fact}')
        if hasattr(result, 'valid_at') and result.valid_at:
            print(f'Valid from: {result.valid_at}')
        if hasattr(result, 'invalid_at') and result.invalid_at:
            print(f'Valid until: {result.invalid_at}')
        print('---')

    #################################################
    # CENTER NODE SEARCH
    #################################################
    # For more contextually relevant results, you can
    # use a center node to rerank search results based
    # on their graph distance to a specific node
    #################################################

    # Use the top search result's UUID as the center node for reranking
    if results and len(results) > 0:
        # Get the source node UUID from the top result
        center_node_uuid = results[0].source_node_uuid

        print('\nReranking search results based on graph distance:')
        print(f'Using center node UUID: {center_node_uuid}')

        reranked_results = await graphiti.search(query, center_node_uuid=center_node_uuid)

        # Print reranked search results
        print('\nReranked Search Results:')
        for result in reranked_results:
            print(f'UUID: {result.uuid}')
            print(f'Fact: {result.fact}')
            if hasattr(result, 'valid_at') and result.valid_at:
                print(f'Valid from: {result.valid_at}')
            if hasattr(result, 'invalid_at') and result.invalid_at:
                print(f'Valid until: {result.invalid_at}')
            print('---')
    else:
        print('No results found in the initial search to use as center node.')

    #################################################
    # NODE SEARCH USING SEARCH RECIPES
    #################################################
    # Graphiti provides predefined search recipes
    # optimized for different search scenarios.
    # Here we use NODE_HYBRID_SEARCH_RRF for retrieving
    # nodes directly instead of edges.
    #################################################

    print(f"\nPerforming node search using _search method with standard recipe NODE_HYBRID_SEARCH_RRF:")
    # Perform a hybrid search combining semantic similarity and BM25 retrieval
    print(f"\nSearch for: '{query}'")

    print(f"Ground Truth: {ground_truth}")

    # Use a predefined search configuration recipe and modify its limit
    node_search_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
    node_search_config.limit = 5  # Limit to 5 results

    # Execute the node search
    node_search_results = await graphiti._search(
        query=query,
        config=node_search_config,
    )

    # Print node search results
    print('\nNode Search Results:')
    for node in node_search_results.nodes:
        print(f'Node UUID: {node.uuid}')
        print(f'Node Name: {node.name}')
        node_summary = node.summary[:100] + '...' if len(node.summary) > 100 else node.summary
        print(f'Content Summary: {node_summary}')
        print(f'Node Labels: {", ".join(node.labels)}')
        print(f'Created At: {node.created_at}')
        if hasattr(node, 'attributes') and node.attributes:
            print('Attributes:')
            for key, value in node.attributes.items():
                print(f'  {key}: {value}')
        print('---')


    #################################################
    # Query using group_ids
    #################################################
    # The group_id field allows you go add a unique 
    # identifier to a group of related episodes. 
    #################################################

    group_id = "Biography_of_Christopher_Diaz"
    print(f"\nPerforming .search() using group_id = '{group_id}':")
    # Perform a hybrid search combining semantic similarity and BM25 retrieval
    print(f"\nSearch for: '{query}'")
    # This will only search for information within the 'product_catalog_2025' group
    results = await graphiti.search(
        query=query,
        group_ids=[group_id]
    )
    print('\nSearch Results:')
    for result in results:
        print(f'UUID: {result.uuid}')
        print(f'Fact: {result.fact}')
        if hasattr(result, 'valid_at') and result.valid_at:
            print(f'Valid from: {result.valid_at}')
        if hasattr(result, 'invalid_at') and result.invalid_at:
            print(f'Valid until: {result.invalid_at}')
        print('---')


async def get_answer_using_db_and_llm(graphiti, driver, query, verbose:bool=False):
    """
    Query the graph database and then pass the results to an LLM to answer a question. 
    """

    # Retrieve relationships (edges) from Graphiti using the .search() method.  This is a hybrid search that combines sematic similarity and BM25 text retrieval. 
    search_results = await graphiti.search(query)

    # Create the context for the LLM from the search results
    context = ""
    if search_results:
        context += "--- Context from Graph Database ---\n"
        for i, result in enumerate(search_results):
            context += f"Fact {i + 1}: {result.fact}\n"
            if hasattr(result, 'valid_at') and result.valid_at:
                context += f"Valid from: {result.valid_at}\n"
            if hasattr(result, 'invalid_at') and result.invalid_at:
                context += f"Valid until: {result.invalid_at}\n"
            context += "---\n"
    else:
        context = "No relevant information found in the graph database."
    #print(f"\ncontext:\n{context}\n")
    i_last = i + 1

    # Node search using search recipes to perform a NODE_HYBRID_SEARCH_RRF and retrieve nodes directly instead of edges.
    # Use a predefined search configuration recipe and modify its limit
    node_search_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
    node_search_config.limit = 5  # Limit to 5 results
    node_search_results = await graphiti._search(query=query, config=node_search_config)
    if node_search_results:
        #for node in node_search_results.nodes:
        for i,node in enumerate(node_search_results.nodes):
            context += f"Fact {i + 1 + i_last}: {node.summary}\n"
            context += "---\n"
    
    if verbose: print(f"\ncontext:\n{context}\n")

    # Create the prompt for the LLM
    prompt = f"""
    You are a concise information extractor.
    Using only the following context, answer the question.
    Provide only the answer, without any additional text, commentary, or explanation.
    If the answer is not in the context, respond with "I cannot answer based on the provided information."

    {context}

    Question: {query}
    Answer:
    """

    if verbose: print(f"\nPrompt:\n{prompt}\n")

    match LLM_SOURCE:
        case "OPENROUTER.AI":
            llm_response = call_llm_openrouter(OPENROUTER_API_KEY, prompt, OPENROUTER_MODEL)
        case "OPENAI":
            llm_response = call_llm_openai(OPENAI_API_KEY, prompt, OPENAI_MODEL)
        case "OLLAMA":
            llm_response = call_llm_ollama(OLLAMA_API_KEY, prompt, OLLAMA_MODEL)
        case _:
            raise Exception(f"Unknown LLM_SOURCE: {LLM_SOURCE}")

    if verbose: 
        print("\nFinal Answer from LLM:")
        print(llm_response)

    return llm_response


async def check_for_fts(kuzu_driver, verbose:bool=False):
    """
    Checks if the Kuzu graph database driver Full-Text Search (FTS) extension is already loaded in the Kuzu database.
    """
    try:
        # Execute the query to show loaded extensions
        query = "CALL SHOW_LOADED_EXTENSIONS() RETURN *;"
        results = await kuzu_driver.execute_query(query)

        #print(f"type(results): {type(results)}")
        # type(results): <class 'tuple'>
        #print(results)
        # ([{'extension name': 'FTS', 'extension source': 'OFFICIAL', 'extension path': 'C:\\Users\\Mark Kiehl/.kuzu/extension/0.11.2/win_amd64/fts/libfts.kuzu_extension'}], None, None)
        
        # The result is a tuple, with the first element being the list of results
        result_list = results[0]
        
        # Iterate through the list of dictionaries
        for row_dict in result_list:
            # Safely access the 'extension name' key
            if row_dict.get('extension name') == 'FTS':
                if verbose: print("FTS extension is already loaded.")
                return True
        
        if verbose: print("FTS extension is not loaded.")   
        return False
        
    except Exception as e:
        print(f"Error checking for FTS extension: {e}")
        return False


async def main():

    # Define a database file for the Kuzu graph database.
    path_file_db = Path(Path.cwd()).joinpath("db_graphiti_kuzu/christopher_diaz.kuzu")
    # Create the subfolders if they don't exist.
    if not path_file_db.parent.is_dir(): path_file_db.parent.mkdir(parents=True)
    
    # Make sure path_file_db exists
    if not path_file_db.is_file(): raise Exception(f"File not found {path_file_db}.  Run 'api_graphiti_kuzu_build.db' to rebuild the file.")

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
                api_key=OLLAMA_API_KEY,
                base_url=OLLAMA_BASE_URL,
                model=OLLAMA_MODEL,
                small_model=OLLAMA_MODEL
            )
        case _:
            raise Exception(f"Unknown LLM_SOURCE: {LLM_SOURCE}")

    # Create the LLM client instance
    llm_client = OpenAIGenericClient(config=llm_config)
    
    # Initialize Graphiti with the Kuzu driver and the custom LLM client.
    graphiti = Graphiti(graph_driver=driver, llm_client=llm_client)

    # NOTE: To use a different embedder (for vector embeddings) with Graphiti, you need to use the OpenAIEmbedder class and pass it a configuration that points to an OpenAI-compatible embeddings API endpoint,

    try: 

        # Check if the Kuzu graph database driver Full-Text Search (FTS) extension is already loaded in the Kuzu database.
        fts_is_loaded = await check_for_fts(driver, verbose=False)
        if not fts_is_loaded:
            # Load the Kuzu graph database driver Full-Text Search (FTS) extension.
            # The extension needs to be loaded before you can create FTS indexes or perform full-text searches on your graph data.
            # FTS (Full-Text Search) is a powerful feature that allows you to search for words or phrases within text-based properties of your data, such as node names or descriptions, rather than just matching exact values. 
            await driver.execute_query("LOAD EXTENSION FTS;")
            print("FTS extension loaded for the current session.")


        # Perform a Cypher query to see all the relationships that were actually created in the graph.
        #await show_all_graphiti_kuzu_relationships(graphiti, driver)


        # Query the database using various text search methods. 
        #   1) Hybrid search to retrieve relationships (edges)
        #   2) Center node search
        #   3) Node hybrid search to retrieve nodes directly.
        # The query below requires the retrieval of several related facts in order to provide the LLM the data needed to answer the question.
        query = "How many children did Christopher Diaz have?" 
        ground_truth = "Three, two sons and one daughter"
        await graphiti_kuzu_db_query_examples(graphiti, driver, query, ground_truth, verbose=True)


        # Query the graph database and then pass the results to an LLM to answer a question. (A single execution version of the code block that follows).  
        """
        query = "What year did Christopher Diaz die?"
        response = await get_answer_using_db_and_llm(graphiti, driver, query, verbose=False)
        print(f"\nquery: {query}")
        print(f"LLM response: {response}")
        """


        # Query the graph database & LLM with all questions. 
        print(f"\nQ&A using Graphiti graph framework with Kuzu graph database:")  
        docs, metadata, qna = docs_fake_christopher_diaz()
        for qa in qna:
            query = qa['Q']
            answer = await get_answer_using_db_and_llm(graphiti, driver, query, verbose=False)
            print(f"\nQuestion: {query}")
            print(f"Ground Truth: {qa['A']}")
            print(f"LLM response: {answer}")
        """
        """

    finally:

        # Close the database connection
        await graphiti.close()
        print("\nDatabase connection closed")


if __name__ == '__main__':
    pass

    asyncio.run(main())


