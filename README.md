# medium_graphiti_kuzu
Graphiti graph framework with Kazu graph database

See the complete public article at:

https://medium.com/@markwkiehl/a-free-graph-rag-system-that-works-with-real-data-67363faa340f


Create a Python virtual environment, activate it, and then extract the contents of the .ZIP file to the virtual environment folder. 

Each script has the list of PIP INSTALLs that you need to execute after creating a Python environment:

pip install graphiti-core
pip install graphiti-core[kuzu]
pip install dotenv
pip install openai
pip install pydantic
pip install requests
pip install openai

In addition to the Python virtual environment (venv) and the PIP INSTALLs above, you will need to configure a .env file with your API key for either OpenRouter.ai (recommended), or OpenAI. An empty .env file is provided for you to edit. 

In your Python IDE, run the script "api_graphiti_kuzu_build_db.py" first to build the graph database. Then run the script "api_graphiti_kuzu_query_db.py" to execute queries against it.

In the scripts provided are several solutions to problems you will encounter if you try to simply use the examples provided by Graphiti. Study the scripts and use them as a template for your own investigation.

Fix The Graphiti Timezone Bug
The file edge_operations.py in Graphiti-core Version 0.20.4 is not configured to handle a timezone-naive or timezone-aware datetime, and as a result will raise errors when adding episodes to the graph database. The fix shown below was reported by me to Graphiti on 21 September 2025:

'''

# Lib\site-packages\graphiti_core\utils\maintenance\edge_operations.py

# Add to the top of the script edge_operations.py
from pytz import UTC

# Find the function resolve_edge_contradictions() and replace it with the following:

def resolve_edge_contradictions(
    resolved_edge: EntityEdge, invalidation_candidates: list[EntityEdge]
) -> list[EntityEdge]:
    if len(invalidation_candidates) == 0:
        return []

    # Ensure resolved_edge.valid_at is timezone-aware
    resolved_valid_at = ensure_utc(resolved_edge.valid_at) if resolved_edge.valid_at else None

    # Determine which contradictory edges need to be expired
    invalidated_edges: list[EntityEdge] = []
    for edge in invalidation_candidates:
        # Ensure edge.valid_at and edge.invalid_at are timezone-aware
        edge_valid_at = ensure_utc(edge.valid_at) if edge.valid_at else None
        edge_invalid_at = ensure_utc(edge.invalid_at) if edge.invalid_at else None

        # (Edge invalid before new edge becomes valid) or (new edge invalid before edge becomes valid)
        if (
            edge_invalid_at is not None
            and resolved_valid_at is not None
            and edge_invalid_at <= resolved_valid_at
        ) or (
            edge_valid_at is not None
            and resolved_edge.invalid_at is not None
            and resolved_edge.invalid_at <= edge_valid_at
        ):
            continue
        # New edge invalidates edge
        elif (
            edge_valid_at is not None
            and resolved_valid_at is not None
            and edge_valid_at < resolved_valid_at
        ):
            # Only update invalid_at if the new edge's valid_at is more recent
            # This check is now redundant due to the previous elif condition but kept for clarity
            if edge_valid_at < resolved_valid_at:
                edge.invalid_at = resolved_valid_at
                edge.expired_at = edge.expired_at if edge.expired_at is not None else utc_now()
                invalidated_edges.append(edge)
    
    return invalidated_edges
'''
