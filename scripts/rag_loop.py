import os
import sys
import logging
import traceback
import openai
import asyncio
from qdrant_client import QdrantClient
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv

os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()
# Setup logging
def setup_logging(task_name, log_dir="LOG"):
    os.makedirs(log_dir, exist_ok=True)
    current_time = datetime.now().strftime("%Y-%m-%d")
    log_file_path = os.path.join(log_dir, f"{task_name}_{current_time}.log")

    logger = logging.getLogger(task_name)
    logger.setLevel(logging.DEBUG)

    # Create handlers
    console_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(log_file_path)

    # Create formatters and add them to handlers
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

logger = setup_logging("hr_workflow")

# API Key setup
openai.api_key = os.getenv("OPENAI_API_KEY")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")

if not openai.api_key or not qdrant_api_key or not qdrant_url:
    logger.error("API keys for OpenAI and Qdrant, as well as Qdrant URL, must be set in environment variables.")
    raise EnvironmentError("API keys for OpenAI and Qdrant, and Qdrant URL must be set in environment variables.")

# Initialize Qdrant client
qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

def get_bert_embedding(text: str) -> List[float]:
    """Generate an embedding for a given text using BERT."""
    logger.debug(f"Generating embedding for text: {text[:130]}...")
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
        logger.debug(f"Generated embedding of length {len(embedding)}")
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding for text: {e}")
        return []

async def get_large_embeddings(text: str) -> List[float]:
    logger.debug(f"Generating large embedding for: {text[:150]}...")
    cleaned_text = text.replace("\t", "").replace("\n", "")
    try:
        response = openai.embeddings.create(input=cleaned_text, model="text-embedding-3-large")
        embedding = response.data[0].embedding  # Correctly access the embedding data
        return embedding
    except Exception as e:
        logger.error(f"Error generating large embedding: {e}")
        return []

def cosine_sim(embedding1: List[float], embedding2: List[float]) -> float:
    """Calculate cosine similarity between two embeddings."""
    return cosine_similarity([embedding1], [embedding2])[0][0]

def filter_similar_context_nodes(context_nodes: List[Dict[str, Any]], threshold: float = 0.80) -> List[Dict[str, Any]]:
    """Filter out context nodes that are too similar to each other."""
    filtered_nodes = []
    embeddings = [get_bert_embedding(node["text"]) for node in context_nodes]

    for i, node in enumerate(context_nodes):
        is_similar = False
        for j in range(i):
            similarity = cosine_sim(embeddings[i], embeddings[j])
            if similarity >= threshold:
                is_similar = True
                break

        if not is_similar:
            filtered_nodes.append(node)

    return filtered_nodes

async def query_qdrant_for_similar_embeddings(collection_name: str, embedding: List[float], top_k=2) -> List[Dict[str, Any]]:
    if not embedding:
        logger.error("No valid embedding to search Qdrant with.")
        return []

    try:
        search_result = qdrant_client.search(
            collection_name=collection_name,
            query_vector=embedding,
            limit=top_k
        )
        logger.debug(f"Qdrant search successful. Results: {len(search_result)}")
        return search_result
    except Exception as e:
        logger.error(f"Error during Qdrant search: {e}")
        return []

async def get_embeddings_and_search_qdrant(embedding_model: str, content_item: str, collection_name: str) -> Dict[str, Any]:
    """
    Perform RAG pipeline by generating embeddings for the content and querying Qdrant.
    """
    try:
        if embedding_model == "bert":
            logger.info("Using BERT-base-uncased for embedding model.")
            embedding = get_bert_embedding(content_item)
        else:
            embedding = await get_large_embeddings(content_item)

        # Perform Qdrant search using the embeddings
        search_results = await query_qdrant_for_similar_embeddings(collection_name, embedding, top_k=2)

        # Extract relevant data from the search results
        context_nodes = [
            {
                "id": result.id,
                "score": result.score,
                "text": result.payload.get("content", ""),
                "source": result.payload.get("section_name", ""),
            }
            for result in search_results
            if hasattr(result, 'id') and hasattr(result, 'score') and hasattr(result, 'payload')
        ]

        # Combine search results texts
        combined_search_results_text = " ".join(node["text"] for node in context_nodes)
        logger.debug(f"Combined content for GPT input: {combined_search_results_text[:500]}")

        # Construct the final response
        return {
            "search_results": combined_search_results_text,
            "context_nodes": context_nodes
        }
    except Exception as e:
        logger.error(f"Error in get_embeddings_and_search_qdrant: {e}")
        traceback.print_exc()
        return {"search_results": "", "context_nodes": []}

#

async def get_context(embedding_model: str, content_item: str, collection_name: str, filter_values: list = None, filter_keys: list = None) -> Dict[str, Any]:
    try:
        # Perform the embedding and Qdrant search
        result_data = await get_embeddings_and_search_qdrant(embedding_model, content_item, collection_name)

        # Extract context nodes
        context_nodes = result_data.get("context_nodes", [])
        logger.debug(f"Total context nodes retrieved: {len(context_nodes)}")

        # Check if filtering is required
        if filter_values and filter_keys:
            filtered_results = [
                node for node in context_nodes
                if any(value in node.get(key, '') for key in filter_keys for value in filter_values)
            ]
        else:
            logger.info("No filtering applied, returning all context nodes.")
            filtered_results = context_nodes

        if not filtered_results:
            logger.error("No context found for script generation.")
            return {"search_results": "", "context_nodes": []}

        # Combine search results texts
        combined_search_results_text = " ".join(
            [f"{result.get('text', '')} {result.get('summary', '')}".strip() for result in filtered_results]
        )
        logger.info("Extracted and combined texts and summaries.")

        return {
            "search_results": combined_search_results_text,
            "context_nodes": filtered_results
        }
    except Exception as e:
        logger.error(f"Error extracting and processing texts: {e}")
        traceback.print_exc()
        return {"search_results": "", "context_nodes": []}


__all__ = ["get_embeddings_and_search_qdrant", "get_context"]
