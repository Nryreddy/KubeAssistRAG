"""
RAGAS Evaluation Adapter.

This module provides an adapter to run RAGAS (Retrieval Augmented Generation Assessment)
evaluations on the RAG pipeline's outputs. It handles dataset formatting, model setup,
and execution of RAGAS metrics.
"""
from __future__ import annotations

import os

from datasets import Dataset
from langchain_openai import OpenAIEmbeddings

import sys, types

# Mock the vertexai module to prevent ImportError if google-cloud-aiplatform is not installed.
# This is often required by langchain_community dependencies even if we don't use VertexAI directly.
mock_vertexai = types.ModuleType("langchain_community.chat_models.vertexai")
mock_vertexai.ChatVertexAI = type("ChatVertexAI", (), {})
sys.modules["langchain_community.chat_models.vertexai"] = mock_vertexai

from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import llm_factory
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)


from app.config import settings


# Define the set of RAGAS metrics to evaluate the RAG pipeline.
# - faithfulness: Measures if the answer is factual given the retrieved context.
# - context_precision: Measures if the relevant context is ranked higher.
# - context_recall: Measures if all relevant context was retrieved.
# - answer_relevancy: Measures how relevant the generated answer is to the question.
METRICS = [
    faithfulness,
    context_precision,
    context_recall,
    answer_relevancy,
]



def _get_ragas_llm():
    """
    Create a Ragas-compatible LLM instance for evaluation grading.
    
    Returns:
        A Ragas LLM wrapper configured with the grader model from settings.
    """
    # Ensure OPENAI_API_KEY is in the environment for Langchain/Ragas internal usage
    os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)
    return llm_factory(settings.llm_model_grader)

def _get_ragas_embeddings():
    """
    Create a Ragas-compatible embeddings instance for evaluation.
    
    Returns:
        A LangchainEmbeddingsWrapper wrapping the OpenAIEmbeddings model defined in settings.
    """
    lc_emb = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )
    return LangchainEmbeddingsWrapper(lc_emb)

def build_dataset(rows: list[dict]) -> Dataset:
    """
    Convert a list of evaluation rows into a HuggingFace Dataset format expected by RAGAS.
    
    Args:
        rows: List of dictionaries containing 'question', 'answer', 'contexts', and 'ground_truth'.
        
    Returns:
        A datasets.Dataset object formatted for RAGAS evaluation.
    """
    return Dataset.from_dict(
        {
            "user_input": [r["question"] for r in rows],
            "response": [r["answer"] for r in rows],
            "retrieved_contexts": [r["contexts"] for r in rows],
            "reference": [r["ground_truth"] for r in rows],
        }
    )

def run(rows: list[dict]) -> list[dict]:
    """
    Run RAGAS evaluation metrics on a provided set of QA rows.
    
    Args:
        rows: A list of dictionaries representing the evaluation data. Each dict should have keys:
              'question', 'answer', 'contexts', and 'ground_truth'.
              
    Returns:
        A list of dictionaries containing the original rows appended with the evaluation metric scores.
    """
    if not rows:
        return []

    # Build the RAGAS dataset
    ds = build_dataset(rows)
    
    # Execute the evaluation using configured LLMs and embeddings
    result = evaluate(
        ds,
        metrics=METRICS,
        llm=_get_ragas_llm(),
        embeddings=_get_ragas_embeddings(),
        show_progress=False,
    )
    
    # Convert the results back to a list of dicts for easier consumption
    return result.to_pandas().to_dict(orient="records")