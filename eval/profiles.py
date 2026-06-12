"""
Eval Profiles Module.

Defines a set of predefined configuration profiles for running the RAG pipeline.
These profiles toggle various features (HyDE, Reranking, CRAG, Self-RAG) on or off
so that the evaluation harness can test the impact of each feature systematically.
"""
from __future__ import annotations

PROFILES: dict[str, dict] = {
    "naive":{
        "search_mode": "dense",
        "enable_hyde": False,
        "enable_rerank": False,
        "enable_crag": False,
        "enable_self_reflective": False,
        "top_k": 5,
    }, 
    "sparse_only": {
        "search_mode": "sparse",
        "enable_hyde": False,
        "enable_rerank": False,
        "enable_crag": False,
        "enable_self_reflective": False,
        "top_k": 5,
    },
    "hybrid": {
        "search_mode": "hybrid",
        "enable_hyde": False,
        "enable_rerank": False,
        "enable_crag": False,
        "enable_self_reflective": False,
        "top_k": 5,
    },
    "hybrid+rerank": {
        "search_mode": "hybrid",
        "enable_hyde": False,
        "enable_rerank": True,
        "enable_crag": False,
        "enable_self_reflective": False,
        "top_k": 5,
    },
    "hybrid+rerank+hyde": {
        "search_mode": "hybrid",
        "enable_hyde": True,
        "enable_rerank": True,
        "enable_crag": False,
        "enable_self_reflective": False,
        "top_k": 5,
    },
    "hybrid+rerank+crag": {
        "search_mode": "hybrid",
        "enable_hyde": False,
        "enable_rerank": True,
        "enable_crag": True,
        "enable_self_reflective": False,
        "top_k": 5,
    },
    "hybrid+crag": {
        "search_mode": "hybrid",
        "enable_hyde": False,
        "enable_rerank": False,
        "enable_crag": True,
        "enable_self_reflective": False,
        "top_k": 5,
    },
    "hybrid+rerank+self_reflective": {
        "search_mode": "hybrid",
        "enable_hyde": False,
        "enable_rerank": True,
        "enable_crag": False,
        "enable_self_reflective": True,
        "top_k": 5,
    },
    "all": {
        "search_mode": "hybrid",
        "enable_hyde": True,
        "enable_rerank": True,
        "enable_crag": True,
        "enable_self_reflective": True,
        "top_k": 5,
    },
}