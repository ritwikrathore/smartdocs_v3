"""Model loading and management for the keyword_code package."""

from .databricks_embedding import load_databricks_embedding_model, DatabricksEmbeddingModel
from .embedding import load_embedding_model

# Import the new Databricks reranker module
try:
    from .databricks_reranker import load_databricks_reranker_model, DatabricksRerankerModel
except ImportError:
    # This allows the application to run even if the reranker module is not available
    load_databricks_reranker_model = None
    DatabricksRerankerModel = None

__all__ = [
    'load_databricks_embedding_model',
    'DatabricksEmbeddingModel',
    'load_embedding_model',
    'load_databricks_reranker_model',
    'DatabricksRerankerModel',
]
