import os
import torch

model_name = os.environ["MODEL_NAME"] if "MODEL_NAME" in os.environ else "Co_NAML_LSTUR"
# Currently included model
assert model_name in [
    "Co_NAML_LSTUR",  # Added for the research paper model
]


class BaseConfig:
    """
    General configurations appiled to all models
    """

    num_epochs = 2
    num_batches_show_loss = 100  # Number of batchs to show loss
    # Number of batchs to check metrics on validation dataset
    num_batches_validate = 500
    batch_size = 128
    learning_rate = 0.0001
    num_workers = 12
    num_clicked_news_a_user = 60  # Number of sampled click history for each user
    num_words_title = 20
    num_words_abstract = 50
    word_freq_threshold = 1
    entity_freq_threshold = 2
    entity_confidence_threshold = 0.5
    negative_sampling_ratio = 3  # K
    dropout_probability = 0.3
    # Modify the following by the output of `src/dataprocess.py`
    num_words = 1 + 70972
    num_categories = 1 + 274
    num_entities = 1 + 12957
    num_users = 1 + 50000
    word_embedding_dim = 768  # Updated for DistilBERT
    category_embedding_dim = 100
    # Modify the following only if you use another dataset
    entity_embedding_dim = 100
    # For additive attention
    # query_vector_dim = 100
    # ---For Co_NAML_LSTUR---
    query_vector_dim = 200
    # ---For Co_NAML_LSTUR---


class Co_NAML_LSTURConfig(BaseConfig):
    """
    Configuration for Co-NAML-LSTUR: Combined model with NAML and LSTUR
    Official implementation matching the research paper results
    """
    dataset_attributes = {
        "news": ["category", "subcategory", "title", "abstract"],
        "record": ["user", "clicked_news_length"],
    }
    
    # Model architecture parameters (optimized for research results)
    num_filters = 300
    window_size = 4
    query_vector_dim = 200
    masking_probability = 0.5
    
    # Use DistilBERT for enhanced semantic understanding
    use_distilbert = True
    
    # Training parameters (46.4M parameters total)
    if torch.backends.mps.is_available():
        # Mac MPS optimized settings
        batch_size = 1
        num_workers = 1
        num_clicked_news_a_user = 30  # Reduced for MPS memory
    else:
        # Standard settings for CUDA/CPU
        batch_size = 64
        num_workers = 8
        num_clicked_news_a_user = 60
    
    # Learning parameters
    learning_rate = 0.0001
    num_epochs = 5
    
    # Early stopping and validation
    num_batches_validate = 500
    num_batches_show_loss = 100
