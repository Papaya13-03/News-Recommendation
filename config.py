import torch

class BaseConfig:
    """
    General configurations appiled to all models
    """
    num_epochs = 2
    num_batches_show_loss = 100
    num_batches_validate = 500
    batch_size = 128
    learning_rate = 0.0001
    num_workers = 12
    num_clicked_news_a_user = 60
    num_words_title = 20
    num_words_abstract = 50
    word_freq_threshold = 1
    entity_freq_threshold = 2
    entity_confidence_threshold = 0.5
    negative_sampling_ratio = 3
    dropout_probability = 0.3
    num_words = 1 + 70972
    num_categories = 1 + 274
    num_entities = 1 + 12957
    num_users = 1 + 50000
    word_embedding_dim = 768
    category_embedding_dim = 100
    entity_embedding_dim = 100
    query_vector_dim = 200

class NAMLxLSTURConfig(BaseConfig):
    """
    Configuration for Co-NAML-LSTUR
    """
    dataset_attributes = {
        "news": ["category", "subcategory", "title", "abstract"],
        "record": ["user", "clicked_news_length"],
    }
    
    num_filters = 300
    window_size = 4
    query_vector_dim = 200
    masking_probability = 0.5
    use_distilbert = True
    
    if torch.backends.mps.is_available():
        batch_size = 1
        num_workers = 1
        num_clicked_news_a_user = 30
    else:
        batch_size = 64
        num_workers = 8
        num_clicked_news_a_user = 60
    
    learning_rate = 0.0001
    num_epochs = 5
    num_batches_validate = 500
    num_batches_show_loss = 100

# Global config instance
config = NAMLxLSTURConfig()
model_name = "NAMLxLSTUR"

# Device selection
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
