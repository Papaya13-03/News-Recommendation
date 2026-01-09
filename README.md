# News Recommendation System

This project implements a News Recommendation System using the **Co-NAML-LSTUR** model, which combines the strengths of NAML (Neural News Recommendation with Attentive Multi-View Learning) and LSTUR (Long- and Short-term User Representation) architectures. It leverages **DistilBERT** for enhanced semantic understanding of news titles and abstracts.

## Project Overview

The goal of this system is to recommend relevant news articles to users based on their reading history. It uses a hybrid approach to model both long-term and short-term user preferences, and employs attention mechanisms to learn informative representations of news articles from multiple views (category, subcategory, title, abstract).

### Key Model: Co_NAML_LSTUR

The core model, located in `model/Co_NAML_LSTUR`, features:

- **News Encoder**: Uses DistilBERT to encode news titles and abstracts, combined with category and subcategory embeddings.
- **User Encoder**: Captures user interests using:
  - **Long-term interest**: Learned from the entire history of clicked news.
  - **Short-term interest**: Learned from recent clicks using GRU (Gated Recurrent Unit).
- **Click Predictor**: Calculates the probability of a user clicking a candidate news article.

## Project Structure

```
.
├── config.py                 # Configuration parameters for training and model
├── dataset.py                # Dataset loading and processing logic
├── train.py                  # Script for training the model
├── inference.py              # Script for running inference (predictions)
├── evaluate.py               # Evaluation metrics (AUC, MRR, nDCG)
├── data_preprocess.py        # Data preprocessing utilities
├── model/                    # Model definitions
│   └── Co_NAML_LSTUR/        # The specific model implementation
├── checkpoint/               # Directory where model checkpoints are saved
├── data/                     # Directory for datasets (train, val, test)
├── runs/                     # TensorBoard logs
└── requirements.txt          # Python dependencies
```

## Installation

1.  **Prerequisites**: Ensure you have Python 3.8+ installed.

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

The model and training parameters are defined in `config.py`. Key parameters include:

- `batch_size`: Batch size for training.
- `learning_rate`: Learning rate for the optimizer.
- `num_epochs`: Number of training epochs.
- `dataset_attributes`: Fields used for news representation (category, title, etc.).

You can also override some configuration using environment variables or command-line arguments during training/inference.

## Usage

### Training

To train the model, run the `train.py` script:

```bash
python train.py
```

Arguments:

- `--model_name`: Name of the model to train (default: "NAMLxLSTUR").
- `--batch_size`: Override batch size.
- `--num_epochs`: Override number of epochs.
- `--device`: Specify device (e.g., "cuda:0", "cpu", "mps").

### Inference

You can run inference to get news recommendations for users using `inference.py`.

**1. Single User Prediction**

Predict probabilities for a specific list of candidate news for a user:

```bash
python inference.py --user_id 1 \
  --clicked_news N37378 N14827 \
  --candidate_news N50398 N48265 \
  --top_k 5
```

**2. Batch Prediction**

Run predictions for multiple users defined in a JSON file:

```bash
python inference.py --batch_file sample_batch_inference.json
```

**Arguments:**

- `--model_name`: Name of the model (default: "NAMLxLSTUR").
- `--checkpoint_path`: Path to a specific checkpoint (optional, defaults to latest).
- `--news_data`: Path to parsed news data (default: "./data/test/news_parsed.tsv").
- `--batch_file`: Path to JSON file for batch processing.

## Model Details

The `model` directory contains the implementation of the neural networks.

- `model/Co_NAML_LSTUR`: Contains the source code for the combined NAML and LSTUR model.
  - This directory typically includes the sub-modules for news encoding (text processing with DistilBERT) and user representation learning.

## Data Format

The system expects data in TSV format (parsed):

- **Behaviors**: `impression_id`, `user_id`, `time`, `history`, `impressions`
- **News**: `news_id`, `category`, `subcategory`, `title`, `abstract`, etc.

See `dataset.py` for details on how data is loaded.
