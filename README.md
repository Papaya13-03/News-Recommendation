# News Recommendation System

This project implements a News Recommendation System using the **Co-NAML-LSTUR** model, which combines the strengths of NAML (Neural News Recommendation with Attentive Multi-View Learning) and LSTUR (Long- and Short-term User Representation) architectures. It leverages **DistilBERT** for enhanced semantic understanding of news titles and abstracts.

## Project Overview

The goal of this system is to recommend relevant news articles to users based on their reading history. It uses a hybrid approach to model both long-term and short-term user preferences, and employs attention mechanisms to learn informative representations of news articles from multiple views (category, subcategory, title, abstract).

### Key Model: NAMLxLSTUR

The core model, now located in `model.py`, features:

- **News Encoder (NAML)**: Utilizes a multi-view learning approach to create comprehensive news representations. It employs attention mechanisms to process title, abstract, category, and subcategory features, while integrating DistilBERT for enhanced semantic understanding.
- **User Encoder (LSTUR)**: Models user preferences by combining long-term representations (learned via user embeddings) with short-term interests derived from clicked news history, using LSTM-based sequential modeling.
- **Click Predictor**: A deep neural network that predicts the probability of a click by synthesizing the learned news and user representations.

## Project Structure

```
.
├── main.py                   # Single entry point for training, inference, and evaluation
├── model.py                  # Consolidated model architecture
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── checkpoint/               # Directory where model checkpoints are saved
├── data/                     # Directory for datasets (train, val, test)
└── runs/                     # TensorBoard logs
```

## Installation

1.  **Prerequisites**: Ensure you have Python 3.8+ installed.

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The project now uses a single entry point `main.py` for all operations.

### Training

To train the model:

```bash
python main.py train
```

**Arguments:**

- `--batch_size`: Override batch size.
- `--learning_rate`: Override learning rate.
- `--num_epochs`: Override number of epochs.
- `--device`: Specify device (e.g., "cuda:0", "cpu", "mps").

### Inference

To run inference (generate recommendations):

**1. Single User Prediction**

```bash
python main.py inference --user_id 1 \
  --clicked_news N37378 N14827 \
  --candidate_news N50398 N48265 \
  --top_k 5
```

**2. Batch Prediction**

```bash
python main.py inference --batch_file sample_batch_inference.json
```

**Arguments:**

- `--checkpoint_path`: Path to a specific checkpoint (optional, defaults to latest).
- `--news_data`: Path to parsed news data (default: "./data/test/news_parsed.tsv").
- `--batch_file`: Path to JSON file for batch processing.

## Model Details

The `model.py` file contains the complete implementation of the NAMLxLSTUR model architecture, including News Encoder, User Encoder, Attention mechanisms, and Click Predictor.

## Data Format

The system expects data in TSV format (parsed) in `data/` directory.

- `data/train/behaviors_parsed.tsv`
- `data/train/news_parsed.tsv`
- `data/test/news_parsed.tsv` (for inference)
