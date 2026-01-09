import argparse
import os
import time
import datetime
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from multiprocessing import Pool
import sys

# Import from new modules
from config import config, model_name, device, NAMLxLSTURConfig
from dataset import BaseDataset, NewsDataset, UserDataset, BehaviorsDataset, NewsInferenceDataset
from utils import time_since, latest_checkpoint, EarlyStopping, calculate_single_user_metric
from model import NAMLxLSTUR

# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, directory, num_workers, max_count=sys.maxsize):
    news_dataset = NewsDataset(os.path.join(directory, "news_parsed.tsv"), config)
    news_dataloader = DataLoader(
        news_dataset,
        batch_size=config.batch_size * 16,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False,
        pin_memory=True,
    )

    news2vector = {}
    for minibatch in tqdm(news_dataloader, desc="Calculating vectors for news"):
        news_ids = minibatch["id"]
        # Filter out existing (optimization)
        unique_news_ids = []
        indices = []
        for i, id in enumerate(news_ids):
            if id not in news2vector:
                unique_news_ids.append(id)
                indices.append(i)
        
        if unique_news_ids:
            # Reconstruct batch for unique items only if needed or just process all
            # Processing all is simpler for batching
            news_vector = model.get_news_vector(minibatch)
            for id, vector in zip(news_ids, news_vector):
                if id not in news2vector:
                    news2vector[id] = vector

    news2vector["PADDED_NEWS"] = torch.zeros(list(news2vector.values())[0].size())

    user_dataset = UserDataset(
        os.path.join(directory, "behaviors.tsv"), "data/train/user2int.tsv", config
    )
    user_dataloader = DataLoader(
        user_dataset,
        batch_size=config.batch_size * 16,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False,
        pin_memory=True,
    )

    user2vector = {}
    for minibatch in tqdm(user_dataloader, desc="Calculating vectors for users"):
        user_strings = minibatch["clicked_news_string"]
        if any(user_string not in user2vector for user_string in user_strings):
            clicked_news_vector = torch.stack(
                [
                    torch.stack([news2vector[x].to(device) for x in news_list], dim=0)
                    for news_list in minibatch["clicked_news"]
                ],
                dim=0,
            ).transpose(0, 1)
            
            user_vector = model.get_user_vector(
                minibatch["user"],
                minibatch["clicked_news_length"],
                clicked_news_vector,
            )
            
            for user, vector in zip(user_strings, user_vector):
                if user not in user2vector:
                    user2vector[user] = vector

    behaviors_dataset = BehaviorsDataset(os.path.join(directory, "behaviors.tsv"))
    behaviors_dataloader = DataLoader(
        behaviors_dataset, batch_size=1, shuffle=False, num_workers=config.num_workers
    )

    count = 0
    tasks = []

    for minibatch in tqdm(behaviors_dataloader, desc="Calculating probabilities"):
        count += 1
        if count == max_count:
            break

        candidate_news_vector = torch.stack(
            [news2vector[news[0].split("-")[0]] for news in minibatch["impressions"]],
            dim=0,
        )
        user_vector = user2vector[minibatch["clicked_news_string"][0]]
        click_probability = model.get_prediction(candidate_news_vector, user_vector)

        y_pred = click_probability.tolist()
        y_true = [int(news[0].split("-")[1]) for news in minibatch["impressions"]]
        tasks.append((y_true, y_pred))

    with Pool(processes=num_workers) as pool:
        results = pool.map(calculate_single_user_metric, tasks)

    aucs, mrrs, ndcg5s, ndcg10s = np.array(results).T
    return np.nanmean(aucs), np.nanmean(mrrs), np.nanmean(ndcg5s), np.nanmean(ndcg10s)

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def train(args):
    # Override config with args
    if args.batch_size: config.batch_size = args.batch_size
    if args.learning_rate: config.learning_rate = args.learning_rate
    if args.num_epochs: config.num_epochs = args.num_epochs

    print("Using device:", device)
    print(f"Training model {model_name}")

    writer = SummaryWriter(
        log_dir=f"./runs/{model_name}/{datetime.datetime.now().replace(microsecond=0).isoformat()}{'-' + os.environ['REMARK'] if 'REMARK' in os.environ else ''}"
    )

    if not os.path.exists("checkpoint"):
        os.makedirs("checkpoint")

    try:
        pretrained_word_embedding = torch.from_numpy(
            np.load("./data/train/pretrained_word_embedding.npy")
        ).float()
    except FileNotFoundError:
        pretrained_word_embedding = None

    model = NAMLxLSTUR(config, pretrained_word_embedding).to(device)
    print(model)

    dataset_train = BaseDataset("data/train/behaviors_parsed.tsv", "data/train/news_parsed.tsv", config)
    print(f"Load training dataset with size {len(dataset_train)}.")

    dataloader = iter(
        DataLoader(
            dataset_train,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            drop_last=True,
            pin_memory=True,
        )
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    start_time = time.time()
    loss_full = []
    exhaustion_count = 0
    step = 0
    early_stopping = EarlyStopping(patience=args.patience)

    checkpoint_dir = os.path.join("./checkpoint", model_name)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    checkpoint_path = latest_checkpoint(checkpoint_dir)
    if checkpoint_path is not None:
        print(f"Load saved parameters in {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        early_stopping(checkpoint["early_stop_value"])
        step = checkpoint["step"]
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        model.train()

    for i in tqdm(
        range(1, config.num_epochs * len(dataset_train) // config.batch_size + 1),
        desc="Training",
    ):
        try:
            minibatch = next(dataloader)
        except StopIteration:
            exhaustion_count += 1
            tqdm.write(
                f"Training data exhausted for {exhaustion_count} times after {i} batches, reuse the dataset."
            )
            dataloader = iter(
                DataLoader(
                    dataset_train,
                    batch_size=config.batch_size,
                    shuffle=True,
                    num_workers=config.num_workers,
                    drop_last=True,
                    pin_memory=True,
                )
            )
            minibatch = next(dataloader)

        step += 1
        y_pred = model(
            minibatch["user"],
            minibatch["clicked_news_length"],
            minibatch["candidate_news"],
            minibatch["clicked_news"],
        )

        y = torch.zeros(len(y_pred)).long().to(device)
        loss = criterion(y_pred, y)

        loss_full.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            writer.add_scalar("Train/Loss", loss.item(), step)

        if i % config.num_batches_show_loss == 0:
            tqdm.write(
                f"Time {time_since(start_time)}, batches {i}, current loss {loss.item():.4f}, average loss: {np.mean(loss_full):.4f}"
            )

        if i % config.num_batches_validate == 0:
            model.eval()
            val_auc, val_mrr, val_ndcg5, val_ndcg10 = evaluate(
                model, "./data/val", config.num_workers, 200000
            )
            model.train()
            writer.add_scalar("Validation/AUC", val_auc, step)
            writer.add_scalar("Validation/MRR", val_mrr, step)
            writer.add_scalar("Validation/nDCG@5", val_ndcg5, step)
            writer.add_scalar("Validation/nDCG@10", val_ndcg10, step)
            tqdm.write(
                f"Time {time_since(start_time)}, batches {i}, val AUC: {val_auc:.4f}, val MRR: {val_mrr:.4f}, val nDCG@5: {val_ndcg5:.4f}, val nDCG@10: {val_ndcg10:.4f}"
            )

            early_stop, get_better = early_stopping(-val_auc)
            if early_stop:
                tqdm.write("Early stop.")
                break
            elif get_better:
                try:
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "step": step,
                            "early_stop_value": -val_auc,
                        },
                        f"./checkpoint/{model_name}/ckpt-{step}.pth",
                    )
                except OSError as error:
                    print(f"OS error: {error}")

# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------

class NewsRecommendationInference:
    def __init__(self, model_name="NAMLxLSTUR", checkpoint_path=None):
        self.model_name = model_name
        self.device = device
        
        self.model = NAMLxLSTUR(config).to(self.device)
        self._load_checkpoint(checkpoint_path)
        
        self.news_dataset = None
        self.news2vector = {}
        print(f"Inference system initialized using device: {self.device}")
    
    def _load_checkpoint(self, checkpoint_path):
        if checkpoint_path is None:
            checkpoint_dir = os.path.join("./checkpoint", self.model_name)
            if not os.path.exists(checkpoint_dir):
                raise FileNotFoundError(f"No checkpoint directory found: {checkpoint_dir}")
            
            all_checkpoints = {
                int(x.split(".")[-2].split("-")[-1]): x 
                for x in os.listdir(checkpoint_dir) if x.endswith('.pth')
            }
            if not all_checkpoints:
                raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
            
            checkpoint_path = os.path.join(checkpoint_dir, all_checkpoints[max(all_checkpoints.keys())])
        
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        print("Model loaded successfully!")
    
    def load_news_data(self, news_path):
        print("Loading news data...")
        self.news_dataset = NewsInferenceDataset(news_path, config)
        print(f"Loaded {len(self.news_dataset)} news articles")
    
    def precompute_news_vectors(self, batch_size=None):
        if self.news_dataset is None:
            raise ValueError("News data not loaded.")
        
        if batch_size is None:
            batch_size = max(1, config.batch_size)
        
        print("Precomputing news vectors...")
        news_dataloader = DataLoader(
            self.news_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
            drop_last=False,
            pin_memory=False,
        )
        
        self.news2vector = {}
        with torch.no_grad():
            for minibatch in tqdm(news_dataloader, desc="Computing news vectors"):
                news_ids = minibatch["id"]
                model_input = {k: v for k, v in minibatch.items() if k != "id"}
                news_vector = self.model.get_news_vector(model_input)
                for news_id, vector in zip(news_ids, news_vector):
                    if news_id not in self.news2vector:
                        self.news2vector[news_id] = vector.cpu()
        
        if self.news2vector:
            sample_vector = list(self.news2vector.values())[0]
            self.news2vector["PADDED_NEWS"] = torch.zeros_like(sample_vector)
        
        print(f"Precomputed vectors for {len(self.news2vector)} news articles")
    
    def get_user_vector(self, user_id, clicked_news_ids):
        if not self.news2vector:
            raise ValueError("News vectors not computed.")
        
        clicked_news_ids = clicked_news_ids[-config.num_clicked_news_a_user:]
        clicked_news_length = len(clicked_news_ids)
        
        padding_needed = config.num_clicked_news_a_user - len(clicked_news_ids)
        clicked_news_ids = ["PADDED_NEWS"] * padding_needed + clicked_news_ids
        
        clicked_news_vectors = []
        for news_id in clicked_news_ids:
            if news_id in self.news2vector:
                clicked_news_vectors.append(self.news2vector[news_id].to(self.device))
            else:
                clicked_news_vectors.append(self.news2vector["PADDED_NEWS"].to(self.device))
        
        clicked_news_vector = torch.stack(clicked_news_vectors, dim=0).unsqueeze(0)
        user_tensor = torch.tensor([user_id], dtype=torch.long)
        clicked_news_length_tensor = torch.tensor([clicked_news_length], dtype=torch.long)
        
        with torch.no_grad():
            user_vector = self.model.get_user_vector(
                user_tensor, 
                clicked_news_length_tensor, 
                clicked_news_vector
            )
        
        return user_vector.squeeze(0)
    
    def predict(self, user_id, clicked_news_ids, candidate_news_ids):
        user_vector = self.get_user_vector(user_id, clicked_news_ids)
        candidate_vectors = []
        for news_id in candidate_news_ids:
            if news_id in self.news2vector:
                candidate_vectors.append(self.news2vector[news_id].to(self.device))
            else:
                candidate_vectors.append(self.news2vector["PADDED_NEWS"].to(self.device))
        
        candidate_news_vector = torch.stack(candidate_vectors, dim=0)
        
        with torch.no_grad():
            click_probabilities = self.model.get_prediction(candidate_news_vector, user_vector)
        
        probs = click_probabilities.cpu().numpy()
        results = list(zip(candidate_news_ids, probs))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def recommend_top_k(self, user_id, clicked_news_ids, candidate_news_ids, k=10):
        predictions = self.predict(user_id, clicked_news_ids, candidate_news_ids)
        return predictions[:k]

    def batch_predict(self, user_data_list, candidate_news_ids):
        results = {}
        for user_data in tqdm(user_data_list, desc="Processing users"):
            user_id = user_data['user_id']
            clicked_news_ids = user_data['clicked_news_ids']
            try:
                predictions = self.predict(user_id, clicked_news_ids, candidate_news_ids)
                results[user_id] = predictions
            except Exception as e:
                print(f"Error processing user {user_id}: {e}")
                results[user_id] = []
        return results


def run_inference(args):
    inferencer = NewsRecommendationInference(
        model_name="NAMLxLSTUR",
        checkpoint_path=args.checkpoint_path
    )
    inferencer.load_news_data(args.news_data)
    inferencer.precompute_news_vectors()
    
    if args.batch_file:
        print(f"Loading batch data from {args.batch_file}")
        with open(args.batch_file, 'r') as f:
            batch_data = json.load(f)
        results = inferencer.batch_predict(batch_data['users'], batch_data['candidate_news'])
        output_file = args.batch_file.replace('.json', '_results.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Batch results saved to {output_file}")
    else:
        print(f"\nMaking prediction for user {args.user_id}")
        # clicked_news/candidate_news are strings in args, but predict expects lists
        # handled by helper or standard parse
        clicked = args.clicked_news
        candidate = args.candidate_news
        
        if not clicked: clicked = ["N37378", "N14827", "N50398"]
        if not candidate: candidate = ["N37378", "N14827", "N50398", "N48265", "N42793"]

        try:
            recommendations = inferencer.recommend_top_k(
                user_id=args.user_id,
                clicked_news_ids=clicked,
                candidate_news_ids=candidate,
                k=args.top_k
            )
            print(f"\nTop {args.top_k} recommendations:")
            for i, (news_id, prob) in enumerate(recommendations, 1):
                print(f"{i}. News {news_id}: {prob:.4f}")
        except Exception as e:
            print(f"Error during prediction: {e}")

# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="News Recommendation System")
    subparsers = parser.add_subparsers(dest="mode", help="Mode of operation")

    # Train parser
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--batch_size", type=int, default=None)
    train_parser.add_argument("--learning_rate", type=float, default=None)
    train_parser.add_argument("--num_epochs", type=int, default=None)
    train_parser.add_argument("--patience", type=int, default=5)
    train_parser.add_argument("--device", type=str, default=None)

    # Inference parser
    inf_parser = subparsers.add_parser("inference", help="Make predictions")
    inf_parser.add_argument("--checkpoint_path", type=str, default=None)
    inf_parser.add_argument("--news_data", type=str, default="./data/test/news_parsed.tsv")
    inf_parser.add_argument("--user_id", type=int, default=1)
    inf_parser.add_argument("--clicked_news", type=str, nargs="+")
    inf_parser.add_argument("--candidate_news", type=str, nargs="+")
    inf_parser.add_argument("--top_k", type=int, default=5)
    inf_parser.add_argument("--batch_file", type=str, default=None)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "inference":
        run_inference(args)
    else:
        print("Please specify a mode: 'train' or 'inference'")
