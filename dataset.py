import torch
import pandas as pd
from torch.utils.data import Dataset
from ast import literal_eval
import os

class BaseDataset(Dataset):
    def __init__(self, behaviors_path, news_path, config):
        super(BaseDataset, self).__init__()
        self.config = config
        self.behaviors_parsed = pd.read_table(behaviors_path)
        self.news_parsed = pd.read_table(
            news_path,
            index_col='id',
            usecols=['id'] + config.dataset_attributes['news'],
            converters={
                attribute: literal_eval
                for attribute in set(config.dataset_attributes['news']) & set([
                    'title', 'abstract', 'title_entities', 'abstract_entities'
                ])
            })
        self.news2dict = self.news_parsed.to_dict('index')
        for key1 in self.news2dict.keys():
            for key2 in self.news2dict[key1].keys():
                self.news2dict[key1][key2] = torch.tensor(
                    self.news2dict[key1][key2])
        padding_all = {
            'category': 0,
            'subcategory': 0,
            'title': [0] * config.num_words_title,
            'abstract': [0] * config.num_words_abstract,
            'title_entities': [0] * config.num_words_title,
            'abstract_entities': [0] * config.num_words_abstract
        }
        for key in padding_all.keys():
            padding_all[key] = torch.tensor(padding_all[key])

        self.padding = {
            k: v
            for k, v in padding_all.items()
            if k in config.dataset_attributes['news']
        }

    def __len__(self):
        return len(self.behaviors_parsed)

    def __getitem__(self, idx):
        item = {}
        row = self.behaviors_parsed.iloc[idx]
        if 'user' in self.config.dataset_attributes['record']:
            item['user'] = row.user
        item["clicked"] = list(map(int, row.clicked.split()))
        item["candidate_news"] = [
            self.news2dict[x] for x in row.candidate_news.split()
        ]
        item["clicked_news"] = [
            self.news2dict[x]
            for x in row.clicked_news.split()[:self.config.num_clicked_news_a_user]
        ]
        if 'clicked_news_length' in self.config.dataset_attributes['record']:
            item['clicked_news_length'] = len(item["clicked_news"])
        repeated_times = self.config.num_clicked_news_a_user - \
            len(item["clicked_news"])
        assert repeated_times >= 0
        item["clicked_news"] = [self.padding
                                ] * repeated_times + item["clicked_news"]

        return item

class NewsDataset(Dataset):
    """Load news for evaluation."""
    def __init__(self, news_path, config):
        super(NewsDataset, self).__init__()
        self.config = config
        self.news_parsed = pd.read_table(
            news_path,
            usecols=["id"] + config.dataset_attributes["news"],
            converters={
                attribute: literal_eval
                for attribute in set(config.dataset_attributes["news"])
                & set(["title", "abstract", "title_entities", "abstract_entities"])
            },
        )
        self.news2dict = self.news_parsed.to_dict("index")
        for key1 in self.news2dict.keys():
            for key2 in self.news2dict[key1].keys():
                if type(self.news2dict[key1][key2]) != str:
                    self.news2dict[key1][key2] = torch.tensor(
                        self.news2dict[key1][key2]
                    )

    def __len__(self):
        return len(self.news_parsed)

    def __getitem__(self, idx):
        return self.news2dict[idx]

class UserDataset(Dataset):
    """Load users for evaluation."""
    def __init__(self, behaviors_path, user2int_path, config):
        super(UserDataset, self).__init__()
        self.config = config
        self.behaviors = pd.read_table(
            behaviors_path, header=None, usecols=[1, 3], names=["user", "clicked_news"]
        )
        self.behaviors["clicked_news"] = self.behaviors["clicked_news"].fillna(" ")
        self.behaviors.drop_duplicates(inplace=True)
        user2int = dict(pd.read_table(user2int_path).values.tolist())
        user_total = 0
        user_missed = 0
        for row in self.behaviors.itertuples():
            user_total += 1
            if row.user in user2int:
                self.behaviors.at[row.Index, "user"] = user2int[row.user]
            else:
                user_missed += 1
                self.behaviors.at[row.Index, "user"] = 0
        print(f"User miss rate: {user_missed / user_total:.4f}")

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        row = self.behaviors.iloc[idx]
        item = {
            "user": row.user,
            "clicked_news_string": row.clicked_news,
            "clicked_news": row.clicked_news.split()[: self.config.num_clicked_news_a_user],
        }
        item["clicked_news_length"] = len(item["clicked_news"])
        repeated_times = self.config.num_clicked_news_a_user - len(item["clicked_news"])
        assert repeated_times >= 0
        item["clicked_news"] = ["PADDED_NEWS"] * repeated_times + item["clicked_news"]
        return item

class BehaviorsDataset(Dataset):
    """Load behaviors for evaluation."""
    def __init__(self, behaviors_path):
        super(BehaviorsDataset, self).__init__()
        self.behaviors = pd.read_table(
            behaviors_path,
            header=None,
            usecols=range(5),
            names=["impression_id", "user", "time", "clicked_news", "impressions"],
        )
        self.behaviors["clicked_news"] = self.behaviors["clicked_news"].fillna(" ")
        self.behaviors.impressions = self.behaviors.impressions.str.split()

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        row = self.behaviors.iloc[idx]
        return {
            "impression_id": row.impression_id,
            "user": row.user,
            "time": row.time,
            "clicked_news_string": row.clicked_news,
            "impressions": row.impressions,
        }

class NewsInferenceDataset(Dataset):
    """Dataset for inference."""
    def __init__(self, news_path, config):
        super(NewsInferenceDataset, self).__init__()
        self.config = config
        self.news_parsed = pd.read_table(
            news_path,
            usecols=["id"] + config.dataset_attributes["news"],
            converters={
                attribute: literal_eval
                for attribute in set(config.dataset_attributes["news"])
                & set(["title", "abstract", "title_entities", "abstract_entities"])
            },
        )
        self.news2dict = self.news_parsed.to_dict("index")
        for key1 in self.news2dict.keys():
            for key2 in self.news2dict[key1].keys():
                if type(self.news2dict[key1][key2]) != str:
                    self.news2dict[key1][key2] = torch.tensor(
                        self.news2dict[key1][key2]
                    )

    def __len__(self):
        return len(self.news_parsed)

    def __getitem__(self, idx):
        return self.news2dict[idx]
