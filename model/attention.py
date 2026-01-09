import torch
import torch.nn as nn
import torch.nn.functional as F

class AdditiveAttention(torch.nn.Module):
    """
    A general additive attention module.
    Originally for NAML.
    """
    def __init__(self,
                 query_vector_dim,
                 candidate_vector_dim,
                 writer=None,
                 tag=None,
                 names=None):
        super(AdditiveAttention, self).__init__()
        self.linear = nn.Linear(candidate_vector_dim, query_vector_dim)
        self.attention_query_vector = nn.Parameter(
            torch.empty(query_vector_dim).uniform_(-0.1, 0.1))
        # For tensorboard
        self.writer = writer
        self.tag = tag
        self.names = names
        self.local_step = 1

    def forward(self, candidate_vector):
        """
        Args:
            candidate_vector: batch_size, candidate_size, candidate_vector_dim
        Returns:
            (shape) batch_size, candidate_vector_dim
        """
        # batch_size, candidate_size, query_vector_dim
        temp = torch.tanh(self.linear(candidate_vector))
        # batch_size, candidate_size
        candidate_weights = F.softmax(torch.matmul(
            temp, self.attention_query_vector),
                                      dim=1)
        if self.writer is not None:
            assert candidate_weights.size(1) == len(self.names)
            if self.local_step % 10 == 0:
                self.writer.add_scalars(
                    self.tag, {
                        x: y
                        for x, y in zip(self.names,
                                        candidate_weights.mean(dim=0))
                    }, self.local_step)
            self.local_step += 1
        # batch_size, candidate_vector_dim
        target = torch.bmm(candidate_weights.unsqueeze(dim=1),
                           candidate_vector).squeeze(dim=1)
        return target


class Attention(torch.nn.Module):
    """
    Attention Net.
    Input embedding vectors (produced by KCNN) of a candidate news and all of user's clicked news,
    produce final user embedding vectors with respect to the candidate news.
    """

    def __init__(self, config):
        super(Attention, self).__init__()
        self.config = config
        # Calculate the dimensions correctly from config
        self.news_dim = config.num_filters
        self.concat_dim = self.news_dim * 2  # news vector + user vector

        self.dnn = nn.Sequential(nn.Linear(self.concat_dim, 16), nn.Linear(16, 1))

    def forward(self, candidate_news_vector, clicked_news_vector):
        """
        Args:
            candidate_news_vector: batch_size, len(window_sizes) * num_filters
            clicked_news_vector: batch_size, num_clicked_news_a_user, len(window_sizes) * num_filters
        Returns:
            user_vector: batch_size, len(window_sizes) * num_filters
        """
        # Ensure dimensions are correct
        # batch_size = clicked_news_vector.size(0) # Unused
        num_clicked = clicked_news_vector.size(1)

        # Expand candidate news to match clicked news for concatenation
        expanded_candidate = candidate_news_vector.unsqueeze(1).expand(
            -1, num_clicked, -1
        )

        # Concatenate along feature dimension
        concat_vectors = torch.cat((expanded_candidate, clicked_news_vector), dim=2)

        # Calculate attention weights
        attention_scores = self.dnn(concat_vectors).squeeze(dim=2)
        clicked_news_weights = F.softmax(attention_scores, dim=1)

        # Apply attention weights to get user vector
        user_vector = torch.bmm(
            clicked_news_weights.unsqueeze(dim=1), clicked_news_vector
        ).squeeze(dim=1)

        return user_vector
