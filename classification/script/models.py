# coding=utf-8
from torch import nn


class LSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, output_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim  # 中間層の次元数
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)  # 単語をベクトル化[vocab_size種類をembedding_dimで表現]
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, output_size)  # 中間層のアウトプットを分類したい数に整えるための線形変換
        self.softmax = nn.LogSoftmax(dim=1)  # 線形変換されたものを確率として出力するためのSoftmax

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        _, lstm_output = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_spaces = self.hidden2tag(lstm_output[0].view(-1, self.hidden_dim))
        tag_scores = self.softmax(tag_spaces)
        return tag_scores
