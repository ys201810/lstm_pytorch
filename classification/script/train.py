# coding=utf-8
import os
import re
import glob
import MeCab
import torch
from torch import nn
import pickle
import linecache
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.optim as optim
import sys
sys.path.append(os.path.join('./', '..', '..', '..'))
from classification.script.models import LSTMClassifier


def make_dataset(data_dir):
    categories = [dir_name for dir_name in os.listdir(data_dir) if os.path.isdir(data_dir + dir_name)]

    datasets = pd.DataFrame(columns=['title', 'category'])
    for category in categories:
        text_files = glob.glob(os.path.join(data_dir, category, '*.txt'))
        for text_file in text_files:
            title = linecache.getline(text_file, 3)
            data = pd.Series([title, category], index=datasets.columns)
            datasets = datasets.append(data, ignore_index=True)

    # データをシャッフル
    datasets = datasets.sample(frac=1).reset_index(drop=True)
    return datasets


def make_wakati(sentence):
    """　文章を分かち書きしたリストにする。

    Args:
        sentence (string):

    Returns:
        list: 記号や英語が削除された、日本語の単語の分かち書きされたリスト
    """
    tagger = MeCab.Tagger('-Owakati')
    sentence = tagger.parse(sentence)
    sentence = re.sub(r'[0-9０-９a-zA-Zａ-ｚＡ-Ｚ]+', " ", sentence) # 半角全角英数字除去
    sentence = re.sub(r'[\．_－―─！＠＃＄％＾＆\-‐|\\＊\“（）＿■×+α※÷⇒—●★☆〇◎◆▼◇△□(：〜～＋=)／*&^%$#@!~`){}［］…\[\]\"\'\”\’'
                      r':;<>?＜＞〔〕〈〉？、。・,\./『』【】「」→←○《》≪≫\n\u3000]+', "", sentence)  # 記号を削除
    wakati = sentence.split(' ')
    wakati = [word for word in wakati if word != ""]  # 空を削除
    return wakati


def sentence2index(sentence, word2index):
    """文章を単語に分割し、単語ごとのindex numを持つ配列にする"""
    wakati = make_wakati(sentence)
    return torch.tensor([word2index[word] for word in wakati], dtype=torch.long)


def main():
    data_dir = '../data/text/'
    dataset_pickle_file = os.path.join('../', 'data', 'text', 'title_category_dataset.pickle')
    category2index = {
        'movie-enter': 0, 'it-life-hack': 1, 'kaden-channel': 2, 'topic-news': 3, 'livedoor-homme': 4, 'peachy': 5,
        'sports-watch': 6, 'dokujo-tsushin': 7, 'smax': 8
    }

    if not os.path.exists(dataset_pickle_file):
        datasets = make_dataset(data_dir)
        with open(dataset_pickle_file, 'wb') as pickle_write_file:
            pickle.dump(datasets, pickle_write_file)
    else:
        with open(dataset_pickle_file, 'rb') as pickle_read_file:
            datasets = pickle.load(pickle_read_file)

    word2index = {}
    for title in datasets['title']:
        wakati_title = make_wakati(title)
        for word in wakati_title:
            if word in word2index:
                continue
            word2index[word] = len(word2index)

    print('vocab size:{}'.format(len(word2index)))

    vocab_size = len(word2index)
    embedding_dim = 10
    hidden_dim = 128
    output_size = len(category2index)

    train_data, test_data = train_test_split(datasets, train_size=0.7)

    model = LSTMClassifier(embedding_dim, hidden_dim, vocab_size, output_size)
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    train_num, test_num = len(train_data), len(test_data)
    train_losses, eval_losses = [], []
    accuracies = []  # [TODO] 比較して精度計算する

    for epoch in range(5):
        train_loss = 0
        train_correct_num = 0

        for title, cat in zip(train_data['title'], train_data['category']):
            model.zero_grad()
            inputs = sentence2index(title, word2index)
            outputs = model(inputs)
            gt = torch.tensor([category2index[cat]], dtype=torch.long)

            _, predict = torch.max(outputs, 1)
            if gt == predict:
                train_correct_num += 1

            loss = criterion(outputs, gt)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss)
        print('epoch:{}\t train loss:{}\t accuracy:{}'.format(
            epoch, train_loss, round(train_correct_num / train_num, 3)))

        # テストデータを確認
        test_loss = 0
        test_correct_num = 0
        with torch.no_grad():
            for title, cat in zip(test_data['title'], test_data['category']):
                inputs = sentence2index(title, word2index)
                outputs = model(inputs)
                gt = torch.tensor([category2index[cat]], dtype=torch.long)

                _, predict = torch.max(outputs, 1)
                if gt == predict:
                    test_correct_num += 1

                loss = criterion(outputs, gt)
                test_loss += loss.item()
            eval_losses.append(test_loss)
            print('epoch:{}\t eval loss:{}\t accuracy:{}'.format(
                epoch, test_loss, round(test_correct_num / test_num, 3)))


if __name__ == '__main__':
    main()
