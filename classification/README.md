## lstm classification written by pytorch
training scripts for lstm classification.

## environment
|module|version|
|:--|:--|
|python|3.6.6|
|torch|1.3.0.post2|
|scikit-learn|0.19.1|
|pandas|0.22.0|
|mecab-python3|0.996.3|

## how to use
1. download dataset(ldcc-20140209.tar.gz) from [here](https://www.rondhuit.com/download.html) on data directory.
2. unzip ldcc-20140209.tar.gz
3. start training

```bash
# 1. download dataset
cd classification/data/
wget https://www.rondhuit.com/download/ldcc-20140209.tar.gz
# 2. unzip dataset
tar xvf ldcc-20140209.tar.gz
# 3 start training
cd ../script
python train.py
```

## description
This is a repository for to make a lstm classification model.

利用しているデータセットは記事のデータで、記事のタイトルからカテゴリを予測するモデルを作成している。  
ダウンロードしたテキストファイルの3行目がタイトルとなっており、それをMecabで形態素解析して単語に分割して  
それぞれの単語をtorch.nn.Embeddingで固定の次元数を持つベクトル表現に変換する。  
これをタイトルを分かち書きしたリストに当てて、単語数 * embeddedに使う次元数のデータに変換する。  
(LSTMは単語の次元数が合っていれば入力できる(?)ので？？)

後は、softmaxで確率に変えてCrossEntropyでLOSSを計算して学習。  
タイトルから記事のカテゴリの予測を学習していく。

## feature work
- [ ] モデルの保存
- [ ] バッチ処理化
- [ ] dataloaderの利用
- [ ] Embeddedの方法をwikipediaのword2vecに変える

## reference
[PyTorchを使ってLSTMで文章分類を実装してみた
](https://qiita.com/m__k/items/841950a57a0d7ff05506)
