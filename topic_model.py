import topmost
from topmost.data import download_dataset
from fastopic import FASTopic

# download_dataset("NeurIPS", cache_path="./data/NeurIPS")
dataset = topmost.data.DynamicDataset("./data/Wikitext_103/Wikitext-103", as_tensor=True)

for x in dataset.train_dataloader:
    print(x["bow"].shape)
# docs = dataset.train_texts

# model = FASTopic(num_topics=50, verbose=True)
# topic_top_words, doc_topic_dist = model.fit_transform(docs)

# from fastopic import FASTopic
# from sklearn.datasets import fetch_20newsgroups
# from topmost.preprocessing import Preprocessing

# docs = fetch_20newsgroups(data_home="data/20NG/20news_home",subset='all', remove=('headers', 'footers', 'quotes'))['data']
# print(docs[0])
# preprocessing = Preprocessing(vocab_size=10000, stopwords='English')

# model = FASTopic(50, preprocessing)
# topic_top_words, doc_topic_dist = model.fit_transform(docs)






# from sklearn.datasets._base import load_files
# import codecs
# import os
# import pickle
# train_path = "data/20NG/20news_home/20news-bydate-train"
# test_path = "data/20NG/20news_home/20news-bydate-test"
# cache_path = "data/20NG/20news_home/20news-bydate.pkz"
# cache = dict(
#     train=load_files(train_path, encoding="latin1"),
#     test=load_files(test_path, encoding="latin1"),
# )
# compressed_content = codecs.encode(pickle.dumps(cache), "zlib_codec")

# with open(cache_path, "wb") as f:
#     f.write(compressed_content)