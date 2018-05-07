from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import NMF, LatentDirichletAllocation
import pickle
import numpy as np

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print ("Topic:" ,topic_idx)
        print (" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

# dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
# documents = dataset.data
#
# print(type(documents[0]))
dir = "../dataset/main_windows/"
target_dir = "../dataset/topics_20/"
files = [r"data_50.pickle", r"data_100.pickle", r"data_150.pickle", r"data_200.pickle"]
for file in files:
    with open(dir+file, "rb") as input_file:
        data = pickle.load(input_file)
    print("working on :"+file)
    documents = []

    for r in data:
        doc = ""
        for word in r:
            doc += " " + word
        documents.append(doc)


    no_features = 1000


    # LDA can only use raw term counts for LDA because it is a probabilistic graphical model
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
    tf = tf_vectorizer.fit_transform(documents)
    tf_feature_names = tf_vectorizer.get_feature_names()

    no_topics = 20

    # Run LDA
    lda = LatentDirichletAllocation(n_components=no_topics, max_iter=300, doc_topic_prior=1, topic_word_prior=.1, learning_method='online', learning_offset=50.,random_state=0).fit(tf)

    no_top_words = 10
    display_topics(lda, tf_feature_names, no_top_words)

    documents = np.array(documents).reshape(-1, 1)

    lda_Z = lda.fit_transform(tf)
    print(lda_Z.shape)  # (NO_DOCUMENTS, NO_TOPICS)

    t = file.split(".")[0].split("_")[1]
    with open(target_dir+r"topics_"+t+".pickle", "wb") as output_file:
        pickle.dump(lda_Z, output_file)
    print("---------------------------------------------------------")
#print(lda.score())
