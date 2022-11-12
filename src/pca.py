import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from sentence_bert import SentenceBertJapanese


class LivedoorPreprocesser:
    def __init__(self, data_dir, topic_names=None) -> None:
        self.data_dir = data_dir
        if topic_names is None:
            topics_dir = glob(os.path.join(data_dir, "*" + os.sep), recursive=True)
            self.topic_names = [os.path.basename(p.rstrip(os.sep)) for p in topics_dir]
        else:
            self.topic_names = topic_names

    def preprocess(self, file):
        with open(file) as f:
            texts = "".join(f.readlines()[2:])
        return texts

    def get(self, max_data_per_topic):
        df = pd.DataFrame(index=[], columns=["filename", "topic", "text"])
        for topic in self.topic_names:
            files_path = os.path.join(self.data_dir, topic + "/*")
            files_path = str(files_path)
            files = glob(files_path)
            for i in range(max_data_per_topic):
                file = files[i]
                texts = self.preprocess(file)
                basename = os.path.basename(file)
                df = df.append(
                    {"filename": basename, "topic": topic, "text": texts},
                    ignore_index=True,
                )
        return df


class PCAVisualizer:
    def __init__(self) -> None:
        pass

    def preprocess(self, embeddings):
        means = np.mean(embeddings, axis=1)
        stds = np.std(embeddings[:], axis=1)
        embeddings = (embeddings - means[:, None]) / (stds[:, None] + 1e-5)
        return embeddings

    def pca(self, texts):
        model = SentenceBertJapanese()
        embeddings = model.encode(texts).numpy()
        embeddings = self.preprocess(embeddings)
        pca = PCA()
        U, S, Vt = pca._fit_full(embeddings, min(embeddings.shape))
        cont_ratio = pca.explained_variance_ratio_
        transformed = np.matmul(embeddings, Vt.T)[:, :2]
        return transformed, Vt, cont_ratio

    def visualize_cont_ratio(self, cont_ratio):
        fig, ax = plt.subplots()
        ax.set_xlabel("ordered sigular value index")
        ax.set_ylabel("cumulative contribution rate")
        X = [i + 1 for i in range(len(cont_ratio))]
        Y = np.cumsum(cont_ratio)
        ax.plot(X, Y, label="contribution ratio")
        plt.savefig("../misc/contribution_ratio.png")

    def apply_pca(self, df):
        texts = df["text"].tolist()
        transformed, Vt, cont_ratio = self.pca(texts)

        np.savetxt("../misc/rotation.txt", Vt)
        self.visualize_cont_ratio(cont_ratio)

        df["X"] = transformed[:, 0]
        df["Y"] = transformed[:, 1]
        return df

    def visualize(self, df):
        df = self.apply_pca(df)

        topic_names = list(set(df["topic"].tolist()))
        fig, ax = plt.subplots()
        ax.set_xlabel("1st Principal Component")
        ax.set_ylabel("2nd Principal Component")
        for topic in topic_names:
            X = df[df["topic"] == topic]["X"].tolist()
            Y = df[df["topic"] == topic]["Y"].tolist()
            ax.scatter(X, Y, label=topic)
        ax.legend(loc=0)
        plt.savefig("../misc/pca_sentence_bert.png")
        plt.show()


if __name__ == "__main__":
    topics = ["sports-watch", "kaden-channel", "movie-enter", "smax"]
    preprocessor = LivedoorPreprocesser("../data/text", topics)
    df = preprocessor.get(max_data_per_topic=30)
    visualizer = PCAVisualizer()
    visualizer.visualize(df)
