from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from recommend import recommend


def main():
    dataframe = pd.read_csv("preprocessed_data.csv", sep='\t')
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(dataframe["clean_abstract"])
    dataframe["tfidf_embedding"] = list(tfidf_matrix.toarray())

    recommend(dataframe, 5678)

    # print(type(np.array(tfidf_df[:1])[0]))
    # print(dataframe["tfidf_embedding"].iloc[0])
    # print(type(dataframe["tfidf_embedding"].iloc[0][0]))
    # print(cosine_similarity(dataframe["tfidf_embedding"].iloc[0].reshape(1, -1), dataframe["tfidf_embedding"].iloc[1].reshape(1, -1))[0][0])
    # print(cosine_similarity(dataframe["tfidf_embedding"].iloc[0].reshape(1, -1), dataframe["tfidf_embedding"].iloc[2].reshape(1, -1))[0][0])
    # print(cosine_similarity(dataframe["tfidf_embedding"].iloc[0].reshape(1, -1), dataframe["tfidf_embedding"].iloc[3].reshape(1, -1))[0][0])


if __name__ == "__main__":
    main()
