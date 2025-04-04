from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def recommend(dataframe, input_paper_id):
    input_paper = input_paper_id
    print("**** Source Paper ****")
    print(dataframe["title"].iloc[input_paper])
    #print(f"Abstract : \n {dataframe["abstract"].iloc[input_paper]}")
    #print()
    #print()
    cosine_scores = []
    for i in range(len(dataframe)):
        score = cosine_similarity(dataframe["tfidf_embedding"].iloc[input_paper].reshape(
            1, -1), dataframe["tfidf_embedding"].iloc[i].reshape(1, -1))[0][0]
        cosine_scores.append(score)

    cosine_scores = np.array(cosine_scores)
    top_k = 5
    top_k_idx = np.argpartition(-cosine_scores, top_k)[:top_k]
    top_k_values = cosine_scores[top_k_idx]
    sorted_idx = top_k_idx[np.argsort(-top_k_values)]

    for id in sorted_idx:
        print(f"({cosine_scores[id]}) : {dataframe["title"].iloc[id]}")
        #print(f"Abstract : \n{dataframe["abstract"].iloc[id]}")
        #print()
        #print()
