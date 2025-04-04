from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

def main():
    tfidf = pd.read_csv("tfidf_data.csv", sep='\t')
    tfidf_matrix = np.array(tfidf["tfidf_embedding"].tolist())
    print(type(tfidf_matrix[0][0]))
    print(tfidf_matrix[0][0])
    
    
if __name__ == "__main__":
    main()
    