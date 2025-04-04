import re
import pandas as pd
import nltk
import string
nltk.download("punkt")

# https://arxiv.org/help/api/user-manual
category_map = {'cs.AI': 'Artificial Intelligence',
                # 'cs.AR': 'Hardware Architecture',
                # 'cs.CC': 'Computational Complexity',
                # 'cs.CE': 'Computational Engineering, Finance, and Science',
                # 'cs.CG': 'Computational Geometry',
                # 'cs.CL': 'Computation and Language',
                # 'cs.CR': 'Cryptography and Security',
                # 'cs.CV': 'Computer Vision and Pattern Recognition',
                # 'cs.CY': 'Computers and Society',
                # 'cs.DB': 'Databases',
                # 'cs.DC': 'Distributed, Parallel, and Cluster Computing',
                # 'cs.DL': 'Digital Libraries',
                # 'cs.DM': 'Discrete Mathematics',
                # 'cs.DS': 'Data Structures and Algorithms',
                # 'cs.ET': 'Emerging Technologies',
                # 'cs.FL': 'Formal Languages and Automata Theory',
                # 'cs.GR': 'Graphics',
                # 'cs.GT': 'Computer Science and Game Theory',
                # 'cs.HC': 'Human-Computer Interaction',
                # 'cs.IR': 'Information Retrieval',
                # 'cs.IT': 'Information Theory',
                # 'cs.LG': 'Machine Learning',
                # 'cs.LO': 'Logic in Computer Science',
                # 'cs.MA': 'Multiagent Systems',
                # 'cs.MM': 'Multimedia',
                # 'cs.MS': 'Mathematical Software',
                # 'cs.NA': 'Numerical Analysis',
                # 'cs.NE': 'Neural and Evolutionary Computing',
                # 'cs.NI': 'Networking and Internet Architecture',
                # 'cs.OH': 'Other Computer Science',
                # 'cs.OS': 'Operating Systems',
                # 'cs.PF': 'Performance',
                # 'cs.PL': 'Programming Languages',
                # 'cs.RO': 'Robotics',
                # 'cs.SC': 'Symbolic Computation',
                # 'cs.SE': 'Software Engineering',
                # 'cs.SI': 'Social and Information Networks',
                # 'cs.SY': 'Systems and Control',
                # 'eess.AS': 'Audio and Speech Processing',
                # 'eess.IV': 'Image and Video Processing',
                # 'eess.SP': 'Signal Processing',
                'stat.ML': 'Machine Learning'
                }


def getDataframe(path, columns, categories, chunk_size=10000):
    chunks = pd.read_json(path, lines=True, chunksize=chunk_size)
    df_list = [chunk[columns] for chunk in chunks]
    df = pd.concat(df_list, ignore_index=True)
    df = df[df["categories"].isin(categories.keys())]
    print(f"Created Dataframe With Keys 👉 {df.keys()}")
    print(f"Dataframe Shape 👉 {df.shape}")
    return df


def checkForNone(dataframe):
    for key in dataframe.keys():
        assert not dataframe[key].hasnans, f"{key} has None 😖"
    print("No None Values Found! 👌")


def cleanAbstract(text):
    if not isinstance(text, str):
        return ""

    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(
        f"[{re.escape(string.punctuation.replace("%", ''))}]", "", text)
    text = text.lower()
    return text


def main():
    dataset_path = "arxiv-metadata-oai-snapshot.json"
    columns = ["id", "authors", "title", "categories", "abstract"]
    dataframe = getDataframe(dataset_path, columns, category_map)
    checkForNone(dataframe)
    dataframe["clean_abstract"] = dataframe["abstract"].apply(cleanAbstract)
    print(dataframe['clean_abstract'])
    dataframe.to_csv("preprocessed_data.csv", sep='\t', encoding='utf-8')


if __name__ == "__main__":
    main()
