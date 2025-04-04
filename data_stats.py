import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    dataframe = pd.read_csv("tfidf_data.csv", sep="\t")
    abstract_lengths = []
    for abstract in dataframe["clean_abstract"]:
        abstract_lengths.append(len(abstract))

    stats = {
        'Min': min(abstract_lengths),
        'Mean': np.mean(abstract_lengths),
        'Median': np.median(abstract_lengths),
        '90th %': np.percentile(abstract_lengths, 90),
        '95th %': np.percentile(abstract_lengths, 95),
        '99th %': np.percentile(abstract_lengths, 99),
        'Max': max(abstract_lengths)
    }

    plt.figure(figsize=(8, 5))
    bars = plt.bar(stats.keys(), stats.values(), color='blue')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 1, f'{height:.1f}',
                 ha='center', va='bottom', fontsize=10)

    plt.title('Abstract Length Statistics')
    plt.ylabel('Number of Tokens')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("data_stats.png")


if __name__ == "__main__":
    main()
