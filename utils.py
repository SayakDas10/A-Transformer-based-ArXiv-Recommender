from torch.utils.data import DataLoader, Dataset
import torch
import torch.optim as optim
import torch.nn as nn
from model import TransformerModel
from transformers import AutoTokenizer
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F


class ArxivDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.abstracts = dataframe["clean_abstract"]
        self.max_length = max_length

    def __len__(self):
        return len(self.abstracts)

    def __getitem__(self, index):
        encoding = self.tokenizer(self.abstracts[index], truncation=True,
                                  padding="max_length", max_length=self.max_length, return_tensors="pt")
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        return input_ids, attention_mask


def train(vocab_size, loader, epochs=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerModel(vocab_size=vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for input_ids, attention_mask in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            input_ids, attention_mask = input_ids.to(
                device), attention_mask.to(device)
            optimizer.zero_grad()
            logits, _ = model(input_ids, attention_mask)
            loss = criterion(
                logits[:, :-1].reshape(-1, vocab_size), input_ids[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    return model
    

def getCLSEmbedding(model, tokenizer, text):
    model.eval()
    with torch.no_grad():
        encoding = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
        input_ids = encoding["input_ids"].cuda()
        attention_mask = encoding["attention_mask"].cuda()
        _, cls_embedding = model(input_ids, attention_mask)
        return cls_embedding
    
def cosine_similarity(emb1, emb2):
    return F.cosine_similarity(emb1, emb2)


def main():
    dataframe = pd.read_csv("preprocessed_data.csv", sep='\t', dtype=object)
    # print(type(dataframe["clean_abstract"].iloc[0]))

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    train_dataset = ArxivDataset(dataframe, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    model = train(vocab_size=tokenizer.vocab_size, loader=train_loader)
    
    abs1, abs2 = dataframe["clean_abstract"].iloc[0], dataframe["clean_abstract"].iloc[1]
    emb1 = getCLSEmbedding(model, tokenizer,  abs1)
    emb2 = getCLSEmbedding(model, tokenizer,  abs2)
    print(f"Cosine Similarity 👉 {cosine_similarity(emb1, emb2)}")
    

if __name__ == "__main__":
    main()
