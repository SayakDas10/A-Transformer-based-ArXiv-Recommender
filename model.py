import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_head=8, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead= n_head), num_layers=num_layers)
        self.output_projection = nn.Linear(d_model, vocab_size)
    
    def forward(self, x, attention_masks):
        x = self.embedding(x)
        x = self.transformer(x)
        logits = self.output_projection(x)
        return logits, x[:, 0, :]
    

         
        
        
