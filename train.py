import torch
import torch.nn as nn
import torch.optim as optim
from transformer import Seq2SeqTransformer
from attention_masks import generate_causal_mask

# 1. Dataset from your provided table (S36-S70)
data = [
    ("What is self-attention?", "Self-attention relates each word to every other word"),
    ("Why is positional encoding required?", "Positional encoding provides word order information"),
    ("In the future, AI will", "In the future, AI will automate decision systems")
]

# Simple Character-level Tokenizer for demonstration
chars = sorted(list(set("".join([i+j for i,j in data]) + "<SOS><EOS><PAD>")))
vocab = {ch: i for i, ch in enumerate(chars)}
def encode(s): return [vocab.get(c, 0) for c in s]

# 2. Model Setup
model = Seq2SeqTransformer(vocab_size=len(chars), d_model=32, nhead=4, num_layers=2)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 3. Training Loop
model.train()
for epoch in range(200):
    for src_text, tgt_text in data:
        src = torch.tensor([encode(src_text)])
        tgt = torch.tensor([encode("<SOS>" + tgt_text)])
        tgt_y = torch.tensor([encode(tgt_text + "<EOS>")])
        
        mask = generate_causal_mask(tgt.size(1))
        
        output = model(src, tgt, mask)
        loss = criterion(output.view(-1, len(chars)), tgt_y.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % 50 == 0: print(f"Epoch {epoch}, Loss: {loss.item()}")