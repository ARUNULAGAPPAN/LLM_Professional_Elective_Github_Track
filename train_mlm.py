import torch
import torch.nn as nn
import torch.optim as optim
from encoder import TransformerEncoderModel

vocab = {"[PAD]": 0, "Transformers": 1, "are": 2, "extremely": 3, "powerful": 4, "[MASK]": 5}
inv_vocab = {v: k for k, v in vocab.items()}

input_seq = torch.tensor([[1, 2, 5, 4]]) 
target_seq = torch.tensor([[1, 2, 3, 4]])

model = TransformerEncoderModel(vocab_size=6, d_model=16, num_heads=2, num_layers=2, num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    mlm_logits, _ = model(input_seq)
    
    loss = criterion(mlm_logits.view(-1, 6), target_seq.view(-1))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    logits, _ = model(input_seq)
    predicted_id = logits[0, 2].argmax().item()
    print(f"\nInput: Transformers are [MASK] powerful")
    print(f"Predicted word: {inv_vocab[predicted_id]}")