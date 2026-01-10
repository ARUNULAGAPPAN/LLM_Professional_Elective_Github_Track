import torch
import torch.nn as nn
import torch.optim as optim
from transformer import Seq2SeqTransformer
from attention_masks import generate_causal_mask

# 1. Dataset
data = [
    ("What is self-attention?", "Self-attention relates each word to every other word"),
    ("Why is positional encoding required?", "Positional encoding provides word order information"),
    ("In the future, AI will", "In the future, AI will automate decision systems")
]

# Simple Character-level Tokenizer
chars = sorted(list(set("".join([i+j for i,j in data]) + "<SOS><EOS><PAD>")))
vocab = {ch: i for i, ch in enumerate(chars)}
def encode(s): return [vocab.get(c, 0) for c in s]

# 2. Model Setup
model = Seq2SeqTransformer(vocab_size=len(chars), d_model=32, nhead=4, num_layers=2)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 3. Training
print("Training the model...")
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
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

print("Training complete!\n")

# 4. Inference function
def generate_text(input_text, max_len=100):
    model.eval()
    src = torch.tensor([encode(input_text)])
    generated = encode("<SOS>")
    
    for _ in range(max_len):
        tgt = torch.tensor([generated])
        mask = generate_causal_mask(tgt.size(1))
        
        with torch.no_grad():
            output = model(src, tgt, mask)
            next_token = output[0, -1].argmax().item()
            generated.append(next_token)
            
            if next_token == vocab.get("<EOS>", -1):
                break
    
    # Decode and clean up output
    result = "".join([chars[i] for i in generated])
    result = result.replace("<SOS>", "").replace("<EOS>", "")
    return result

# 5. Interactive Demo
print("=" * 50)
print("Transformer Seq2Seq Demo")
print("=" * 50)
print("Type your input and press Enter to get a response.")
print("Type 'quit' or 'exit' to stop.\n")

while True:
    try:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
        
        response = generate_text(user_input)
        print(f"Model: {response}\n")
    except KeyboardInterrupt:
        print("\nGoodbye!")
        break
