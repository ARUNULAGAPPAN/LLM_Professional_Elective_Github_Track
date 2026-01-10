import torch
import torch.nn as nn
import torch.optim as optim
from encoder import TransformerEncoderModel

vocab = {"[PAD]": 0, "Transformers": 1, "are": 2, "extremely": 3, "powerful": 4, "[MASK]": 5}
inv_vocab = {v: k for k, v in vocab.items()}

def train_model():
    print("\n" + "="*50)
    print("TRAINING MASKED LANGUAGE MODEL")
    print("="*50)
    
    input_seq = torch.tensor([[1, 2, 5, 4]])
    target_seq = torch.tensor([[1, 2, 3, 4]])
    
    model = TransformerEncoderModel(vocab_size=6, d_model=16, num_heads=2, num_layers=2, num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    print("\nTraining on: 'Transformers are [MASK] powerful'")
    print("Target word: 'extremely'\n")
    
    for epoch in range(100):
        mlm_logits, _ = model(input_seq)
        loss = criterion(mlm_logits.view(-1, 6), target_seq.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    print("\nTraining Complete!")
    return model

def predict(model, sentence):
    tokens = sentence.split()
    token_ids = []
    mask_position = -1
    
    for i, token in enumerate(tokens):
        if token in vocab:
            token_ids.append(vocab[token])
            if token == "[MASK]":
                mask_position = i
        else:
            print(f"Warning: '{token}' not in vocabulary, skipping...")
            return None
    
    if mask_position == -1:
        print("No [MASK] token found in the input!")
        return None
    
    input_tensor = torch.tensor([token_ids])
    
    model.eval()
    with torch.no_grad():
        logits, _ = model(input_tensor)
        predicted_id = logits[0, mask_position].argmax().item()
        predicted_word = inv_vocab[predicted_id]
    
    return predicted_word

def main():
    print("\n" + "="*50)
    print("TRANSFORMER ENCODER - MLM DEMO")
    print("="*50)
    
    print("\nAvailable vocabulary:")
    for word, idx in vocab.items():
        print(f"  {idx}: {word}")
    
    model = train_model()
    
    print("\n" + "="*50)
    print("INTERACTIVE PREDICTION MODE")
    print("="*50)
    print("\nEnter sentences with [MASK] to predict the missing word.")
    print("Example: Transformers are [MASK] powerful")
    print("Type 'quit' to exit.\n")
    
    while True:
        user_input = input("Enter sentence: ").strip()
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        predicted = predict(model, user_input)
        if predicted:
            filled_sentence = user_input.replace("[MASK]", f"**{predicted}**")
            print(f"Predicted word: {predicted}")
            print(f"Filled sentence: {filled_sentence}\n")

if __name__ == "__main__":
    main()
