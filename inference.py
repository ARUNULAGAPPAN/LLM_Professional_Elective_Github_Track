def generate_text(model, start_phrase, max_len=50):
    model.eval()
    src = torch.tensor([encode(start_phrase)])
    generated = encode("<SOS>")
    
    for _ in range(max_len):
        tgt = torch.tensor([generated])
        mask = generate_causal_mask(tgt.size(1))
        
        with torch.no_grad():
            output = model(src, tgt, mask)
            next_token = output[0, -1].argmax().item()
            generated.append(next_token)
            
            if next_token == vocab["<EOS>"]: break
            
    return "".join([chars[i] for i in generated])