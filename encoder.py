import torch.nn as nn
from positional_encoding import PositionalEncoding
from attention import MultiHeadAttention

class TransformerEncoderModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.mlm_head = nn.Linear(d_model, vocab_size)
        self.classifier_head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        out = self.transformer_encoder(x)
        
        mlm_logits = self.mlm_head(out)
        cls_logits = self.classifier_head(out.mean(dim=1))
        
        return mlm_logits, cls_logits