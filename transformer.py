import torch.nn as nn
from decoder import TransformerDecoder

class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super().__init__()
        # Encoder: Extracts meaning from Input
        self.encoder_emb = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Decoder: Generates Output
        self.decoder = TransformerDecoder(vocab_size, d_model, nhead, num_layers)

    def forward(self, src, tgt, tgt_mask):
        # 1. Encode source
        enc_out = self.encoder(self.encoder_emb(src))
        # 2. Decode using source memory and target mask
        out = self.decoder(tgt, enc_out, tgt_mask)
        return out