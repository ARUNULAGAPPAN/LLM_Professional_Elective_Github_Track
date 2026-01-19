 Transformer Encoder with Masked Language Modeling (MLM) & Sentence Classification
Project Overview

This project demonstrates the implementation and conceptual understanding of the Transformer Encoder architecture, focusing on:

Autoencoding via Masked Language Modeling (MLM)

Sentence-level classification

Parallel sequence processing using Self-Attention

Unlike traditional RNN-based models, the Transformer Encoder processes all tokens simultaneously, enabling efficient learning of global contextual relationships.

 Objective

The primary objectives of this experiment are:

To implement a Transformer Encoder from first principles

To understand how Self-Attention replaces recurrence

To explore autoencoding through MLM

To perform sentence classification using a shared encoder backbone

🏗️ Architecture Overview
Input Tokens
     │
     ▼
Word Embedding
     │
     ▼
Positional Encoding
     │
     ▼
+------------------------------------+
|        Encoder Layer × N            |
|                                    |
| 1. Multi-Head Self-Attention        |
| 2. Add & Layer Normalization        |
| 3. Feed-Forward Network             |
| 4. Add & Layer Normalization        |
+------------------------------------+
               │
     ┌─────────┴─────────┐
     │                   │
     ▼                   ▼
MLM Head          Classification Head
(Token Prediction)   (Sentence Label)


🧠 Core Concepts Explained
🔹 Positional Encoding

Transformers lack recurrence and convolution. To preserve token order, positional information is injected using sine and cosine functions of varying frequencies.

This allows the model to:

Distinguish word positions

Generalize to longer sequences

Learn relative distances between tokens

🔹 Self-Attention Mechanism (Q, K, V)

Self-Attention enables each token to attend to every other token in the sequence.

Query (Q) – What the token is searching for

Key (K) – What the token represents

Value (V) – The information the token provides

The attention score determines how much focus one word gives to another, enabling deep contextual understanding.

🔹 Multi-Head Attention

Instead of a single attention function, the model uses multiple attention heads to capture:

Syntactic relationships

Semantic dependencies

Long-range interactions

Each head attends to different aspects of the sentence.

🔹 Autoencoding with Masked Language Modeling (MLM)

In MLM:

Certain words are replaced with a [MASK] token

The model predicts the original word using both left and right context

This forces the encoder to learn bidirectional representations, unlike traditional left-to-right language models.

📊 Results & Visualization
🔥 Attention Heatmap (Visualization)

(Insert your generated Seaborn / Matplotlib attention heatmap here)

The visualization demonstrates that the token “powerful” attends strongly to “Transformers”, indicating that the model correctly captures the subject–attribute relationship.

🧪 Sample Input & Output
Type	Sentence
Input	Transformers are [MASK] powerful
Target	extremely
Model Prediction	extremely

✅ The correct prediction confirms effective contextual understanding by the encoder.

🧩 Key Takeaways

Transformer Encoders eliminate recurrence using Self-Attention

MLM enables bidirectional contextual learning

A single encoder can support multiple downstream tasks

Attention visualizations provide model interpretability

📚 Applications

Language understanding (BERT-style models)

Text classification

Question answering

Sentence similarity

Pretraining large language models

🏁 Conclusion

This project successfully demonstrates how a Transformer Encoder learns rich, contextual representations through Self-Attention and MLM. The architecture highlights why Transformers have become the backbone of modern NLP systems.

