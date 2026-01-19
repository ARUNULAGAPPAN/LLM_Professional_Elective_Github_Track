🔹 Objective
To implement and understand the Transformer Encoder architecture. This experiment focuses on Autoencoding through Masked Language Modeling (MLM) and sentence classification. Unlike RNNs, the Encoder processes all words in a sentence simultaneously using Self-Attention.
🏗️ Architecture Diagram
code
Text
Input Tokens -> Word Embedding -> Positional Encoding
                                      |
         +----------------------------+
         |   [ Encoder Layer x N ]
         |   1. Multi-Head Self-Attention
         |   2. Add & Norm (Residual)
         |   3. Feed-Forward Network
         |   4. Add & Norm (Residual)
         +----------------------------+
                      |
        +-------------+-------------+
        |                           |
  [ MLM Head ]               [ Classifier Head ]
(Predicts Mask)            (Predicts Sentiment)
🔍 Detailed Architecture Explanation
Positional Encoding: Since Transformers have no recurrence, we use Sine and Cosine functions to inject the relative position of words into the embeddings.
Self-Attention (
Q
,
K
,
V
Q,K,V
): The model calculates a "score" for every word relative to every other word.
Query (
Q
Q
): What I am looking for.
Key (
K
K
): What I contain.
Value (
V
V
): The information I provide.
Autoencoding (MLM): By masking a word (e.g., [MASK]), we force the encoder to use the Global Context (words to the left and right) to reconstruct the original input.
📊 Results & Visualization
Attention Heatmap
(Placeholder: Add your generated seaborn heatmap here)
The heatmap shows how the word "powerful" attends strongly to "Transformers", demonstrating that the model understands the subject-attribute relationship.
Sample Input/Output
Input: Transformers are [MASK] powerful
Target: extremely
Model Prediction: extremely
