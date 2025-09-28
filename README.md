## 📁 Project Structure

```
├── config.py                      # Configuration parameters for model and training
├── input_embedding.py             # Token + positional embedding layer
├── multi_head_attention.py        # Multi-head self-attention mechanism
├── layer_norm.py                  # Layer normalization module
├── feed_forward.py                # Feed-forward network used inside encoder blocks
├── encoder.py                     # Transformer encoder block and stack
├── transformer_classifier.py      # Encoder-only model architecture for classification
├── tokenizer.py                   # Text tokenizer and preprocessing utility
├── dataset.py                     # Dataset loading and preparation (for emotion detection)
├── train_utils.py                 # Utilities for training (schedulers, metrics, etc.)
├── train.py                       # Training script for the classifier
├── inference.py                   # Inference script for predicting emotions from text
```



# 🚀 Transformer Architectures from Scratch in PyTorch

This repository contains two major implementations of the Transformer architecture:

1. **Full Transformer (Encoder-Decoder)** — A clean, from-scratch PyTorch implementation of the original [*Attention is All You Need*](https://arxiv.org/abs/1706.03762) paper.
2. **Encoder-Only Transformer for Emotion Detection** — A custom-built encoder-only transformer model trained on a text-based emotion classification task.

Both models are implemented without using high-level modules like `torch.nn.Transformer`, allowing for full transparency and learning of the underlying mechanics of transformer-based architectures.

---

## 📁 File Structure




---

## 📌 1. Full Transformer (Encoder-Decoder)

- Implements all core components:
  - Scaled Dot-Product Attention
  - Multi-Head Attention
  - Positional Encoding
  - Encoder & Decoder blocks
  - Final linear + softmax layer
- Follows architecture described in *"Attention is All You Need"* (Vaswani et al., 2017)
- No external libraries like `torch.nn.Transformer` or `transformers` used

🛠 Ideal for:
- Learning internals of transformer architecture
- Educational demonstrations
- Extending into other seq2seq tasks (e.g., translation, summarization)

> 🗂 Note: This model is implemented as a base and **not trained** on a specific task in this repo.

---

## 📌 2. Encoder-Only Transformer for Emotion Detection

- Built using the encoder stack from the original transformer
- Trained on a text classification task for **emotion detection**
- Pipeline includes:
  - Custom tokenizer
  - Embedding & encoder layers
  - Classification head
  - Training and evaluation scripts

🎯 Task: Predict the emotion (e.g., joy, anger, sadness, etc.) expressed in a sentence.

🧪 Sample usage (inference):

```python
from inference import predict_emotion

text = "I can't believe how happy I am today!"
emotion = predict_emotion(text)
print(f"Predicted Emotion: {emotion}")



