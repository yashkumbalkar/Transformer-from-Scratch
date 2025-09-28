## 📁 Project Structure


├── config.py # Configuration parameters for model and training

├── input_embedding.py # Token + positional embedding layer
├── multi_head_attention.py # Multi-head self-attention mechanism
├── layer_norm.py # Layer normalization module
├── feed_forward.py # Feed-forward network used inside encoder blocks
├── encoder.py # Transformer encoder block and stack
├── transformer_classifier.py # Encoder-only model architecture for classification
├── tokenizer.py # Text tokenizer and preprocessing utility
├── dataset.py # Dataset loading and preparation (for emotion detection)
├── train_utils.py # Utilities for training (schedulers, metrics, etc.)
├── train.py # Training script for the classifier
├── inference.py # Inference script for predicting emotions from text

