from config import *
from input_embedding import InputEmbeddings
from encoder import Encoder
import torch.nn as nn
import torch

class TransformerClassifier(nn.Module):
    def __init__(self):
        '''
        A Transformer model for emotion classification
        
        Returns:
        classifier_out - Classification output (logits for each emotion class)
        '''
        super().__init__()
        self.d_model = D_MODEL
        self.num_layers = NUM_LAYERS
        self.vocab_size = VOCAB_SIZE
        self.num_classes = NUM_CLASSES
        
        # Input embeddings
        self.inp_embedding = InputEmbeddings()
        
        # Encoder stack
        self.encoder = Encoder(self.d_model, self.num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.d_model // 2, self.num_classes)
        )
        
    def forward(self, input_ids, attention_mask):
        # Get input embeddings
        embedded = self.inp_embedding(input_ids)  # (batch_size, seq_len, d_model)
        
        # Pass through encoder
        encoder_out = self.encoder(embedded, attention_mask)  # (batch_size, seq_len, d_model)
        
        # Global average pooling over sequence length
        # Mask out padded tokens for proper averaging
        mask_expanded = attention_mask.squeeze().unsqueeze(-1).expand(encoder_out.size()).float()
        sum_embeddings = torch.sum(encoder_out * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        
        # Classification
        classifier_out = self.classifier(pooled_output)  # (batch_size, num_classes)
        
        return classifier_out
    
def build_transformer_model():
    transformer = TransformerClassifier()
    # Xavier uniform initialization
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return transformer







