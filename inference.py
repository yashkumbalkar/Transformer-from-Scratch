import torch
import torch.nn.functional as F
from config import *
from transformer_classifier import build_transformer_model
from tokenizer import tokenizer
from train import Trainer

class TextClassificationInference:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizer
        self.class_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']  # Update this for your dataset
        
        # Load model
        self.model = build_transformer_model()
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Special tokens
        self.cls_token_id = self.tokenizer.token_to_id("[CLS]")
        self.sep_token_id = self.tokenizer.token_to_id("[SEP]")
        self.pad_token_id = self.tokenizer.token_to_id("[PAD]")
        
        print(f"✓ Model loaded successfully on {self.device}")
        print(f"✓ Ready to classify text into: {self.class_labels}")
        
    def preprocess_text(self, text):
        """Preprocess single text input"""
        # Tokenize text
        text_tokens = torch.tensor(self.tokenizer.encode(text).ids, dtype=torch.long)
        
        # Truncate if too long
        if text_tokens.size(0) > MAX_SEQ_LEN - 2:
            text_tokens = text_tokens[:MAX_SEQ_LEN - 2]
        
        # Calculate padding
        num_padding = MAX_SEQ_LEN - len(text_tokens) - 2
        padding_tokens = torch.tensor([self.pad_token_id] * num_padding, dtype=torch.long)
        
        # Create input sequence: [CLS] + text + [SEP] + [PAD]...
        input_ids = torch.cat([
            torch.tensor([self.cls_token_id], dtype=torch.long),
            text_tokens,
            torch.tensor([self.sep_token_id], dtype=torch.long),
            padding_tokens
        ])
        
        # Create attention mask
        attention_mask = (input_ids != self.pad_token_id).unsqueeze(0).unsqueeze(0).long()
        
        return input_ids.unsqueeze(0), attention_mask  # Add batch dimension
    
    def predict(self, text):
        """Predict class for single text"""
        if not text.strip():
            return {
                'text': text,
                'predicted_class': 'unknown',
                'confidence': 0.0,
                'all_probabilities': {}
            }
            
        input_ids, attention_mask = self.preprocess_text(text)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)  # Raw logits
            probabilities = F.softmax(outputs, dim=1)        # Convert to probabilities
            predicted_class = torch.argmax(probabilities, dim=1).item()  # Get the class index
            confidence = probabilities[0][predicted_class].item()
        
        return {
            'text': text,
            'predicted_class': self.class_labels[predicted_class],
            'confidence': confidence,
            'all_probabilities': {
                label: prob.item() for label, prob in zip(self.class_labels, probabilities[0])
            }
        }





if __name__ == "__main__":

    # Training Model
    trainer = Trainer()
    trainer.train()

    # Prediction Using Trained Model
    model_path = f"/content/transformer_emotion_classifier/transformer_emotion_classifier_best_epoch_10.pt_epoch_10.pt"
  
    inference = TextClassificationInference(model_path)
  
    # Prediction Text
    text = "I am so happy today! Everything is going perfectly."
    result = inference.predict(text)
