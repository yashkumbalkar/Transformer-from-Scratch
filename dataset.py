from datasets import load_dataset
from torch.utils.data import Dataset
from tokenizers import Tokenizer
import torch
from config import *

class EmotionClassificationDataset(Dataset):

    def __init__(self, split: str, tokenizer: Tokenizer, max_seq_len: int):
        '''
        A PyTorch Dataset class to load the emotion classification dataset from HuggingFace 
        with all the preprocessing and tokenization completed.

        Args:
        split: Dataset split ('train', 'validation', 'test')
        tokenizer: Trained tokenizer
        max_seq_len: Maximum sequence length for input text

        Returns:
        model_inp: Model input data
        '''
        super().__init__()
        
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        # Load dataset from HuggingFace
        self.dataset = load_dataset(DATASET_ID, split=split)
        
        # Special tokens
        self.cls_token = torch.tensor([tokenizer.token_to_id("[CLS]")]).to(torch.int64)
        self.sep_token = torch.tensor([tokenizer.token_to_id("[SEP]")]).to(torch.int64)
        self.pad_token = torch.tensor([tokenizer.token_to_id("[PAD]")]).to(torch.int64)
        self.pad_token_id = tokenizer.token_to_id("[PAD]")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        text = data["text"] 
        label = data["label"] 
        
        # Tokenize text
        text_tokens = torch.tensor(self.tokenizer.encode(text).ids).to(torch.int64)
        
        # Truncate if too long
        if text_tokens.size(0) > self.max_seq_len - 2:  # -2 for [CLS] and [SEP]
            text_tokens = text_tokens[:self.max_seq_len - 2]
        
        # Calculate padding
        num_padding_tokens = self.max_seq_len - len(text_tokens) - 2  # -2 for [CLS] and [SEP]
        padding_tokens = torch.tensor([self.pad_token_id] * num_padding_tokens).to(torch.int64)
        
        # Create input sequence: [CLS] + text + [SEP] + [PAD]...
        input_ids = torch.cat([
            self.cls_token, 
            text_tokens, 
            self.sep_token, 
            padding_tokens
        ], dim=0)
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (input_ids != self.pad_token_id).unsqueeze(0).unsqueeze(0).to(torch.int64)
        
        # Convert label to tensor
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        model_inp = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label_tensor,
            "text": text
        }
        
        return model_inp
