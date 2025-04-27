import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path 

from ..prixfixe.predictor import Predictor
from .utils import n2id, revcomp

class AutosomePredictor(Predictor):
    def __init__(self,
                 model: nn.Module, 
                 model_pth: str | Path, 
                 device: torch.device):

        self.model = model.to(device)
        self.model.load_state_dict(torch.load(model_pth))  # Loading the model weights
        self.model = self.model.eval()

        self.use_reverse_channel = True
        self.use_single_channel = True
        self.seqsize = 250   # Your model is trained on sequences of length 250
        self.device = device

    def _preprocess_sequence(self, seq: str):
        """Preprocesses the input sequence to produce a tensor of shape (4, seqsize)."""
        seq_i = [n2id(x) for x in seq]  # Convert sequence to indices using n2id()
        code = torch.from_numpy(np.array(seq_i))  # Convert to tensor
        code = F.one_hot(code, num_classes=5)  # One-hot encoding (5 classes)
        
        code[code[:, 4] == 1] = 0.25  # Handle ambiguous nucleotides
        code = code[:, :4].float()  # Take only the first 4 columns (A, C, G, T)
        return code.transpose(0, 1)  # Transpose to shape (4, seqsize)

    def _add_channels(self, seq: torch.Tensor, rev_value: int):
        """Adds reverse and singleton channels to the input tensor."""
        to_concat = [seq]
        
        if self.use_reverse_channel:
            rev = torch.full((1, self.seqsize), rev_value, dtype=torch.float32)
            to_concat.append(rev)
            
        if self.use_single_channel:
            single = torch.full((1, self.seqsize), 0, dtype=torch.float32)
            to_concat.append(single)

        if len(to_concat) > 1:
            x = torch.concat(to_concat, dim=0)
        else:
            x = seq

        return x

    def predict(self, sequence: str) -> list[float]:
        """Generates predictions for the input sequence and its reverse complement."""
        # Check sequence length and trim if necessary
        if len(sequence) > self.seqsize:
            sequence = sequence[:self.seqsize]  # Trim to the expected input size

        # Process the forward sequence
        seq_tensor = self._preprocess_sequence(sequence)  # Preprocess sequence to get tensor
        x = self._add_channels(
            seq=seq_tensor,   # Pass the tensor here
            rev_value=0
        )
        
        # Process the reverse complement of the sequence
        rev_seq_tensor = self._preprocess_sequence(revcomp(sequence))  # Process the reverse complement
        x_rev = self._add_channels(
            seq=rev_seq_tensor,  # Pass the tensor here
            rev_value=1
        )
        
        # Move tensors to the correct device
        x = x.to(self.device)
        x_rev = x_rev.to(self.device)
        
        # Make predictions for forward and reverse sequences
        y = self.model(x[None])
        y_rev = self.model(x_rev[None])
        
        # Extract predictions for both heads
        y1 = y[0].detach().cpu().flatten().tolist()
        y2 = y[1].detach().cpu().flatten().tolist()
        
        y_rev1 = y_rev[0].detach().cpu().flatten().tolist()
        y_rev2 = y_rev[1].detach().cpu().flatten().tolist()
        
        # Average the predictions for each head separately
        y1 = (np.array(y1) + np.array(y_rev1)) / 2
        y2 = (np.array(y2) + np.array(y_rev2)) / 2
        
        # Return both predictions as a list
        return [y1.item(), y2.item()]