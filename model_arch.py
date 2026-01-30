import torch
import torch.nn as nn

class BiLSTMStudent(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(BiLSTMStudent, self).__init__()
        # 1. Embedding Layer: Turns word IDs into dense vectors
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # 2. Bi-LSTM: Processes text in both directions (forward and backward)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, 
                            num_layers=2, 
                            bidirectional=True, 
                            dropout=0.3, 
                            batch_first=True)
        
        # 3. Fully Connected Layer: Maps LSTM output to 2 classes (Pos/Neg)
        # Note: hidden_dim * 2 because it's bidirectional (concat forward + backward)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, text):
        # text shape: [batch_size, seq_len]
        embedded = self.dropout(self.embedding(text))
        
        # output: all hidden states, (hidden, cell): final states
        output, (hidden, cell) = self.lstm(embedded)
        
        # Concatenate the final hidden state from the forward and backward passes
        cat_hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        
        return self.fc(self.dropout(cat_hidden))