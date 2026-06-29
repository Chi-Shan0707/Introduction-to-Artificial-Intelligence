import torch

import torch.nn as nn

class SequenceClassifier(nn.Module):

    def __init__(self, vocab_size, embed_dim,hidden_size,num_classes):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.lstm = nn.LSTM(
            input_size=embed_dim,

            hidden_size=hidden_size,

            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):

        embedded = self.embedding(x)

        out, (h_n, c_n)= self.lstm(embedded)

        last_hidden_state = h_n[-1]

        logits = self.fc(last_hidden_state)

        return logits
    
