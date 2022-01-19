from torch import nn
import random
random.seed(1234)

import torch
import random

import torch
import torch.nn as nn
from torchvision.models import resnet18

class Encoder(nn.Module):
    def __init__(self, embedding_size):
        super(Encoder, self).__init__()

        self.backbone = nn.Sequential(*list(resnet18(pretrained=True).children())[:-1])
        self.embedding = nn.Linear(in_features= 512, out_features = embedding_size, bias=True)
    
    def forward(self, img):
        output = self.backbone(img)
        output = torch.flatten(output, start_dim=1)
        output = self.embedding(output)

        return output

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.GRU(
            embedding_size,
            hidden_size,
            num_layers=2,
            bidirectional=True
        )
        self.output_linear = nn.Linear(2*hidden_size, vocab_size)

    def forward(self, batch, hidden):

        batch_emb = self.embedding(batch)
        batch_emb = batch_emb.unsqueeze(0)
        outputs, hidden = self.rnn(batch_emb, hidden)

        return self.output_linear(outputs).squeeze(0), hidden

class Seq2seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, src_batch, trg_batch, teacher_forcing_prob=1.0):
        img_embedding = self.encoder(src_batch).unsqueeze(0)

        input_ids = trg_batch[:,0]
        batch_size = src_batch.shape[0]
        trg_max_len = len(trg_batch[0])

        outputs = torch.zeros(trg_max_len, batch_size, self.decoder.vocab_size).to(self.device)
        _, hidden = self.decoder.rnn(img_embedding)

        for t in range(1, trg_max_len):

            decoder_outputs, hidden = self.decoder(input_ids, hidden)

            outputs[t] = decoder_outputs
            _, top_ids = torch.max(decoder_outputs, dim=-1)

            input_ids = trg_batch[:,t] if random.random() > teacher_forcing_prob else top_ids
            
        return outputs
