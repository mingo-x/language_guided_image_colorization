import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
import utils


def get_caption_encoder(ver, *args):
    if 'lstm' in ver:
        return CaptionEncoderLSTM(*args)
    elif 'gru' in ver:
        return CaptionEncoderGRU(*args)


# standard bilstm
class CaptionEncoderLSTM(nn.Module):
    def __init__(self, word_embedding_dim, hidden_dim, vocab_size, train_vocab_embeddings, dropout=0.2, emb_freeze=True):
        super(CaptionEncoderLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, word_embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(train_vocab_embeddings))
        self.embedding.weight.requires_grad = (not emb_freeze)
        self.hidden_size = hidden_dim / 2
        self.lstm = nn.LSTM(word_embedding_dim, self.hidden_size, num_layers=1,
                            bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(p=dropout)
        utils.init_modules([self.lstm])
        print('LSTM caption encoder, dropout {}, embedding frozen {}'.format(dropout, emb_freeze))

    def forward(self, captions, lens):
        bsz, max_len = captions.size()
        embeds = self.dropout(self.embedding(captions))

        lens, indices = torch.sort(lens, 0, True)
        _, (enc_hids, _) = self.lstm(pack(embeds[indices], lens.tolist(), batch_first=True))
        enc_hids = torch.cat((enc_hids[0], enc_hids[1]), 1)
        _, _indices = torch.sort(indices, 0)
        enc_hids = enc_hids[_indices]

        return enc_hids


class CaptionEncoderGRU(nn.Module):
    def __init__(self, word_embedding_dim, hidden_dim, vocab_size, train_vocab_embeddings, dropout=0.2, emb_freeze=True):
        super(CaptionEncoderGRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, word_embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(train_vocab_embeddings))
        self.embedding.weight.requires_grad = (not emb_freeze)
        self.hidden_size = hidden_dim / 2
        self.gru = nn.GRU(
            word_embedding_dim, 
            self.hidden_size, 
            num_layers=1, 
            bidirectional=True, 
            batch_first=True)
        self.dropout = nn.Dropout(p=dropout)
        utils.init_modules([self.gru])
        print('GRU caption encoder, dropout {}, embedding frozen {}'.format(dropout, emb_freeze))

    def forward(self, captions, lens):
        bsz, max_len = captions.size()
        embeds = self.dropout(self.embedding(captions))

        lens, indices = torch.sort(lens, 0, True)
        _, enc_hids = self.gru(pack(embeds[indices], lens.tolist(), batch_first=True))
        enc_hids = torch.cat((enc_hids[0], enc_hids[1]), 1)
        _, _indices = torch.sort(indices, 0)
        enc_hids = enc_hids[_indices]

        return enc_hids


class FiLM(nn.Module):
    """
    A Feature-wise Linear Modulation Layer from
    'FiLM: Visual Reasoning with a General Conditioning Layer'

    How this layer works : 
    x = Variable(torch.randn(2, 64, 32 ,32))       
    gammas = Variable(torch.randn(2, 64)) # gammas and betas have to be 64 
    betas = Variable(torch.randn(2, 64))           
    y = film(x, gammas, betas)
    print y.size()
    y is : [2, 64, 32, 32]
 
    """
    def forward(self, x, gammas, betas):
        gammas = gammas.unsqueeze(2).unsqueeze(3).expand_as(x)
        betas = betas.unsqueeze(2).unsqueeze(3).expand_as(x)
        return (gammas * x) + betas


class FiLMV1(nn.Module):
    def forward(self, x, gammas, betas):
        gammas = gammas.unsqueeze(2).unsqueeze(3).expand_as(x)
        betas = betas.unsqueeze(2).unsqueeze(3).expand_as(x)
        return (gammas + 1) * x + betas


class FiLMWithAttn(nn.Module):
    def forward(self, x, gammas, betas, sa):
        gammas = gammas.unsqueeze(2).unsqueeze(3).expand_as(x)
        betas = betas.unsqueeze(2).unsqueeze(3).expand_as(x)
        return ((gammas + sa) * x) + betas
