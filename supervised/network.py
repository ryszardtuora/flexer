from torch import nn, cat
import torch


class Encoder(nn.Module):
    def __init__(self, char_num, embedding_dim, encoder_dim):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(char_num, embedding_dim)
        self.recurrent = nn.LSTM(embedding_dim, encoder_dim, bidirectional=True)

    def forward(self, char_vector, input_lengths):
        embedded = self.embedding(char_vector)
        permuted = embedded.permute([1,0,2])
        total_length = permuted.size(0)
        packed = nn.utils.rnn.pack_padded_sequence(permuted, input_lengths, enforce_sorted=False)
        out, (hidden, cell) = self.recurrent(packed)
        hidden = torch.cat([hidden[0], hidden[1]], axis=1).unsqueeze(0)
        outputs, x = nn.utils.rnn.pad_packed_sequence(out, total_length=total_length)
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, char_num, embedding_dim, tag_dim, decoder_dim):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(char_num, embedding_dim)
        self.recurrent = nn.LSTM(2*embedding_dim + tag_dim, decoder_dim)
        self.classifier = nn.Linear(decoder_dim, char_num)
        self.softmax = nn.Softmax(dim=1)
        #self.dropout = nn.Dropout(0.5)

    def forward(self, prev_char_vector, lemma_char_vector, last_hidden, last_cell, tag_vector):
        embedded_prev_char = self.embedding(prev_char_vector)
        embedded_lemma_char = self.embedding(lemma_char_vector)
        concatenated = cat([embedded_lemma_char, embedded_prev_char, tag_vector], axis=1)
        concatenated = concatenated.unsqueeze(0)
        recurrent_out, (hidden, cell) = self.recurrent(concatenated, (last_hidden, last_cell))
        recurrent_out = recurrent_out.squeeze()
        classifier_out = self.classifier(recurrent_out)
        soft_out = self.softmax(classifier_out)
        return soft_out, recurrent_out, hidden, cell



def maskNLLLoss(inp, target, mask):
    err = torch.gather(inp, 1, target.view(-1, 1)).squeeze(1)
    crossEntropy = -torch.log(err)
    loss = crossEntropy.masked_select(mask).mean()
    return loss


#LSTM zamiast GRU
#Bidirectionality
