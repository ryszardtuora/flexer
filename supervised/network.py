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
        outputs, lengths  = nn.utils.rnn.pad_packed_sequence(out, total_length=total_length)
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim):
        super(Attention, self).__init__()
        self.encoder_layer = nn.Linear(encoder_dim, decoder_dim)
        self.decoder_layer = nn.Linear(decoder_dim, decoder_dim)
        self.final_layer = nn.Parameter(torch.FloatTensor(decoder_dim).uniform_(-0.1, 0.1))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, decoder_hidden, encoder_hiddens):
        num_timesteps = encoder_hiddens.shape[0]
        processed_decoder = self.decoder_layer(decoder_hidden.repeat(num_timesteps, 1, 1))
        processed_encoder = self.encoder_layer(encoder_hiddens)
        out = torch.tanh(processed_encoder + processed_decoder) @ self.final_layer
        weights = self.softmax(out).unsqueeze(1).permute((2,1,0))
        weighed = torch.bmm(weights, encoder_hiddens.permute((1,0,2))).squeeze()
        return weighed

class Decoder(nn.Module):
    def __init__(self, char_num, embedding_dim, tag_dim, encoder_dim, decoder_dim, use_attention):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(char_num, embedding_dim)
        self.use_attention = use_attention
        if self.use_attention:
            self.attention = Attention(encoder_dim, decoder_dim)
            self.recurrent = nn.LSTM(embedding_dim + tag_dim + encoder_dim, decoder_dim)
        else:
            self.recurrent = nn.LSTM(2*embedding_dim + tag_dim + encoder_dim, decoder_dim)
        self.decoder_dim = decoder_dim
        self.classifier = nn.Linear(decoder_dim, char_num)
        self.softmax = nn.Softmax(dim=1)
        #self.dropout = nn.Dropout(0.5)

    def initialize_zero_step(self, batch_size, device):
        zero_hidden = torch.zeros((1, batch_size, self.decoder_dim)).to(device)
        zero_cell = torch.zeros((1, batch_size, self.decoder_dim)).to(device)
        return zero_hidden, zero_cell

    def forward(self, prev_char_vector, lemma_char_vector, encoder_output, last_hidden, last_cell, tag_vector):
        embedded_prev_char = self.embedding(prev_char_vector)
        if self.use_attention:
            encoder_hiddens = encoder_output[0]
            attended_encoder = self.attention(last_hidden, encoder_hiddens)
            concatenated = cat([attended_encoder, embedded_prev_char, tag_vector], axis=1)

        else:
            embedded_lemma_char = self.embedding(lemma_char_vector)
            encoder_hidden =  encoder_output[1][0]
            concatenated = cat([embedded_lemma_char, encoder_hidden, embedded_prev_char, tag_vector], axis=1)
        concatenated = concatenated.unsqueeze(0)
        recurrent_out, (hidden, cell) = self.recurrent(concatenated, (last_hidden, last_cell))
        recurrent_out = recurrent_out.squeeze()
        classifier_out = self.classifier(recurrent_out)
        if len(classifier_out.shape) < 2:
            classifier_out = classifier_out.unsqueeze(0)
        soft_out = self.softmax(classifier_out)
        return soft_out, recurrent_out, hidden, cell



def maskNLLLoss(inp, target, mask, device):
    err = torch.gather(inp, 1, target.view(-1, 1)).squeeze(1)
    crossEntropy = -torch.log(err)
    loss = crossEntropy.masked_select(mask).mean().to(device)
    return loss


#LSTM zamiast GRU
#Bidirectionality
