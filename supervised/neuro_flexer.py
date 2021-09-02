import random
import torch
from data_loader import DataLoader
from network import maskNLLLoss

class NeuroFlexer():
    def __init__(self, data_loader, encoder, decoder):
        self.data_loader = data_loader
        self.encoder = encoder
        self.decoder = decoder

    def train_on_batch(self, in_char_tensors, in_mask, out_char_tensors, out_mask, tag_tensors, encoder_optimizer, decoder_optimizer, teacher_forcing_ratio):
        self.encoder.train()
        self.decoder.train()
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        in_lens = in_mask.sum(axis=1)
        out_lens = out_mask.sum(axis=1)
        loss = 0
        batch_size = in_char_tensors.shape[0]
        encoder_outputs, encoder_hidden = self.encoder(in_char_tensors, in_lens)
        prev_char = torch.LongTensor([self.data_loader.characters.index("START") for _ in range(batch_size)])

        decoder_hidden = encoder_hidden
        decoder_cell = torch.zeros(decoder_hidden.shape)

        out_char_tensors = out_char_tensors.permute([1,0])
        out_mask = out_mask.permute([1,0])
        in_char_tensors = in_char_tensors.permute([1,0])

        use_teacher_forcing = random.random() < teacher_forcing_ratio
        max_target_len = max(out_lens)
        if use_teacher_forcing:
            for t in range(max_target_len):
                if t < len(in_char_tensors):
                    lemma_char = in_char_tensors[t]
                else:
                    lemma_char = torch.LongTensor([self.data_loader.characters.index("END") for _ in range(batch_size)])
                decoder_output, _, decoder_hidden, decoder_cell = self.decoder(
                    prev_char, lemma_char, decoder_hidden, decoder_cell, tag_tensors
                )
                prev_char = out_char_tensors[t]
                mask_loss = maskNLLLoss(decoder_output, out_char_tensors[t], out_mask[t].bool())
                loss += mask_loss
        else:
            for t in range(max_target_len):
                if t < len(in_char_tensors):
                    lemma_char = in_char_tensors[t]
                else:
                    lemma_char = torch.LongTensor([self.data_loader.characters.index("END") for _ in range(batch_size)])
                decoder_output, _, decoder_hidden, decoder_cell = self.decoder(
                    prev_char, lemma_char, decoder_hidden, decoder_cell, tag_tensors
                )
                _, topi = decoder_output.topk(1)
                prev_char = torch.LongTensor([[topi[i][0] for i in range(batch_size)]]).squeeze()
                mask_loss = maskNLLLoss(decoder_output, out_char_tensors[t], out_mask[t].bool())
                loss += mask_loss
        loss.backward()

        _ = torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1)
        _ = torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 1)

        encoder_optimizer.step()
        decoder_optimizer.step()
        return loss.item()

    def test_on_batch(self, in_char_tensors, in_mask, out_char_tensors, out_mask, tag_tensors):
        _ = self.encoder.eval()
        _ = self.decoder.eval()
        loss = 0
        batch_size = in_char_tensors.shape[0]
        with torch.no_grad():
            in_lens = in_mask.sum(axis=1)
            out_lens = out_mask.sum(axis=1)
            encoder_outputs, encoder_hidden = self.encoder(in_char_tensors, in_lens)
            prev_char = torch.LongTensor([self.data_loader.characters.index("START") for _ in range(batch_size)])
            decoder_hidden = encoder_hidden
            decoder_cell = torch.zeros(decoder_hidden.shape)
            out_char_tensors = out_char_tensors.permute([1,0])
            out_mask = out_mask.permute([1,0])
            in_char_tensors = in_char_tensors.permute([1,0])
            max_target_len = max(out_lens)
            top_indices = []
            for t in range(max_target_len):
                if t < len(in_char_tensors):
                    lemma_char = in_char_tensors[t]
                else:
                    lemma_char = torch.LongTensor([self.data_loader.characters.index("END") for _ in range(batch_size)])
                decoder_output, _, decoder_hidden, decoder_cell = self.decoder(
                    prev_char, lemma_char, decoder_hidden, decoder_cell, tag_tensors
                )
                _, topi = decoder_output.topk(1)
                prev_char = torch.LongTensor([[topi[i][0] for i in range(batch_size)]]).squeeze()

                mask_loss = maskNLLLoss(decoder_output, out_char_tensors[t], out_mask[t].bool())
                loss += mask_loss
                top_indices.append(topi)
        decoder_outputs = torch.cat(top_indices, axis=1)
        return loss, decoder_outputs

    def neural_process_word(self, lemma, full_tag): 
        in_char_tensor, in_mask = self.data_loader.words_to_tensor([lemma])
        tag_tensor = self.data_loader.tags_to_tensors([full_tag])

        with torch.no_grad():
            in_lens = in_mask.sum(axis=1)
            encoder_outputs, encoder_hidden = self.encoder(in_char_tensor, in_lens)
            prev_char = torch.LongTensor([self.data_loader.characters.index("START")])
            decoder_hidden = encoder_hidden
            decoder_cell = torch.zeros(decoder_hidden.shape)
            in_char_tensor = in_char_tensor.permute([1,0])
            top_indices = []
            continuation = True
            t = 0
            while continuation:
                if t < len(in_char_tensor):
                    lemma_char = in_char_tensor[t]
                else:
                    lemma_char = torch.LongTensor([self.data_loader.characters.index("END")])
                decoder_output, _, decoder_hidden, decoder_cell = self.decoder(
                    prev_char, lemma_char, decoder_hidden, decoder_cell, tag_tensor
                )
                _, topi = decoder_output.topk(1)
                continuation = topi[0].item() != self.data_loader.characters.index("END")
                prev_char = torch.LongTensor([[topi[0][0]]]).squeeze(0)

                top_indices.append(topi)
                t+=1
        decoder_outputs = torch.cat(top_indices, axis=1)
        word = decoder_outputs[0]
        chars = [self.data_loader.characters[i] for i in word if self.data_loader.characters[i]!="END"]
        inflected = "".join(chars)
        return inflected 


