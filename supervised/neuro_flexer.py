import random
import torch
from data_loader import DataLoader
from network import maskNLLLoss

cpu_device = torch.device("cpu")

class NeuroFlexer():
    def __init__(self, data_loader, encoder, decoder, device):
        self.data_loader = data_loader
        self.device = device
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)

    def move_to_device(self, *args):
        moved = [x.to(self.device) for x in args]
        return moved

    def train_on_batch(self, in_char_tensors, in_mask, out_char_tensors, out_mask, tag_tensors, encoder_optimizer, decoder_optimizer, teacher_forcing_ratio):
        in_char_tensors, in_mask, out_char_tensors, out_mask, tag_tensors = self.move_to_device(in_char_tensors, in_mask, out_char_tensors, out_mask, tag_tensors)
        self.encoder.train()
        self.decoder.train()
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        in_lens = in_mask.sum(axis=1).to(cpu_device)
        out_lens = out_mask.sum(axis=1)
        loss = 0
        batch_size = in_char_tensors.shape[0]
        encoder_output = self.encoder(in_char_tensors, in_lens)
        prev_char = torch.LongTensor([self.data_loader.characters.index("START") for _ in range(batch_size)]).to(self.device)

        decoder_hidden, decoder_cell = self.decoder.initialize_zero_step(batch_size, self.device)

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
                    lemma_char = torch.LongTensor([self.data_loader.characters.index("END") for _ in range(batch_size)]).to(self.device)
                decoder_output, _, decoder_hidden, decoder_cell = self.decoder(
                    prev_char, lemma_char, encoder_output, decoder_hidden, decoder_cell, tag_tensors
                )
                prev_char = out_char_tensors[t]
                mask_loss = maskNLLLoss(decoder_output, out_char_tensors[t], out_mask[t].bool(), self.device)
                loss += mask_loss
        else:
            for t in range(max_target_len):
                if t < len(in_char_tensors):
                    lemma_char = in_char_tensors[t]
                else:
                    lemma_char = torch.LongTensor([self.data_loader.characters.index("END") for _ in range(batch_size)]).to(self.device)
                decoder_output, _, decoder_hidden, decoder_cell = self.decoder(
                    prev_char, lemma_char, encoder_output, decoder_hidden, decoder_cell, tag_tensors
                )
                _, topi = decoder_output.topk(1)
                prev_char = torch.LongTensor([[topi[i][0] for i in range(batch_size)]]).squeeze().to(self.device)
                mask_loss = maskNLLLoss(decoder_output, out_char_tensors[t], out_mask[t].bool(), self.device)
                loss += mask_loss
        loss.backward()

        _ = torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1)
        _ = torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 1)

        encoder_optimizer.step()
        decoder_optimizer.step()
        return loss.item()

    def test_on_batch(self, in_char_tensors, in_mask, out_char_tensors, out_mask, tag_tensors):
        in_char_tensors, in_mask, out_char_tensors, out_mask, tag_tensors = self.move_to_device(in_char_tensors, in_mask, out_char_tensors, out_mask, tag_tensors)
        _ = self.encoder.eval()
        _ = self.decoder.eval()
        loss = 0
        batch_size = in_char_tensors.shape[0]
        with torch.no_grad():
            in_lens = in_mask.sum(axis=1).to(cpu_device)
            out_lens = out_mask.sum(axis=1)
            encoder_output = self.encoder(in_char_tensors, in_lens)
            prev_char = torch.LongTensor([self.data_loader.characters.index("START") for _ in range(batch_size)]).to(self.device)

            decoder_hidden, decoder_cell = self.decoder.initialize_zero_step(batch_size, self.device)

            out_char_tensors = out_char_tensors.permute([1,0])
            out_mask = out_mask.permute([1,0])
            in_char_tensors = in_char_tensors.permute([1,0])
            max_target_len = max(out_lens)
            top_indices = []
            for t in range(max_target_len):
                if t < len(in_char_tensors):
                    lemma_char = in_char_tensors[t]
                else:
                    lemma_char = torch.LongTensor([self.data_loader.characters.index("END") for _ in range(batch_size)]).to(self.device)
                decoder_output, _, decoder_hidden, decoder_cell = self.decoder(
                    prev_char, lemma_char, encoder_output, decoder_hidden, decoder_cell, tag_tensors
                )
                _, topi = decoder_output.topk(1)
                prev_char = torch.LongTensor([[topi[i][0] for i in range(batch_size)]]).squeeze().to(self.device)

                mask_loss = maskNLLLoss(decoder_output, out_char_tensors[t], out_mask[t].bool(), self.device)
                loss += mask_loss
                top_indices.append(topi)
        decoder_outputs = torch.cat(top_indices, axis=1)
        return loss, decoder_outputs

    def neural_process_word(self, lemma, full_tag):
        in_char_tensor, in_mask = self.data_loader.words_to_tensor([lemma])
        in_char_tensor, in_mask = self.move_to_device(in_char_tensor, in_mask)
        tag_tensor = self.data_loader.tags_to_tensors([full_tag]).to(self.device)

        with torch.no_grad():
            in_lens = in_mask.sum(axis=1)
            encoder_output = self.encoder(in_char_tensor, in_lens)
            prev_char = torch.LongTensor([self.data_loader.characters.index("START")]).to(self.device)

            decoder_hidden, decoder_cell = self.decoder.initialize_zero_step(1, self.device)

            in_char_tensor = in_char_tensor.permute([1,0])
            top_indices = []
            continuation = True
            t = 0
            while continuation:
                if t < len(in_char_tensor):
                    lemma_char = in_char_tensor[t]
                else:
                    lemma_char = torch.LongTensor([self.data_loader.characters.index("END")]).to(self.device)
                decoder_output, _, decoder_hidden, decoder_cell = self.decoder(
                    prev_char, lemma_char, encoder_output, decoder_hidden, decoder_cell, tag_tensor
                )
                _, topi = decoder_output.topk(1)
                continuation = topi[0].item() != self.data_loader.characters.index("END")
                prev_char = torch.LongTensor([[topi[0][0]]]).squeeze(0).to(self.device)

                top_indices.append(topi)
                t+=1
        decoder_outputs = torch.cat(top_indices, axis=1)
        word = decoder_outputs[0]
        chars = [self.data_loader.characters[i] for i in word if self.data_loader.characters[i]!="END"]
        inflected = "".join(chars)
        return inflected 


