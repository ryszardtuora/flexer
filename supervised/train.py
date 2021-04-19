import random
from constants import CHARACTERS, ALL_FEATS
from utils import lines_to_training_examples
from network import Encoder, Decoder, maskNLLLoss
from torch import optim
from tqdm import tqdm
import torch

#encoder and decoder optimizer
def train_on_batch(in_char_tensors, in_mask, out_char_tensors, out_mask, tag_tensors, encoder, decoder,
                   encoder_optimizer, decoder_optimizer):
    encoder.train()
    decoder.train()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    in_lens = in_mask.sum(axis=1)
    out_lens = out_mask.sum(axis=1)
    loss = 0
    batch_size = in_char_tensors.shape[0]
    encoder_outputs, encoder_hidden = encoder(in_char_tensors, in_lens)
    decoder_input = torch.LongTensor([CHARACTERS.index("START") for _ in range(batch_size)])
    decoder_hidden = encoder_hidden
    decoder_cell = torch.zeros(decoder_hidden.shape)

    out_char_tensors = out_char_tensors.permute([1,0])
    out_mask = out_mask.permute([1,0])

    use_teacher_forcing = random.random() < TEACHER_FORCING_RATIO
    max_target_len = max(out_lens)
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, _, decoder_hidden, decoder_cell = decoder( #2gi czy 3ci jako hidden?
                decoder_input, decoder_hidden, decoder_cell, tag_tensors
            )
            decoder_input = out_char_tensors[t]
            mask_loss = maskNLLLoss(decoder_output, out_char_tensors[t], out_mask[t].bool())
            loss += mask_loss
    else:
        for t in range(max_target_len):
            decoder_output, _, decoder_hidden, decoder_cell = decoder(
                decoder_input, decoder_hidden, decoder_cell, tag_tensors
            )
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]]).squeeze()
            mask_loss = maskNLLLoss(decoder_output, out_char_tensors[t], out_mask[t].bool())
            loss += mask_loss
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.item()


def test_on_batch(in_char_tensors, in_mask, out_char_tensors, out_mask, tag_tensors, encoder, decoder):
    _ = encoder.eval()
    _ = decoder.eval()
    loss = 0
    batch_size = in_char_tensors.shape[0]
    with torch.no_grad():
        in_lens = in_mask.sum(axis=1)
        out_lens = out_mask.sum(axis=1)
        encoder_outputs, encoder_hidden = encoder(in_char_tensors, in_lens)
        decoder_input = torch.LongTensor([CHARACTERS.index("START") for _ in range(batch_size)])
        decoder_hidden = encoder_hidden
        decoder_cell = torch.zeros(decoder_hidden.shape)

        out_char_tensors = out_char_tensors.permute([1,0])
        out_mask = out_mask.permute([1,0])
        max_target_len = max(out_lens)
        top_indices = []
        for t in range(max_target_len):
            decoder_output, _, decoder_hidden, decoder_cell = decoder(
                decoder_input, decoder_hidden, decoder_cell, tag_tensors
            )
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]]).squeeze()
            mask_loss = maskNLLLoss(decoder_output, out_char_tensors[t], out_mask[t].bool())
            loss += mask_loss
            top_indices.append(topi)
    decoder_outputs = torch.cat(top_indices, axis=1)
    return loss, decoder_outputs


def accuracy(out, target):
    correct, total = 0, 0
    for out_seq, target_seq in zip(out, target):
        total += 1
        for out_char, target_char in zip(out_seq, target_seq):
            if out_char != target_char:
                break
            else:
                if out_char == CHARACTERS.index("END"):
                    correct += 1
    return correct, total


def process_word(lemma, form, tag, encoder, decoder):
    line = "\t".join([form, lemma, tag, ""])
    batch = [line]
    (in_char_tensors, in_mask), (out_char_tensors, out_mask), tag_tensors = lines_to_training_examples(batch)
    dev_loss, decoder_outputs = test_on_batch(in_char_tensors, in_mask, out_char_tensors, out_mask, tag_tensors, encoder, decoder)
    chars = [CHARACTERS[i] for i in decoder_outputs[0]]
    return chars


FILE_NAME = "sgjp-20210411.tab"

with open(FILE_NAME) as f:
  data = f.read()

_, text = data.split("</COPYRIGHT>\n")
lines = text.split("\n")[:-1]
random.shuffle(lines)


random.seed(42)
BATCH_SIZE = 32
NUM_CHARS = len(CHARACTERS)
EMBEDDING_DIM = 42#50
ENCODER_WIDTH = 100
DECODER_DIM = 100
TAG_DIM = len(ALL_FEATS)
TEACHER_FORCING_RATIO = 0.5

train_data = lines[:100000]
dev_data = lines[-10000:]

learning_rate=0.0002 # 0.0002
decoder_learning_ratio = 5.0
encoder = Encoder(NUM_CHARS, EMBEDDING_DIM, ENCODER_WIDTH)
decoder = Decoder(NUM_CHARS, EMBEDDING_DIM, ENCODER_WIDTH, TAG_DIM, DECODER_DIM)
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)


EPOCHS = 30
num_batches = len(train_data)//BATCH_SIZE
num_dev_batches = len(dev_data)//BATCH_SIZE
last_acc = 0

encoder = torch.load("encoder.mdl")
decoder = torch.load("decoder.mdl")

for epoch in range(EPOCHS):
  random.shuffle(train_data)
  print(f"epoch no {epoch+1}")
  epoch_loss = 0
  for n in tqdm(range(num_batches)):
    batch = train_data[n*BATCH_SIZE:(n+1)*BATCH_SIZE]
    (in_char_tensors, in_mask), (out_char_tensors, out_mask), tag_tensors = lines_to_training_examples(batch)
    loss = train_on_batch(in_char_tensors, in_mask, out_char_tensors, out_mask, tag_tensors, encoder, decoder,
                     encoder_optimizer, decoder_optimizer)
    epoch_loss += loss
  print(f"\ttrain loss: {epoch_loss:.2f}")
  correct, total = 0, 0
  for n in range(num_dev_batches):
    batch = dev_data[n*BATCH_SIZE:(n+1)*BATCH_SIZE]
    (in_char_tensors, in_mask), (out_char_tensors, out_mask), tag_tensors = lines_to_training_examples(batch)
    dev_loss, decoder_outputs = test_on_batch(in_char_tensors, in_mask, out_char_tensors, out_mask, tag_tensors, encoder, decoder)
    batch_correct, batch_total = accuracy(out_char_tensors, decoder_outputs)
    correct += batch_correct
    total += batch_total
  acc = correct/total * 100
  if acc > last_acc:
      last_acc = acc
      torch.save(encoder, "encoder.mdl")
      torch.save(decoder, "decoder.mdl")
  print(f"\tdev loss: {dev_loss:.2f}")
  print(f"{acc:.2f}%")



#process_word("muszkieterki", "muszkieterkami", "subst:pl:inst:f", encoder, decoder)

#torch.save(encoder, "encoder.mdl")
#torch.save(decoder, "decoder.mdl")

# podawanie charów z lematu po kroku/atencja
# EOS token
# czy permutacje czegoś nie psują? np. przy masce!
# obcinanie gradientów
# beam search
# enkoderowi też można podawać na każdym kroku tag
# funkcja aktywacji,
# dropout
# czy hidden to ostatni stan pomijając padding?
