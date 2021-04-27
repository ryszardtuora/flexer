import pandas
import numpy
import random
from constants import CHARACTERS, ALL_FEATS
from utils import lines_to_training_examples, sample_batch
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
    prev_char = torch.LongTensor([CHARACTERS.index("START") for _ in range(batch_size)])

    decoder_hidden = encoder_hidden
    decoder_cell = torch.zeros(decoder_hidden.shape)

    out_char_tensors = out_char_tensors.permute([1,0])
    out_mask = out_mask.permute([1,0])
    in_char_tensors = in_char_tensors.permute([1,0])

    use_teacher_forcing = random.random() < TEACHER_FORCING_RATIO
    max_target_len = max(out_lens)
    if use_teacher_forcing:
        for t in range(max_target_len):
            if t < len(in_char_tensors):
                lemma_char = in_char_tensors[t]
            else:
                lemma_char = torch.LongTensor([CHARACTERS.index("END") for _ in range(batch_size)])
            decoder_output, _, decoder_hidden, decoder_cell = decoder(
                prev_char, lemma_char, decoder_hidden, decoder_cell, tag_tensors
            )
            prev_char = out_char_tensors[t]
            if t < len(in_char_tensors):
                lemma_char = in_char_tensors[t]
            else:
                lemma_char = torch.LongTensor([CHARACTERS.index("END") for _ in range(batch_size)])
            mask_loss = maskNLLLoss(decoder_output, out_char_tensors[t], out_mask[t].bool())
            loss += mask_loss
    else:
        for t in range(max_target_len):
            if t < len(in_char_tensors):
                lemma_char = in_char_tensors[t]
            else:
                lemma_char = torch.LongTensor([CHARACTERS.index("END") for _ in range(batch_size)])
            decoder_output, _, decoder_hidden, decoder_cell = decoder(
                prev_char, lemma_char, decoder_hidden, decoder_cell, tag_tensors
            )
            _, topi = decoder_output.topk(1)
            prev_char = torch.LongTensor([[topi[i][0] for i in range(batch_size)]]).squeeze()
            if t < len(in_char_tensors):
                lemma_char = in_char_tensors[t]
            else:
                lemma_char = torch.LongTensor([CHARACTERS.index("END") for _ in range(batch_size)])
            mask_loss = maskNLLLoss(decoder_output, out_char_tensors[t], out_mask[t].bool())
            loss += mask_loss
    loss.backward()

    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1)

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
        prev_char = torch.LongTensor([CHARACTERS.index("START") for _ in range(batch_size)])
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
                lemma_char = torch.LongTensor([CHARACTERS.index("END") for _ in range(batch_size)])
            decoder_output, _, decoder_hidden, decoder_cell = decoder(
                prev_char, lemma_char, decoder_hidden, decoder_cell, tag_tensors
            )
            _, topi = decoder_output.topk(1)
            prev_char = torch.LongTensor([[topi[i][0] for i in range(batch_size)]]).squeeze()

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

def process_words(lemmas, forms, tags, encoder, decoder):
    lines = []
    for lemma, form, tag in zip(lemmas, forms, tags):
        line = "\t".join([form, lemma, tag, ""])
        lines.append(line)
    (in_char_tensors, in_mask), (out_char_tensors, out_mask), tag_tensors = lines_to_training_examples(lines)
    dev_loss, decoder_outputs = test_on_batch(in_char_tensors, in_mask, out_char_tensors, out_mask, tag_tensors, encoder, decoder)
    out_words = []
    for word in decoder_outputs:
        chars = [CHARACTERS[i] for i in word]
        out = "".join(chars)
        out_words.append(out)
    return out_words

def load_data(split=0.9):
    # Frequency list
    df = pandas.read_csv("flist.tsv", sep="\t", header=0)
    total = sum(df["Frekwencja"])
    min_freq = 20
    fdict = {a:b for a,b in zip(df["Lemat"]+"\t"+df["Część mowy"], df["Frekwencja"])}
    fdict

    #simplifying fdict
    simplified_fdict = {}
    for key, val in fdict.items():
        if pandas.isna(key):
            continue
        simplified_key = key.split("\t")[0]
        if simplified_key in simplified_fdict:
            simplified_fdict[simplified_key] += val
        else:
            simplified_fdict[simplified_key] = val

    # Dictionary data
    FILE_NAME = "sgjp-20210411.tab"
    with open(FILE_NAME) as f:
      data = f.read()
    _, text = data.split("</COPYRIGHT>\n")
    lines = text.split("\n")[:-1]

    lemma_to_lines = {}
    for i, l in enumerate(lines):
        _, lemma, _ = l.split("\t", 2)
        if lemma in lemma_to_lines:
            lemma_to_lines[lemma].append(i)
        else:
            lemma_to_lines[lemma] = [i]
    keys = list(lemma_to_lines.keys())
    random.shuffle(keys)

    new_fdict = {}
    for key in keys:
        freq_key = key.split(":")[0]
        if freq_key in simplified_fdict:
            new_fdict[key] = simplified_fdict[freq_key]
        else:
            new_fdict[key] = min_freq

    split_point = int(len(keys)*split)
    dev_split = (1-split)/2
    dev_split_point = int(len(keys)*(split+dev_split))
    train_keys = keys[:split_point]
    dev_keys = keys[split_point:dev_split_point]
    test_keys = keys[dev_split_point:]

    train_freq_dist = [new_fdict[k] for k in train_keys]
    dev_freq_dist = [new_fdict[k] for k in dev_keys]
    test_freq_dist = [new_fdict[k] for k in test_keys]

    return lines, lemma_to_lines, train_keys, dev_keys, test_keys, \
           train_freq_dist, dev_freq_dist, test_freq_dist

BATCH_SIZE = 512
NUM_CHARS = len(CHARACTERS)
EMBEDDING_DIM = 42#50
ENCODER_WIDTH = 70#50#100
DECODER_DIM = 140#100
TAG_DIM = len(ALL_FEATS)
TEACHER_FORCING_RATIO = 0.5

learning_rate=0.0002
decoder_learning_ratio = 5.0

if __name__ == "__main__":
    random.seed(42)
    lines, lemma_to_lines, train_keys, dev_keys, test_keys, train_freq_dist, \
    dev_freq_dist, test_freq_dist = load_data()

    encoder = Encoder(NUM_CHARS, EMBEDDING_DIM, ENCODER_WIDTH)
    decoder = Decoder(NUM_CHARS, EMBEDDING_DIM, TAG_DIM, DECODER_DIM)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)


    EPOCHS = 5
    num_train_batches = 5000
    num_dev_batches = 500
    last_acc = 0


    for epoch in range(EPOCHS):
      print(f"epoch no {epoch+1}")
      epoch_loss = 0
      for n in tqdm(range(num_train_batches)):
        (in_char_tensors, in_mask), (out_char_tensors, out_mask), tag_tensors = sample_batch(train_keys, train_freq_dist, lines, lemma_to_lines, BATCH_SIZE)
        loss = train_on_batch(in_char_tensors, in_mask, out_char_tensors, out_mask, tag_tensors, encoder, decoder,
                         encoder_optimizer, decoder_optimizer)
        epoch_loss += loss
      print(f"\ttrain loss: {epoch_loss:.2f}")
      correct, total = 0, 0
      dev_epoch_loss = 0
      for n in range(num_dev_batches):
        (in_char_tensors, in_mask), (out_char_tensors, out_mask), tag_tensors = sample_batch(dev_keys, dev_freq_dist, lines, lemma_to_lines, BATCH_SIZE)
        dev_loss, decoder_outputs = test_on_batch(in_char_tensors, in_mask, out_char_tensors, out_mask, tag_tensors, encoder, decoder)
        batch_correct, batch_total = accuracy(out_char_tensors, decoder_outputs)
        correct += batch_correct
        total += batch_total
        dev_epoch_loss += dev_loss
      acc = correct/total * 100
      if acc > last_acc:
          last_acc = acc
          torch.save(encoder.state_dict(), "encoder.mdl")
          torch.save(decoder.state_dict(), "decoder.mdl")
      print(f"\tdev loss: {dev_epoch_loss:.2f}")
      print(f"\t dev accuracy: {acc:.2f}%")

    encoder = torch.load("encoder.mdl")#
    decoder = torch.load("decoder.mdl")

    correct, total = 0, 0
    test_epoch_loss = 0
    num_test_batches = 500
    for n in range(num_test_batches):
        (in_char_tensors, in_mask), (out_char_tensors, out_mask), tag_tensors = sample_batch(test_keys, test_freq_dist, lines, lemma_to_lines, BATCH_SIZE)
        test_loss, decoder_outputs = test_on_batch(in_char_tensors, in_mask, out_char_tensors, out_mask, tag_tensors, encoder, decoder)
        batch_correct, batch_total = accuracy(out_char_tensors, decoder_outputs)
        correct += batch_correct
        total += batch_total
        test_epoch_loss += test_loss
    acc = correct/total * 100
    print(f"test loss: {test_epoch_loss:.2f}")
    print(f"test accuracy: {acc:.2f}%")


    #torch.save(encoder, "encoder.mdl")
    #torch.save(decoder, "decoder.mdl")
    """
    # zlewanie rozpodobniaczy?
    # xxx?
    # podawanie lemma charów jest złe, poprawiłem tutaj, ale nie w innych miejscach, dwa razy był podawany pierwszy znak
    # podzielić zbiór po lematach, żeby lematy nie powtarzały się w dev secie?
    # supervised slot filler (cloze task SIGMORPH 2018)
    # sprawdzić jak się dodają prefiksy
    # atencja z fixed length (dłuższych słów niż 40 tokenów się raczej nie spodziewamy, a nawet jeśli, to można coś uciąć biorąc pod uwagę przewagę sufiksów)
    # EOS token
    # czy permutacje czegoś nie psują? np. przy masce!
    # obcinanie gradientów
    # beam search
    # enkoderowi też można podawać na każdym kroku tag
    # funkcja aktywacji,
    # jednak duże litery mogą się przydać przy akronimach
    # dropout
    # czy hidden to ostatni stan pomijając padding?
    # w paperze nie nadpisują informacji z enkodera (jest zawsze konkatenowanaa), ale to nie jest standard jak sami przyznają
    # lematyzację można przecież zrobić dodając przykłady odwrotne!, wtedy reinfleksja też powinna być możliwa
    # Unimorph
    # statystyki infiksacji/sufiksacji itd
    # podpróbkowanie frewkencyjne

    # Avenues:
    # halucynacje i słowotwórstwo
    # PDB vs UD
    # Inne języki: czeski/rosyjski
    # odmiana lematów z rozpodobniaczami
    #
    Problem: polegamy zbyt mocno na lematyzatorze

1 opcja: nauczyć lematyzować (dawać odwrotne zadanie, ew. z dodatkowym wymiarem dla lematyzacji)
2 opcja: reinfleksja (losować jedną formę z lematu, i przechodzić do drugiej)

    """
