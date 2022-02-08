import json
import pandas
import numpy
import random
import click
import torch
from data_loader import DataLoader
from network import Encoder, Decoder 
from torch import optim
from tqdm import tqdm
from neuro_flexer import NeuroFlexer

BATCH_SIZE = 512
EMBEDDING_DIM = 42#50
ENCODER_WIDTH = 70#50#100
DECODER_DIM = 140#100
TEACHER_FORCING_RATIO = 0.5
LEARNING_RATE=0.0002
DECODER_LEARNING_RATIO = 5.0
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


def accuracy(data_loader, target, out):
    correct, total = 0, 0
    for out_seq, target_seq in zip(out.cpu(), target):
        total += 1
        for out_char, target_char in zip(out_seq, target_seq):
            if out_char != target_char:
                break
            else:
                if out_char == data_loader.characters.index("END"):
                    correct += 1
    return correct, total

def load_data(dictfile, flist, split=0.9):
    # Frequency list
    df = pandas.read_csv(flist, sep="\t", header=None)
    df.columns = ["lemma", "pos", "frequency"]
    total = sum(df["frequency"])
    min_freq = 15 
    fdict = {a:b for a,b in zip(df["lemma"]+"\t"+df["pos"], df["frequency"])}

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
    with open(dictfile) as f:
      text = f.read()
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


@click.command()
@click.option("--dry", is_flag=True)
@click.option("--attn", is_flag=True)
@click.option("--epochs", default=3)
@click.option("--lang", default="pl")
def main(**kwargs):
    epochs = kwargs["epochs"]
    dry_run = kwargs["dry"]
    use_attention = kwargs["attn"]
    lang = kwargs["lang"]

    if dry_run:
        num_train_batches = 200
        num_dev_batches = 100
        num_test_batches = 100

    else:
        if lang == "pl":
            num_train_batches = 10000
        elif lang == "ru":
            num_train_batches = 800
        num_dev_batches = 500
        num_test_batches = 500

    if lang == "pl":
        dictfile = "polish_flexer/pl_dict.tab"
        flist = "polish_flexer/pl_freq.tsv"
        morphology_file = "polish_flexer/pl_morph.json"

    elif lang == "ru":
        dictfile = "russian_flexer/ru_dict.tab"
        flist = "russian_flexer/ru_freq.tsv"
        morphology_file = "russian_flexer/ru_morph.json"



    lower_case = True
    with open(morphology_file) as f:
        morphology = json.load(f)
    data_loader = DataLoader(morphology, lower_case)
    encoder = Encoder(len(data_loader.characters), EMBEDDING_DIM, ENCODER_WIDTH)
    decoder = Decoder(len(data_loader.characters), EMBEDDING_DIM, len(data_loader.all_feats), ENCODER_WIDTH*2, DECODER_DIM, use_attention=use_attention)
    print(f"decoder shape: {decoder.recurrent.weight_ih_l0.shape} {decoder.recurrent.weight_hh_l0.shape}")
    neuro = NeuroFlexer(data_loader, encoder, decoder, device)

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=LEARNING_RATE)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=LEARNING_RATE * DECODER_LEARNING_RATIO)

    random.seed(42)
    lines, lemma_to_lines, train_keys, dev_keys, test_keys, train_freq_dist, \
    dev_freq_dist, test_freq_dist = load_data(dictfile, flist)

    last_acc = 0
    for epoch in range(epochs):
      print(f"epoch no {epoch+1}")
      print("TRAIN")
      epoch_loss = 0
      for n in tqdm(range(num_train_batches)):
        (in_char_tensors, in_mask), (out_char_tensors, out_mask), tag_tensors = data_loader.sample_batch(train_keys, train_freq_dist, lines, lemma_to_lines, BATCH_SIZE)
        loss = neuro.train_on_batch(in_char_tensors, in_mask, out_char_tensors, out_mask, tag_tensors, encoder_optimizer, decoder_optimizer, TEACHER_FORCING_RATIO)
        epoch_loss += loss
      print(f"\ttrain loss: {epoch_loss:.2f}")
      correct, total = 0, 0
      dev_epoch_loss = 0
      print("DEV")
      for n in tqdm(range(num_dev_batches)):
        (in_char_tensors, in_mask), (out_char_tensors, out_mask), tag_tensors = data_loader.sample_batch(dev_keys, dev_freq_dist, lines, lemma_to_lines, BATCH_SIZE)
        dev_loss, decoder_outputs = neuro.test_on_batch(in_char_tensors, in_mask, out_char_tensors, out_mask, tag_tensors)
        batch_correct, batch_total = accuracy(data_loader, out_char_tensors, decoder_outputs)
        correct += batch_correct
        total += batch_total
        dev_epoch_loss += dev_loss
      acc = correct/total * 100
      if acc > last_acc:
          last_acc = acc
          if not dry_run or True:
              torch.save(encoder.state_dict(), f"{lang}_encoder.mdl")
              torch.save(decoder.state_dict(), f"{lang}_decoder.mdl")
      print(f"\tdev loss: {dev_epoch_loss:.2f}")
      print(f"\t dev accuracy: {acc:.2f}%")

    if not dry_run or True:
        encoder.load_state_dict(torch.load(f"{lang}_encoder.mdl"))#
        decoder.load_state_dict(torch.load(f"{lang}_decoder.mdl"))

    correct, total = 0, 0
    test_epoch_loss = 0
    
    print("TEST", len(test_keys), len(test_freq_dist))
    for n in tqdm(range(num_test_batches)):
        (in_char_tensors, in_mask), (out_char_tensors, out_mask), tag_tensors = data_loader.sample_batch(test_keys, test_freq_dist, lines, lemma_to_lines, BATCH_SIZE)
        test_loss, decoder_outputs = neuro.test_on_batch(in_char_tensors, in_mask, out_char_tensors, out_mask, tag_tensors)
        batch_correct, batch_total = accuracy(data_loader, out_char_tensors, decoder_outputs)
        correct += batch_correct
        total += batch_total
        test_epoch_loss += test_loss
    acc = correct/total * 100
    print(f"test loss: {test_epoch_loss:.2f}")
    print(f"test accuracy: {acc:.2f}%")


if __name__ == "__main__":
    main()
