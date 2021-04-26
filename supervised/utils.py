import random
import torch
from itertools import product
from .constants import CHARACTERS, INVALID_TAGS, ALL_FEATS
from .settings import LOWER_CASE



def words_to_tensor(words, append_start=None, append_end=None):
    pad_token = CHARACTERS.index("PAD")
    char_indices = []
    for word in words:
        if LOWER_CASE:
            word = word.lower()
        chars = [c for c in word if c in CHARACTERS]
        if append_end:
            chars.append("END")
        if append_start:
            chars.insert(0, "START")
        indices = [CHARACTERS.index(c) for c in chars]
        char_indices.append(indices)
    max_len = max([len(ci) for ci in char_indices])
    for ci in char_indices:
        padding = [pad_token] * (max_len - len(ci))
        ci += padding
    words_tensor = torch.LongTensor(char_indices)
    mask_tensor = (words_tensor != pad_token).int()
    return words_tensor, mask_tensor

def tags_to_tensors(tags):
    batch_size = len(tags)
    tag_tensor = torch.zeros((batch_size, len(ALL_FEATS)))
    for bi, tag in enumerate(tags):
        feats = tag.split(":")
        feat_lists = []
        for feat in feats:
            feat_vals = [fv for fv in feat.split(".") if fv not in INVALID_TAGS]
            if feat_vals:
                feat_lists.append(feat_vals)
        feat_combinations = list(product(*feat_lists))
        representative = random.choice(feat_combinations)
        indices = [ALL_FEATS.index(v) for v in representative]
        for i in indices:
            tag_tensor[bi][i] = 1
    return tag_tensor

def lines_to_training_examples(lines):
    forms, lemmas, tags = [], [], []
    for line in lines:
        form, lemma, tag, _ = line.split("\t", 3)
        lemma = lemma.split(":")[0]
        forms.append(form)
        lemmas.append(lemma)
        tags.append(tag)
    in_char_tensors = words_to_tensor(lemmas)
    out_char_tensors = words_to_tensor(forms, append_end=True)
    tag_tensors = tags_to_tensors(tags)
    return in_char_tensors, out_char_tensors, tag_tensors
