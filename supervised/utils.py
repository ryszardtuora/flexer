import random
import torch
from itertools import product

FUNCTIONAL_TOKENS = ["PAD", "START", "END"]


class DataLoader():
    def __init__(self, morphology, lower_case):
        self.morphology = morphology
        self.lower_case = lower_case
        
        self.characters = morphology["CHARACTERS"]
        if self.lower_case:
            self.characters = morphology["LOWER_CHARACTERS"]
        else:
            self.characters = morphology["CHARACTERS"]
        self.characters.extend(FUNCTIONAL_TOKENS)
        self.all_feats = []
        for feat in self.morphology["VAL2ATTR"].keys():
            self.all_feats.append(feat)

    def words_to_tensor(self, words, append_start=None, append_end=None):
        characters = self.characters 
        pad_token = characters.index("PAD")
        char_indices = []
        for word in words:
            if self.lower_case:
                word = word.lower()
            chars = [c for c in word if c in characters]
            if append_end:
                chars.append("END")
            if append_start:
                chars.insert(0, "START")
            indices = [characters.index(c) for c in chars]
            char_indices.append(indices)
        max_len = max([len(ci) for ci in char_indices])
        for ci in char_indices:
            padding = [pad_token] * (max_len - len(ci))
            ci += padding
        words_tensor = torch.LongTensor(char_indices)
        mask_tensor = (words_tensor != pad_token).int()
        return words_tensor, mask_tensor

    def tags_to_tensors(self, tags):
        batch_size = len(tags)
        disallowed_feats = self.morphology["DISALLOWED_FEATS"]
        tag_tensor = torch.zeros((batch_size, len(self.all_feats)))
        for bi, tag in enumerate(tags):
            feats = tag.split(":")
            feat_lists = []
            for feat in feats:
                feat_vals = [fv for fv in feat.split(".") if fv not in disallowed_feats]
                if feat_vals:
                    feat_lists.append(feat_vals)
            feat_combinations = list(product(*feat_lists))
            representative = random.choice(feat_combinations)
            indices = [self.all_feats.index(v) for v in representative]
            for i in indices:
                tag_tensor[bi][i] = 1
        return tag_tensor

    def lines_to_training_examples(self, lines):
        forms, lemmas, tags = [], [], []
        for line in lines:
            form, lemma, tag, _ = line.split("\t", 3)
            lemma = lemma.split(":")[0]
            forms.append(form)
            lemmas.append(lemma)
            tags.append(tag)
        in_char_tensors = self.words_to_tensor(lemmas)
        out_char_tensors = self.words_to_tensor(forms, append_end=True)
        tag_tensors = self.tags_to_tensors(tags)
        return in_char_tensors, out_char_tensors, tag_tensors

    def sample_batch(self, keys, freq_dist, lines, lemma_to_lines, size):
        sampled_keys = random.choices(population=keys, weights=freq_dist, k=size)
        line_ids = [lemma_to_lines[k] for k in sampled_keys]
        ins, outs, tags = [], [] , []
        for line_list in line_ids:
            if len(line_list) > 1:
                ind1, ind2 = random.sample(line_list, 2)
            else:
                ind1, ind2 = 0, 0
            line1, line2 = lines[ind1], lines[ind2]
            in_form,  _ = line1.split("\t", 1)
            out_form, _ , out_tag = line2.split("\t")
            outs.append(out_form)
            ins.append(in_form)
            tags.append(out_tag)
        in_char_tensors = self.words_to_tensor(ins)
        out_char_tensors = self.words_to_tensor(outs, append_end=True)
        tag_tensors = self.tags_to_tensors(tags)
        return in_char_tensors, out_char_tensors, tag_tensors

