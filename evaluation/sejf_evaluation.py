import os
import subprocess
import wget
import random
import spacy
from tqdm import tqdm
from collections import Counter
from pandas import DataFrame

random.seed(42)
nlp = spacy.load("pl_spacy_model_morfeusz")
flexer = nlp.get_pipe("flexer")

import json
with open("../rules_induced.json") as f:
  rules = json.load(f)
flexer.accomodation_rules = rules


def prepare_data():
  sejf_url = "http://zil.ipipan.waw.pl/SEJF?action=AttachFile&do=get&target=SEJF-1.1.tar.gz"
  wget.download(sejf_url)
  fname = "SEJF-1.1.tar.gz"
  subprocess.call(["tar", "-xvf", fname])
  dic_file = os.path.join("SEJF-1.1", "SEJF-1.1-dlcf.dic")
  with open(dic_file) as f:
    txt = f.read().strip()
  lines = txt.split("\n")
  data = []
  for line in lines:
    orth, rest = line.split(",", 1)
    base, tag = rest.split(":", 1)
    split_tag = tag.split(":", 1)
    if len(split_tag) == 1:
      # deleting phrases without morphological information
      continue
    pos, morph = split_tag
    morph = morph.replace("n1", "n")
    morph = morph.replace("n2", "n")
    morph = morph.replace("p1", "m1")
    morph = morph.replace("p2", "n")
    morph = morph.replace("p3", "n")
    if morph and " jak " not in orth:
      data.append((orth, base, pos, morph))
      # deleting comparative phrases, which usually inflect based on semantic criteria
  return data

def evaluate_flexer(data):
  output = []
  for orth, base, pos, morph in tqdm(data):
    base_doc = nlp(base)
    base_head = [t for t in base_doc if t.dep_ == "ROOT"][0]
    orth_doc = nlp(orth)
    orth_head = [t for t in orth_doc if t.dep_ == "ROOT"][0]
    flexed = flexer.flex_mwe(base_head, morph).lower()
    row = {}
    row["base"] = base
    row["structure"] = ",".join([t.tag_ for t in base_doc])
    row["orth"] = orth
    row["target"] = morph
    row["length"] = len(base_doc)
    row["flexed"] = flexed
    row["exact"] = flexed == orth.lower()
    row["permuted"] = set(flexed.split()) == set(orth.lower().split())
    output.append(row)
  return output

def evaluate_lemmatizer(data):
  output = []
  for orth, base, pos, morph in tqdm(data):
    base_doc = nlp(base)
    base_head = [t for t in base_doc if t.dep_ == "ROOT"][0]
    orth_doc = nlp(orth)
    orth_head = [t for t in orth_doc if t.dep_ == "ROOT"][0]
    lemmatized = flexer.lemmatize_mwe(orth_head).lower()
    lemmatized_concat = " ".join([t.lemma_ for t in orth_doc]).lower()
    row = {}
    row["base"] = base
    row["structure"] = ",".join([t.tag_ for t in base_doc])
    row["orth"] = orth
    row["length"] = len(base_doc)
    row["lemmatized"] = lemmatized
    row["exact"] = lemmatized == base.lower()
    row["permuted"] = set(lemmatized.split()) == set(base.lower().split())
    # concatenation-of-lemmas baseline
    row["concat_exact"] = lemmatized_concat == base.lower()
    row["concat_permuted"] = set(lemmatized_concat.split()) == set(base.lower().split())
    output.append(row)
  return output

data = prepare_data()
random.shuffle(data)
test_ex = data[:5000]

flex_output = evaluate_flexer(test_ex)
lem_output = evaluate_lemmatizer(test_ex)

flex_df = DataFrame(flex_output)
#flex_df.to_csv("SEJF_flex_out.tsv", sep="\t")
flex_df.to_csv("SEJF_flex_out_induced.tsv", sep="\t")
lem_df = DataFrame(lem_output)
#lem_df.to_csv("SEJF_lem_out.tsv", sep="\t")
lem_df.to_csv("SEJF_lem_out_induced.tsv", sep="\t")
