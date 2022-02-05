import random
import json
from tqdm import tqdm
from collections import Counter
from pandas import DataFrame
from flexer_module import Flexer
from morf_wrapper import MorfWrapper
from combo_wrapper import ComboWrapper
from pseudo_morph import PseudoMorph

random.seed(42)

def load_data(mwe_file, inflection_dict=None):
    data = []
    with open(mwe_file) as f:
        lines = f.read()
    for line in lines.split("\n")[:-1]:
        form, lemma, full_tag = line.split("\t")
        split_tag = full_tag.split(":", 1)
        pos_tag = split_tag[0]
        if len(split_tag)>1:
            feats = split_tag[1]
        else:
            feats = ""
        base_words = lemma.split(" ")
        if inflection_dict is None or all([inflection_dict.analyse(bw) for bw in base_words]):
            data.append((form, lemma, pos_tag, feats))
    random.shuffle(data)
    return data

def evaluate_flexer(data):
  output = []
  for orth, base, pos, morph in tqdm(data):
    base_tokens = nlp(base)
    orth_tokens = nlp(orth)
    flexed = flexer.flex_phrase(base_tokens, base_tokens, morph).lower()
    row = {}
    row["base"] = base
    row["structure"] = ",".join([f"{t.pos_tag}:{t.feats}" for t in base_tokens])
    row["orth"] = orth
    row["target"] = morph
    row["length"] = len(base_tokens)
    row["flexed"] = flexed
    row["exact"] = flexed == orth.lower()
    row["permuted"] = set(flexed.split()) == set(orth.lower().split())
    output.append(row)
  return output

def evaluate_lemmatizer(data):
  output = []
  for orth, base, pos, morph in tqdm(data):
    base_tokens = nlp(base)
    orth_tokens = nlp(orth)
    # filtering deviant cases
    head_tok = [tok for tok in base_tokens if tok.deprel == "root"][0]
    # base plural
    if "pl" in head_tok.feats.split(":"):
        continue

    # lemmatization into an infinitive
    if head_tok.pos_tag in ["ppas", "pact", "pcon", "pant", "ger"]:
        continue



    lemmatized = flexer.lemmatize_phrase(orth_tokens, orth_tokens).lower()
    lemmatized_concat = " ".join([t.lemma for t in orth_tokens]).lower()
    row = {}
    row["base"] = base
    row["structure"] = ",".join([f"{t.pos_tag}:{t.feats}" for t in base_tokens])
    row["orth"] = orth
    row["length"] = len(base_tokens)
    row["lemmatized"] = lemmatized
    row["exact"] = lemmatized == base.lower()
    row["permuted"] = set(lemmatized.split()) == set(base.lower().split())
    # concatenation-of-lemmas baseline
    row["concat_exact"] = lemmatized_concat == base.lower()
    row["concat_permuted"] = set(lemmatized_concat.split()) == set(base.lower().split())
    output.append(row)
  return output

data = load_data("polish_flexer/pl_mwe.tab")
test_ex = data[:5000]
print(len(data))


morphology_file = "polish_flexer/pl_morph.json"
nlp = ComboWrapper("polish-herbert-large", use_translation=False)
inflection_dict = MorfWrapper()
#inflection_dict = PseudoMorph("polish_flexer/pl_dict.tab")

infl_mode = "DICT"
print(infl_mode)
flexer = Flexer(nlp, morphology_file, inflection_dict, "pl_encoder.mdl", "pl_decoder.mdl", infl_mode)
flex_output = evaluate_flexer(test_ex)
flex_df = DataFrame(flex_output)
flex_df.to_csv("pl_flex_out_dict.tsv", sep="\t")
print("Inflection: ", sum(flex_df["permuted"])/len(flex_df) * 100)

lem_output = evaluate_lemmatizer(test_ex)
lem_df = DataFrame(lem_output)
lem_df.to_csv("pl_lem_out_dict.tsv", sep="\t")
print("Lemmatization: ", sum(lem_df["permuted"])/len(lem_df) * 100)
print("Lemmatization baseline:", sum(lem_df["concat_permuted"])/len(lem_df) * 100)



infl_mode = "NEURO"
print(infl_mode)
flexer = Flexer(nlp, morphology_file, inflection_dict, "pl_encoder.mdl", "pl_decoder.mdl", infl_mode)

flex_output = evaluate_flexer(test_ex)
flex_df = DataFrame(flex_output)
flex_df.to_csv("pl_flex_out_neuro.tsv", sep="\t")
print("Inflection: ", sum(flex_df["permuted"])/len(flex_df) * 100)

lem_output = evaluate_lemmatizer(test_ex)
lem_df = DataFrame(lem_output)
lem_df.to_csv("pl_lem_out_neuro.tsv", sep="\t")
print("Lemmatization: ", sum(lem_df["permuted"])/len(lem_df) * 100)
print("Lemmatization baseline:", sum(lem_df["concat_permuted"])/len(lem_df) * 100)


