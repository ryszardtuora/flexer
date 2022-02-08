import json
import torch
from network import Decoder, Encoder
from neuro_flexer import NeuroFlexer
from data_loader import DataLoader

class Flexer(object):
  name = "flexer"
  def __init__(self, nlp, morphology_file, inflection_dict, encoder_model, decoder_model, inflection_mode):
    self.nlp = nlp # this object should return a list of Token objects
    with open(morphology_file) as f:
      data = json.load(f)
    self.attr2feats = data["ATTR2FEATS"]
    self.val2attr = data["VAL2ATTR"]
    self.governing_deprels = data["GOVERNING_DEPRELS"] # a list of deprels with inverted dependency structure (i.e. governing children)
    self.accomodation_rules = data["ACCOMODATION_RULES"] # A deprel -> agreement attrs dict
    if "ADMISSIBLE_COMBINATIONS" in data:
      self.admissible_combinations = data["ADMISSIBLE_COMBINATIONS"]
    else:
      self.admissible_combinations = None
    self.inflection_dict = inflection_dict
    self.inflexible_pos = data["INFLEXIBLE_POS"]
    self.disallowed_feats = data["DISALLOWED_FEATS"]

    if inflection_mode == "DICT":
        self.inflection_mode = "DICT"

    elif inflection_mode == "NEURO":
        self.inflection_mode = "NEURO"
        embedding_dim = 42
        encoder_width = 70
        tag_dim = len(data["VAL2ATTR"])
        decoder_dim = 140

        morphology = data
        data_loader = DataLoader(morphology, lower_case=True)
        num_chars = len(data_loader.characters)

        encoder = Encoder(num_chars, embedding_dim, encoder_width)
        encoder.load_state_dict(torch.load(encoder_model))
        decoder = Decoder(num_chars, embedding_dim, tag_dim, encoder_width*2, decoder_dim, use_attention=True)
        decoder.load_state_dict(torch.load(decoder_model))

        encoder.eval()
        decoder.eval()
        self.neuro_flexer = NeuroFlexer(data_loader, encoder, decoder)

  def neural_flex(self, lemma, tag, target_pattern):
    split_target_pattern = target_pattern.split(":")
    attr_to_val = {self.val2attr[val]:val for val in tag.split(":")}

    for val in target_pattern.split(":"):
        attr_to_val[self.val2attr[val]] = val
    full_tag = ":".join(attr_to_val.values())
    if self.admissible_combinations:
        split_full_tag = full_tag.split(":") 
        ranking = sorted(self.admissible_combinations, key=lambda a: self.tag_distance(a.split(":"), split_full_tag))
        full_tag = ranking[0]
    inflected = self.neuro_flexer.neural_process_word(lemma, full_tag)

    return inflected

  def tag_distance(self, taglist1, taglist2):
    distance = len(set(taglist1).symmetric_difference(set(taglist2)))
    return distance

  def dict_flex(self, lemma, current_tag, target_pattern):
    def gen_to_tag(gen):
      full_tag = gen["full_tag"]
      split_tag = full_tag.split(":")
      return split_tag

    def short_distance(taglist1, taglist2):
      distance = len(set(taglist1).difference(set(taglist2)))
      return distance

    split_current_tag = current_tag.split(":")
    split_target_pattern = target_pattern.split(":")
    generation = self.inflection_dict.generate(lemma)

    satisfactory = [g for g in generation if all([f in gen_to_tag(g) for f in split_target_pattern])]
    if not satisfactory:
      return None
    else:
      for entry in satisfactory:
          entry["score"] = self.tag_distance(split_current_tag, gen_to_tag(entry))
      srt = sorted(satisfactory, key=lambda g:g["score"])
      # we choose the form most levenshtein similar to our initial tag
      inflected = srt[0]["form"]

  
    return inflected

  def get_case_fun(self, token_string):
    if token_string.isupper():
      case_fun = lambda s: s.upper()
    elif token_string.islower():
      case_fun = lambda s: s.lower()
    elif token_string.istitle():
      case_fun = lambda s: s.capitalize()
    else:
      case_fun = lambda s: s
    return case_fun

  def flex_token(self, token, target_pattern):
    # pattern is a ":" separated list of desired attributes for the new word to take on
    # the new word will be selected from the options provided by the generator
    # as the levenshtein nearest pattern counting from the pressent token's features

    if target_pattern in ["", None]:
      return token.orth

    target_pattern = ":".join([f for f in target_pattern.split(":") if f not in self.disallowed_feats])

    token_string = token.orth
    case_fun = self.get_case_fun(token_string)
    lemma = token.lemma

    pos_tag = token.pos_tag
    if pos_tag in self.inflexible_pos:
        return lemma

    feats = token.feats
    full_tag = pos_tag
    if feats != "":
      cleaned_feats = [f for f in feats.split(":") if f not in self.disallowed_feats] 
      full_tag = ":".join([pos_tag] + cleaned_feats)

    if self.inflection_mode == "DICT":
      inflected = self.dict_flex(lemma, full_tag, target_pattern)
      if inflected is None:
        inflected = token.orth

    elif self.inflection_mode == "NEURO":
      inflected = self.neural_flex(lemma, full_tag, target_pattern)

    inflected = case_fun(inflected)
    return inflected

  def get_children(self, head, tokens):
    head_ind = head.ind
    children = [tok for tok in tokens if tok.head == head_ind]
    return children

  def get_subtree(self, head, tokens):
    subtree = set([head])
    children = self.get_children(head, tokens)
    for child in children:
        subtree.update(self.get_subtree(child, tokens))
    return subtree

  def flex_subtree(self, head, tokens, target_pattern):
    ind_to_inflected = {}
    children = self.get_children(head, tokens)
    children_to_inflect = [child for child in children if child.deprel not in self.governing_deprels]
    governing_children = [child for child in children if child.deprel in self.governing_deprels]

    if governing_children:
      inflected_head = head.orth + head.whitespace
      governor = governing_children[0]
      inflected_governor_subtree = self.flex_subtree(governor, tokens, target_pattern)
      ind_to_inflected.update(inflected_governor_subtree)

    else:
      inflected_head = self.flex_token(head, target_pattern) + head.whitespace

    ind_to_inflected[head.ind] = inflected_head
    for child in children_to_inflect:
      child_deprel = child.deprel
      if child_deprel in self.accomodation_rules:
        accomodable_attrs = self.accomodation_rules[child_deprel]
      else:
        accomodable_attrs = []
      # we're not limiting ourselves to target pattern, but rather propagate all the features of the new tag, which go through the rule
      head_feats = {self.val2attr[v]:v for v in head.feats.split(":") if v in self.val2attr}
      #print("head_feats", head_feats)
      target_feats = {self.val2attr[v]:v for v in target_pattern.split(":") if v in self.val2attr}
      #print("target_feats", target_feats)
      head_feats.update(target_feats)
      #print("target_profile", head_feats)
      target_child_feats = list(head_feats.values())
      feats = target_child_feats

      #
      #feats = [f for f in target_pattern.split(":") if f in self.val2attr] # limiting to supported features
      accomodable_feats = [f for f in feats if self.val2attr[f] in accomodable_attrs]
      child_pattern = ":".join(accomodable_feats)
      inflected_subtree = self.flex_subtree(child, tokens, child_pattern)
      ind_to_inflected.update(inflected_subtree)
    return ind_to_inflected

  def lemmatize_subtree(self, head, tokens):
    # The algorithm recurrently goes through each child and inflects it into the pattern
    # corresponding to the base form of the head of the phrase.
    # The algorithm is rule based.
    ind_to_lemmatized = {}
    children = self.get_children(head, tokens)
    children_to_lemmatize = [child for child in children if child.deprel not in self.governing_deprels]
    governing_children = [child for child in children if child.deprel in self.governing_deprels]
    if governing_children:
      governor = governing_children[0]
      lemmatized_governor_subtree = self.lemmatize_subtree(governor, tokens)
      ind_to_lemmatized.update(lemmatized_governor_subtree)
      ind_to_lemmatized[head.ind] = head.orth
      target_pattern = ""

    else:
      # BASIC:
      lemmatized_head = head.lemma + head.whitespace
      target_pattern = self.nlp(lemmatized_head)[0].feats

      #target_pattern = self.inflection_dict.get_feats_from_lemma(head.lemma, "")
      #print(target_pattern)
      #lemmatized_head, _, target_pattern = self.inflection_dict.lemmatize(head.orth) 
      ind_to_lemmatized[head.ind] = lemmatized_head + head.whitespace

    for child in children_to_lemmatize:
      child_deprel = child.deprel
      if child_deprel in self.accomodation_rules:
          accomodable_attrs = self.accomodation_rules[child_deprel]
          feats = [f for f in target_pattern.split(":") if f in self.val2attr] # limiting to supported features
          accomodable_feats = [f for f in feats if self.val2attr[f] in accomodable_attrs]
          child_pattern = ":".join(accomodable_feats)
          lemmatized_subtree = self.flex_subtree(child, tokens, child_pattern)# TODO shouldnt this take into account only the attributes which are to be inflected given a particular deprel?
      else:
          lemmatized_subtree = {tok.ind: tok.orth + tok.whitespace for tok in self.get_subtree(child, tokens)}
      ind_to_lemmatized.update(lemmatized_subtree)
    return ind_to_lemmatized

  def lemmatize_phrase(self, tokens, phrase_tokens):
    phrase_inds = [tok.ind for tok in phrase_tokens]
    ind_to_lemmatized = {}
    independent_subtrees = [tok for tok in phrase_tokens if tok.head not in phrase_inds]
    for independent_head in independent_subtrees:
      ind_to_lemmatized.update(self.lemmatize_subtree(independent_head, tokens))
    seq = sorted([(i,t) for i, t in ind_to_lemmatized.items()])
    phrase = "".join([t for i, t in seq]).strip()
    return phrase

  def trivial_lemmatize_phrase(self, tokens, phrase_tokens):
    # just concatenate the lemmas
    seq = [tok.lemma + tok.whitespace for tok in phrase_tokens]
    phrase = "".join([t for i, t in seq]).strip()
    return phrase

  def flex_phrase(self, tokens, phrase_tokens, target_pattern):
    phrase_inds = [tok.ind for tok in phrase_tokens]
    ind_to_inflected = {}
    independent_subtrees = [tok for tok in phrase_tokens if tok.head not in phrase_inds]
    for independent_head in independent_subtrees:
        ind_to_inflected.update(self.flex_subtree(independent_head, tokens, target_pattern))
    seq = sorted([(i, t) for i, t in ind_to_inflected.items()])
    phrase = "".join([t for i, t in seq]).strip()
    return phrase

  def trivial_flex_phrase(self, tokens, phrase_tokens, target_pattern):
    # just concatenate the inflected forms
    seq = [self.flex_token(token, target_pattern) for token in phrase_tokens]
    phrase = "".join([t for i, t in seq]).strip()
    return phrase

