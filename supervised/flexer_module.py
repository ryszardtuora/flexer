import distance
import json
import torch
from network import Decoder, Encoder

#FUNCTIONAL_TOKENS = ["PAD", "START", "END"]
#CHARACTERS = list('-./abcdefghijklmnopqrstuvwxyzóąćęłńśźż’') + FUNCTIONAL_TOKENS
#
#
#ALL_FEATS = []
#for attr, feats in ATTR2FEATS:
#    ALL_FEATS.extend(feats)
#
#NUM_CHARS = len(CHARACTERS)
#EMBEDDING_DIM = 42#50
#ENCODER_WIDTH = 70#50#100
#DECODER_DIM = 140#100
#TAG_DIM = len(ALL_FEATS)
#

def word_to_tensor(word):
    pad_token = CHARACTERS.index("PAD")
    char_indices = []
    word = word.lower()
    chars = [c for c in word if c in CHARACTERS]
    char_indices = [CHARACTERS.index(c) for c in chars]
    max_len = len(char_indices)
    pad_vector = [0 for _ in range(max_len)]
    words_tensor = torch.LongTensor([char_indices, pad_vector])
    mask_tensor = (words_tensor != pad_token).int()
    return words_tensor, mask_tensor

def tag_to_tensor(tag):
    tag_tensor = torch.zeros((2, len(ALL_FEATS)))
    feats = tag.split(":")
    indices = [ALL_FEATS.index(v) for v in feats]
    for i in indices:
        tag_tensor[0][i] = 1
    return tag_tensor

class Flexer(object):
  name = "flexer"
  def __init__(self, nlp, morphology_file, inflection_dict):
    self.nlp = nlp # this object should return a list of Token objects
    with open(morphology_file) as f:
      data = json.load(f)
    self.attr2feats = data["ATTR2FEATS"]
    self.val2attr = data["VAL2ATTR"]
    self.governing_deprels = data["GOVERNING_DEPRELS"] # a list of deprels with inverted dependency structure (i.e. governing children)
    self.accomodation_rules = data["ACCOMODATION_RULES"] # A deprel -> agreement attrs dict
    self.inflection_dict = inflection_dict



    ##self.inflection_encoder = Encoder(NUM_CHARS, EMBEDDING_DIM, ENCODER_WIDTH)
    #self.inflection_encoder.load_state_dict(torch.load("encoder.mdl"))
    #self.inflection_decoder = Decoder(NUM_CHARS, EMBEDDING_DIM, TAG_DIM, DECODER_DIM)
    #self.inflection_decoder.load_state_dict(torch.load("decoder.mdl"))
    #self.inflection_encoder.eval()
    #self.inflection_decoder.eval()
    self.inflection_mode = "DICT"

  def neural_process_word(self, in_char_tensor, in_mask, tag_tensor):
    with torch.no_grad():
      in_lens = in_mask.sum(axis=1)
      encoder_outputs, encoder_hidden = self.inflection_encoder(in_char_tensor, in_lens)
      prev_char = torch.LongTensor([CHARACTERS.index("START") for _ in range(2)])
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
          lemma_char = torch.LongTensor([CHARACTERS.index("END") for _ in range(2)])
        decoder_output, _, decoder_hidden, decoder_cell = self.inflection_decoder(
          prev_char, lemma_char, decoder_hidden, decoder_cell, tag_tensor
        )
        _, topi = decoder_output.topk(1)
        continuation = topi[0].item() != CHARACTERS.index("END")
        prev_char = torch.LongTensor([[topi[i][0] for i in range(2)]]).squeeze()

        top_indices.append(topi)
        t+=1
    decoder_outputs = torch.cat(top_indices, axis=1)
    return decoder_outputs

  def neural_flex(self, lemma, tag, target_pattern):
    split_target_pattern = target_pattern.split(":")
    char_tensor, in_mask = word_to_tensor(lemma)
    attr_to_val = {self.val2attr[val]:val for val in tag.split(":")}
    for val in target_pattern.split(":"):
        attr_to_val[self.val2attr[val]] = val
    if attr_to_val["pos"] in ["xxx", "interp", "qub"]:
        return lemma
    full_tag = ":".join(attr_to_val.values())
    tag_tensor = tag_to_tensor(full_tag)
    out = self.neural_process_word(char_tensor, in_mask, tag_tensor)
    word = out[0]
    chars = [CHARACTERS[i] for i in word if CHARACTERS[i]!="END"]
    inflected = "".join(chars)
    return inflected

  #def dict_flex(self, lemma, tag, target_pattern):
    #def gen_to_tag(gen):
      #return gen[2].split(":")

    ##split_tag = tag.split(":")
    #split_target_pattern = target_pattern.split(":")
    #generation = self.morf.generate(lemma)
    ## we select only those generated forms, which satisfy the required pattern
    #right = [g for g in generation if all([f in gen_to_tag(g) for f in split_target_pattern])]
    #if right == []:
      #return None
    #else:
      #srt = sorted(right, key = lambda g: distance.levenshtein(split_tag, gen_to_tag(g)))
      ## we choose the form most levenshtein similar to our initial tag
      #inflected = srt[0][0]
    #return inflected
  
  def dict_flex(self, lemma, current_tag, target_pattern):
    def gen_to_tag(gen):
      full_tag = gen["full_tag"]
      split_tag = full_tag.split(":")
      return split_tag

    split_current_tag = current_tag.split(":")
    split_target_pattern = target_pattern.split(":")
    generation = self.inflection_dict.generate(lemma)
    satisfactory = [g for g in generation if all([f in gen_to_tag(g) for f in split_target_pattern])]
    if not satisfactory:
      return None# TODO może tutaj zwracać jednak najbliższą levenshteinem do DOCELOWEGO, albo inną miarą dystansu do sumy?
    else:
        srt = sorted(satisfactory, key=lambda g:distance.levenshtein(split_current_tag, gen_to_tag(g)))
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

    token_string = token.orth
    case_fun = self.get_case_fun(token_string)
    lemma = token.lemma

    pos_tag = token.pos_tag
    feats = token.feats
    full_tag = pos_tag
    if feats != "":
      full_tag += ":" + feats

    #split_tag = full_tag.split(":")
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
      id_to_inflected.update(inflected_governor_subtree)

    else:
      inflected_head = self.flex_token(head, target_pattern) + head.whitespace

    ind_to_inflected[head.ind] = inflected_head
    for child in children_to_inflect:
      child_deprel = child.deprel
      if child_deprel in self.accomodation_rules:
        accomodable_attrs = self.accomodation_rules[child_deprel]
      else:
        accomodable_attrs = []
      feats = [f for f in target_pattern.split(":") if f in self.val2attr] # limiting to supported features
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
      lemmatized_governor_subtree = self.lemmatize_subtree(governor)
      ind_to_lemmatized.update(lemmatized_governor_subtree)
      ind_to_lemmatized[head.ind] = head.orth
      target_pattern = ""

    else:
      lemmatized_head = head.lemma + head.whitespace
      target_pattern = self.nlp(lemmatized_head)[0].feats
      ind_to_lemmatized[head.ind] = lemmatized_head

    for child in children_to_lemmatize:
      child_deprel = child.deprel
      if child_deprel in self.accomodation_rules:
        lemmatized_subtree = self.flex_subtree(child, tokens, target_pattern)# TODO shouldnt this take into account only the attributes which are to be inflected given a particular deprel?
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

  def flex_phrase(self, tokens, phrase_tokens, target_pattern):
    phrase_inds = [tok.ind for tok in phrase_tokens]
    ind_to_inflected = {}
    independent_subtrees = [tok for tok in phrase_tokens if tok.head not in phrase_inds]
    for independent_head in independent_subtrees:
        ind_to_inflected.update(self.flex_subtree(independent_head, tokens, target_pattern))
    seq = sorted([(i, t) for i, t in ind_to_inflected.items()])
    phrase = "".join([t for i, t in seq]).strip()
    return phrase



