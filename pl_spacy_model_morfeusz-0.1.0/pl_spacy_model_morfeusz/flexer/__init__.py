import morfeusz2
import distance
from spacy.tokens.token import Token


class Flexer(object):
  name = "flexer"
  def __init__(self, nlp):
    self.nlp = nlp
    try:
      self.nlp.tokenizer.morf.generate("")
    except RuntimeError:
      # morfeusz does not have the generate dictionary loaded
      self.nlp.tokenizer.morf = morfeusz2.Morfeusz(expand_tags = True, whitespace = morfeusz2.KEEP_WHITESPACES, generate = True)
    self.morf = self.nlp.tokenizer.morf
    self.val2attr = {'pri': 'person', 'sec': 'person','ter': 'person',
                     'nom': 'case', 'acc': 'case', 'dat': 'case', 'gen': 'case', 'inst': 'case', 'loc': 'case', 'voc': 'case',
                     'f': 'gender', 'm1': 'gender', 'm2': 'gender', 'm3': 'gender', 'n': 'gender',
                     'aff': 'negation', 'neg': 'negation', 
                     'sg': 'number', 'pl': 'number',
                     'pos': 'degree', 'com': 'degree', 'sup': 'degree', 
                     'perf': 'aspect', 'imperf': 'aspect',
                     'npraep': 'prepositionality', 'praep': 'prepositionality', 
                     'congr': 'accomodability', 'rec': 'accomodability', 
                     'akc': 'accentibility', 'nakc': 'accentibility', 
                     'agl': 'agglutination', 'nagl': 'agglutination', 
                     'nwok': 'vocality', 'wok': 'vocality', 
                     'npun': 'fullstoppedness', 'pun': 'fullstoppedness', 
                     'col': 'collectivity', 'ncol': 'collectivity'
                    }

    self.accomodation_rules = {"amod": ["number", "case", "gender"],
                               "amod:flat": ['number', 'gender', 'case'],
                               "acl": ["number", "case", "gender"],
                               "aux": ['number', 'gender', 'aspect'],
                               "aux:clitic": ['number'],
                               "aux:pass": ['number'],
                               "cop": ['number'],
                               "det:numgov": ["gender"], 
                               "det": ["number", "case", "gender"], 
                               "det:nummod": ["number", "case", "gender"], 
                               "det:poss": ["number", "case", "gender"],
                               "conj": ['case', "number", 'collectivity', 'degree', 'negation'],
                               "nummod": ["case"],
                               "nummod:gov": ['number'], 
                               "nsubj:pass": ['number'],
                               "fixed": ["case"],
                               "flat": ['number'],

                              }

 
 
  



  def __call__(self, doc):
    # this component does nothing in __call__
    # its functionality is performed via the flex method
    return doc

  def flex(self, token, pattern):
    # token is a spacy token
    # pattern is a ":" separated list of desired attributes for the new word to take on
    # the new word will be selected from the options provided by the generator
    # as the levenshtein nearest pattern counting from the pressent token's features
    if pattern in ["", None]:
      return token.orth_

    if type(token) == str:
      token = self.nlp(token)[0]
    split_pattern = pattern.split(":")
    lemma = token.lemma_
    
    pos_tag = token.tag_
    feats = token._.feats
    tag = pos_tag
    if feats != "":
      tag += ":" + feats
      
    split_tag = tag.split(":")

    def gen_to_tag(gen):
      return gen[2].split(":")
    
    generation = self.morf.generate(lemma)
    right = [g for g in generation if all([f in gen_to_tag(g) for f in split_pattern])]
    # we select only those generated forms, which satisfy the required pattern
    
    if right == []:
      return token.orth_

    else:  
      srt = sorted(right, key = lambda g: distance.levenshtein(split_tag, gen_to_tag(g)))
      # we choose the form most levenshtein similar to our initial tag
      newform = srt[0][0]
      return newform


  def flex_subtree(self, token, pattern):
    id_to_inflected = {}
    inflected_token = self.flex(token, pattern) + token.whitespace_
    id_to_inflected[token.i] = inflected_token
    children = token.children 
    for child in children:
      child_deprel = child.dep_
      try:
        accomodable_attrs = self.accomodation_rules[child_deprel] 
      except KeyError:
        accomodable_attrs = []
      feats = [f for f in pattern.split(":") if f in self.val2attr] # limiting to supported features
      accomodable_feats = [f for f in feats if self.val2attr[f] in accomodable_attrs]
      child_pattern = ":".join(accomodable_feats)
      inflected_subtree = self.flex_subtree(child, child_pattern)
      id_to_inflected.update(inflected_subtree)
    return id_to_inflected

  def flex_mwe(self, token, pattern):
    if type(token) != Token:
      raise ValueError("This method requires passing a spacy.tokens.token.Token argument!")
    id_to_flexed = self.flex_subtree(token, pattern)
    seq = sorted([(i,t) for i, t in id_to_flexed.items()])
    phrase = "".join([t for i, t in seq])
    return phrase


