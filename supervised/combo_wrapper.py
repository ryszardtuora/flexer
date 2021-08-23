from combo.predict import COMBO
from translator import Translator

class Token():
    def __init__(self, ind, orth, lemma, pos_tag, feats, head, deprel, whitespace, span):
        self.ind = ind
        self.orth = orth
        self.lemma = lemma
        self.pos_tag = pos_tag
        self.feats = feats
        self.head = head
        self.deprel = deprel
        self.whitespace = whitespace
        self.span = span

    def __repr__(self):
        representation = f"{self.ind}. {self.orth}"
        return representation

class ComboWrapper():
    def __init__(self, model, use_translation):
        self.nlp = COMBO.from_pretrained(model)
        self.model_name = model
        self.translator = Translator()
        self.use_translation = use_translation

    def extract_unimorph(self, sentence):
        conllu = self.nlp.dump_line(sentence)
        lines = conllu.split("\n")[:-2]
        translated_lines = [self.translator.translate(line) for line in lines]
        return translated_lines

    def process(self, text):
        sentence = self.nlp(text)
        translated_morph =  self.extract_unimorph(sentence)
        tokens = sentence.tokens
        tok_objs = []
        end_ind = 0
        for ind, tok in enumerate(tokens):
            orth = tok.token
            lemma = tok.lemma
            pos_tag, feats = self.read_morphology(tok)
            if self.use_translation:
                pos_tag, feats = translated_morph[ind]
            head = tok.head-1
            deprel = tok.deprel
            start_ind = text[end_ind:].find(orth) + end_ind
            end_ind = start_ind + len(orth) - 1
            subsequent_ind = end_ind + 1
            span = (start_ind, subsequent_ind)
            if subsequent_ind < len(text) and text[subsequent_ind].isspace():
                whitespace = " " 
            else:
                whitespace = ""
            tok_obj = Token(ind, orth, lemma, pos_tag, feats, head, deprel, whitespace, span)
            tok_objs.append(tok_obj)
        return tok_objs

    def read_morphology(self, token):
        if self.model_name.startswith("polish"):
            # handling different ways of encoding morphology 
            split_tag = token.xpostag.split(":", 1)
            pos_tag = split_tag[0]
            if len(split_tag) > 1:
                feats = split_tag[1]
            else:
                feats = ""

        else:
            pos_tag = token.upostag
            feats = token.feats
        return pos_tag, feats

    def __call__(self, text):
        return self.process(text)

