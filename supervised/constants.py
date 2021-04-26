
LOWER_CASE = True
FUNCTIONAL_TOKENS = ["PAD", "START", "END"]
if LOWER_CASE:
    CHARACTERS = list('-./abcdefghijklmnopqrstuvwxyzóąćęłńśźż’') + FUNCTIONAL_TOKENS
else:
    CHARACTERS = list('-./ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzóąĆćęŁłńŚśŹźŻż’') + FUNCTIONAL_TOKENS

INVALID_TAGS = ["pt"]

ATTR2FEATS = [
    ("pos", ("pacta", "pcon", "subst", "fin", "impt", "imps", "ger", "adja", "adv", "ppron3", "conj", "ppron12", "part", "pant", "pred", "prep", "adjp", "num", "inf", "frag", "pact", "cond", "bedzie", "depr", "brev", "comp", "interj", "aglt", "numcomp", "adjc", "winien", "praet", "ppas", "adj")),
    ("person", ("pri", "sec", "ter")),
    ("case", ("nom", "acc", "dat", "gen", "inst", "loc", "voc")),
    ("gender", ("f", "m1", "m2", "m3", "n")),
    ("negation", ("aff", "neg")),
    ("number", ("sg", "pl")),
    ("degree", ("pos", "com", "sup")),
    ("aspect", ("perf", "imperf")),
    ("prepositionality", ("npraep", "praep")),
    ("accomodability", ("congr", "rec")),
    ("accentibility", ("akc", "nakc")),
    ("agglutination", ("agl", "nagl")),
    ("vocality", ("nwok", "wok")),
    ("fullstoppedness", ("npun", "pun")),
    ("collectivity", ("col", "ncol")),
]

ALL_FEATS = []
for attr, feats in ATTR2FEATS:
  ALL_FEATS.extend(feats)
