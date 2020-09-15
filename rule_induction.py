import pandas

df = pandas.read_csv("UD_Polish-PDB/acc.csv")

labels = df["Unnamed: 0"]
hrules = {l: [] for l in labels}
lrules = {l: [] for l in labels}
crules = {l: [] for l in labels}

for feature in [c for c in df.columns if c != "Unnamed: 0"]:
  for i, label in enumerate(labels):
    if df[feature][i] > 0.85:
      lrules[label].append(feature)
      if df[feature][i] > 0.95:
        hrules[label].append(feature)
        if df[feature][i] != 1:
          crules[label].append(feature)

for ruleset in [hrules, lrules, crules]:
  to_pop = []
  for label in ruleset:
    if ruleset[label] == []:
      to_pop.append(label)
  for label in to_pop:
    ruleset.pop(label)
  

"""
#crules
{
# 'nsubj': ['person'],
 'conj': ['case', 'collectivity', 'degree', 'negation'],
 'amod': ['number', 'gender', 'case'],
# 'obl': ['negation'],
 'nummod:gov': ['number'],
# 'fixed': ['number'],
 'acl': ['number', 'gender', 'case'],
 'nummod': ['case'], #['number', 'gender', 'case'],
# 'nmod:arg': ['collectivity'],
# 'det:numgov': ['gender'],
 'det': ['number', 'gender', 'case'], #['number', 'gender', 'case', 'degree'],
 'aux:pass': ['number'], #['number', 'gender'],
 'nsubj:pass': ['number'], #['number', 'gender', 'case'],
# 'xcomp': ['number'],
# 'det:nummod': ['gender'],
 'det:poss': ['number', 'gender', 'case'],
 'flat': ['number'],
 'aux:clitic': ['number'],
 'cop': ['number'],
 'aux': ['number', 'gender', 'aspect'],
 'amod:flat': ['number', 'gender', 'case']
}


#hrules
nsubj ['person', 'collectivity', 'negation']
nmod ['degree', 'negation', 'accomodability']
obj ['negation', 'fullstoppedness']
conj ['case', 'collectivity', 'degree', 'negation', 'accentibility', 'prepositionality']
obl:arg ['degree', 'negation']
amod ['number', 'gender', 'case', 'degree', 'fullstoppedness']
obl ['negation']
acl:relcl ['collectivity', 'degree', 'negation']
nummod:gov ['number', 'collectivity']
advcl ['negation', 'agglutination']
fixed ['number', 'aspect', 'collectivity', 'degree', 'vocality']
acl ['number', 'gender', 'case', 'negation']
nummod ['number', 'gender', 'case', 'collectivity']
nmod:arg ['collectivity']
det:numgov ['gender', 'collectivity']
obl:agent ['collectivity', 'negation']
iobj ['negation']
ccomp ['degree']
det ['number', 'gender', 'case', 'degree']
aux:pass ['number', 'gender']
nsubj:pass ['number', 'gender', 'case', 'negation']
ccomp:obj ['negation']
appos ['aspect', 'collectivity', 'negation']
xcomp ['number', 'negation']
det:nummod ['number', 'gender', 'case', 'collectivity']
det:poss ['number', 'gender', 'case', 'degree']
obl:cmpr ['collectivity', 'negation']
flat ['number', 'collectivity', 'fullstoppedness']
aux:clitic ['number']
cop ['number']
aux ['number', 'gender', 'aspect']
advcl:relcl ['person']
amod:flat ['number', 'gender', 'case', 'degree']
advcl:cmpr ['person', 'degree']
parataxis:insert ['collectivity', 'accomodability']
csubj ['number', 'person']
nmod:flat ['collectivity']
ccomp:cleft ['number', 'gender', 'collectivity']
parataxis:obj ['collectivity']
orphan ['degree', 'accentibility']
csubj:pass ['number', 'case', 'aspect']
nmod:pred ['number', 'aspect', 'negation']
list ['case']
root ['collectivity', 'degree', 'negation']
nummod:flat ['number', 'gender', 'case']
xcomp:cleft ['number']

"""
