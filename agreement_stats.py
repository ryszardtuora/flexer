import os
import pandas
import json
import conllu


attr2vals = {
  "person": ['ter', 'sec', 'pri'],
  "case": ['nom', 'acc','dat', 'gen', 'inst', 'loc', 'voc'],
  "gender": ['f', 'm1', 'm2', 'm3', 'n'],
  "negation": ['aff', 'neg'],
  "number": ['sg', 'pl'],
  "degree": ['pos', 'com', 'sup'],
  "aspect": ['perf', 'imperf'],
  "prepositionality": ['npraep', 'praep'],
  "accomodability": ['congr', 'rec'],
  "accentibility": ['akc', 'nakc'],
  "agglutination": ['agl', 'nagl'],
  "vocality": ['nwok', 'wok'],
  "fullstoppedness": ['npun', 'pun'],
  "collectivity": ['col', 'ncol'],
}

val2attr = {}
for attr in attr2vals:
  for val in attr2vals[attr]:
    val2attr[val]=attr

agreement_table = {}
cooccurence_table = {}


docs = []
PDB_FOLDER = "UD_Polish-PDB-master"
conllufiles = [f for f in os.listdir(PDB_FOLDER) if f.endswith(".conllu")]
for conllufile in conllufiles:
  conllupath = os.path.join(PDB_FOLDER, conllufile)
  with open(conllupath) as f:
    txt = f.read()
  docs.extend(conllu.parse(txt))

for doc in docs:
  # clearing multitoken annotations
  to_clear = []
  for tok in doc:
    if type(tok["id"]) != int:
      to_clear.append(tok)
  for tok in to_clear:
    doc.remove(tok)
  # calculating the child-head agreement statistics
  tags_list = [tok["xpostag"] for tok in doc]
  heads = [tok["head"]-1 for tok in doc]
  for tok in doc:
    deprel = tok["deprel"]
    if deprel == "root": # root has no head to agree with
      continue
    position = tok["id"]-1
    feats = tags_list[position].split(":")[1:]
    head_position = heads[position]
    head_feats = tags_list[head_position].split(":")[1:]
    for feat in feats:
      if feat in ["pt", "nol"]: # plurale tantum and nol are not supported by morfeusz
        continue
      attr = val2attr[feat] # translating values into attributes for which they are defined
      if feat in head_feats:
        if deprel in agreement_table:
          if attr in agreement_table[deprel]:
            agreement_table[deprel][attr] += 1
          else:
            agreement_table[deprel][attr] = 1
        else:
            agreement_table[deprel] = {attr:1}
      possible_vals = attr2vals[attr]
      if any([hf in possible_vals for hf in head_feats]):# we take into account only those pairs, in which both elements do have a value for a given attribute
        if deprel in cooccurence_table:
          if attr in cooccurence_table[deprel]:
            cooccurence_table[deprel][attr] += 1
          else:
            cooccurence_table[deprel][attr] = 1
        else:
          cooccurence_table[deprel] = {attr:1}


data_table = {
                deprel: {
                          attribute: agreement_table[deprel][attribute]/cooccurence_table[deprel][attribute] 
                          for attribute in agreement_table[deprel]
                        }
                for deprel in agreement_table
             }

df = pandas.DataFrame(data_table).T.sort_index()
df.to_csv("agreement_stats.csv")
THRESHOLD = 0.95
selected = df[df>THRESHOLD].T.to_dict()
rules = {deprel:[] for deprel in selected}
for deprel in selected:
  for attr in selected[deprel]:
    if selected[deprel][attr] > THRESHOLD:
      rules[deprel].append(attr)

rules = {k:v for k,v in rules.items() if v != []}
with open("rules_induced.json", "w") as f:
  json.dump(rules, f, indent=2)



