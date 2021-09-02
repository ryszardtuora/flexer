import pandas

#df = pandas.read_csv("UD_Polish-PDB/acc.csv")
df = pandas.read_csv("agreement_stats.csv")

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
  

