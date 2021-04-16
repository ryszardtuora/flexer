import spacy
import pandas

nlp = spacy.load("pl_spacy_model_morfeusz")
                        
data = pandas.read_csv("kpwr-1.1-fixed-keywords-test.tsv", sep="\t", header=None)
#data = pandas.read_csv("kpwr-1.2-ne-test-fix.tsv", sep="\t", header=None)
lemma = data[0]
orth = data[1]

flexer = nlp.get_pipe("flexer")

correct = 0
total = 0
correct_case = 0
correct_nb = 0
correct_nb_retained = 0
for l, o in zip(lemma, orth):
  doc = nlp(o.lower())
  head = [t for t in doc if t.dep_ == "ROOT"][0]
  if "pl" in head._.feats.split(":"):
    nb = "pl"
  else:
    nb = "sg"
  lemmatized = flexer.lemmatize_mwe(head)
  doc_lemmatized = nlp(lemmatized)
  lemmatized_head = [t for t in doc_lemmatized if t.dep_ == "ROOT"][0]
  if l.lower() == flexer.flex_mwe(lemmatized_head, nb).lower():
    correct_nb_retained += 1
  if l.lower() in [lemmatized.lower(), flexer.flex_mwe(lemmatized_head, "pl").lower()]:
    correct_nb += 1
  if lemmatized.lower() == l.lower():
    correct += 1
  else:
    print("gold:{} source:{} system:{}".format(l, o, lemmatized))
  if lemmatized == l:
    correct_case +=1
  total += 1

print(correct/total)
print(correct_case/total)
print(correct_nb/total)
print(correct_nb_retained/total)

