import os
import subprocess
import wget

def prepare_data():
  if not os.path.exists("SEJF-1.1"):
    sejf_url = "http://zil.ipipan.waw.pl/SEJF?action=AttachFile&do=get&target=SEJF-1.1.tar.gz"
    wget.download(sejf_url)
    fname = "SEJF-1.1.tar.gz"
    subprocess.call(["tar", "-xvf", fname])
  dic_file = os.path.join("SEJF-1.1", "SEJF-1.1-dlcf.dic")
  with open(dic_file) as f:
    txt = f.read().strip()
  lines = txt.split("\n")
  data = []
  for line in lines:
    orth, rest = line.split(",", 1)
    base, tag = rest.split(":", 1)
    split_tag = tag.split(":", 1)
    if len(split_tag) == 1:
      # deleting phrases without morphological information
      continue
    pos, morph = split_tag
    morph = morph.replace("n1", "n")
    morph = morph.replace("n2", "n")
    morph = morph.replace("p1", "m1")
    morph = morph.replace("p2", "n")
    morph = morph.replace("p3", "n")
    if morph and " jak " not in orth:
      data.append((orth, base, pos, morph))
      # deleting comparative phrases, which usually inflect based on semantic criteria
  return data


