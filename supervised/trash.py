def word_to_tensor(word):
  word_length = len(word)
  char_tensor = torch.LongTensor((word_length))
  chars = [c for c in word if c in CHARACTERS] # filtering out rare characters
  for i, char in enumerate(chars):
    char_index = CHARACTERS.index(char)
    char_tensor[i] = char_index
  return char_tensor

def tag_to_tensors(tag):
  feats = tag.split(":")
  feat_lists = []
  for feat in feats:
    feat_vals = [fv for fv in feat.split(".") if fv not in INVALID_TAGS]
    if feat_vals:
      feat_lists.append(feat_vals)
  feat_combinations = product(*feat_lists)
  tensors = []
  for fc in feat_combinations:
    indices = [ALL_FEATS.index(v) for v in fc]
    tensor = torch.zeros((len(ALL_FEATS)))
    for i in indices:
      tensor[i] = 1
    tensors.append(tensor)
  return tensors

def line_to_training_examples(line):
  form, lemma, tag, _ = line.split("\t", 3)
  lemma = lemma.split(":")[0]
  in_char_tensor = word_to_tensor(lemma)
  out_char_tensor = word_to_tensor(form)
  tag_tensors = tag_to_tensors(tag)
  return in_char_tensor, out_char_tensor, tag_tensors
