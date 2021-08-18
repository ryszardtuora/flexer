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


def group_by_lemmas(lines, split=0.9):
    dict = {}
    for i, l in enumerate(lines):
        lemma = l.split("\t")[1]
        if lemma in dict:
            dict[lemma].append(i)
        else:
            dict[lemma] = [i]


def group_by_lemmas(lines, split=0.9):
    dict = {}
    for i, l in enumerate(lines):
        lemma = l.split("\t")[1]
        if lemma in dict:
            dict[lemma].append(i)
        else:
            dict[lemma] = [i]
    keys = list(dict.keys())
    random.shuffle(keys)
    split_point = int(len(keys)*split)
    dev_split = (1-split)/2
    dev_split_point = int(len(keys)*(split+dev_split))
    train_keys = keys[:split_point]
    dev_keys = keys[split_point:dev_split_point]
    test_keys = keys[dev_split_point:]
    seg_lists = []
    for seg in [train_keys, dev_keys, test_keys]:
        seg_list = [None for _ in range(sum([len(dict[k]) for k in seg]))]
        ind = 0
        for key in seg:
            for i in dict[key]:
                seg_list[ind] = lines[i]
                ind += 1
        seg_lists.append(seg_list)
    train, dev, test = seg_lists
    return train, dev, test


def x():
    with open("")





numpy.random.choice(keys, size=2, replace=False, p=freq_dist)
