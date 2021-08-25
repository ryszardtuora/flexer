import morfeusz2

class MorfWrapper():
    def __init__(self):
        self.morf = morfeusz2.Morfeusz(expand_tags = True, whitespace = morfeusz2.KEEP_WHITESPACES, generate = True)

    def generate(self, lemma):
        try:
            generated = self.morf.generate(lemma)
            processed = [{"form": g[0], "full_tag": g[2]} for g in generated]
        except RuntimeError:
            processed = [{"form": lemma, "full_tag": "X"}]
        return processed

    def lemmatize(self, form):
        analyses = self.morf.analyse(form)
        chosen_analysis = analyses[0]
        morpho = chosen_analysis[2]
        lemma = morpho[1].split(":", 1)[0]

        # recover the feats from the lemma
        lemma_analyses = [a for a in self.morf.analyse(lemma) if a[2][1].split(":", 1)[0] == lemma]
        chosen_lemma_analysis = lemma_analyses[0]
        lemma_morpho = chosen_lemma_analysis[2]
        tag = lemma_morpho[2]
        split_tag = tag.split(":", 1)
        pos_tag = split_tag[0]
        if len(split_tag) == 1:
            feats = ""
        else:
            feats = split_tag[1]
        return lemma, pos_tag, feats


    def get_feats_from_lemma(self, lemma, default_feats=""):
        try:
            lemma_analyses = [a for a in self.morf.analyse(lemma) if a[2][1].split(":", 1)[0] == lemma]
            chosen_lemma_analysis = lemma_analyses[0]
        except IndexError:
            return default_feats
        lemma_morpho = chosen_lemma_analysis[2]
        tag = lemma_morpho[2]
        split_tag = tag.split(":", 1)
        pos_tag = split_tag[0]
        if len(split_tag) == 1:
            feats = ""
        else:
            feats = split_tag[1]
        return feats





