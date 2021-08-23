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



