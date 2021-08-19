import morfeusz2

class MorfWrapper():
    def __init__(self):
        self.morf = morfeusz2.Morfeusz(expand_tags = True, whitespace = morfeusz2.KEEP_WHITESPACES, generate = True)

    def generate(self, lemma):
        generated = self.morf.generate(lemma)
        processed = [{"form": g[0], "full_tag": g[2]} for g in generated]
        return processed



