class PseudoMorph():
    def __init__(self, dictfile):
        self.gen_dict = {}
        self.an_dict = {}
        with open(dictfile) as f:
            txt = f.read()
        lines = txt.split("\n")
        for l in lines:
            if l and " " not in l:
                form, lemma, tag = l.split("\t")
                if lemma in self.gen_dict:
                    self.gen_dict[lemma].append((form, tag))
                else:
                    self.gen_dict[lemma] = [(form, tag)]
                if form in self.an_dict:
                    self.an_dict[form].append((lemma, tag))
                else:
                    self.an_dict[form] = [(lemma, tag)]

    def analyse(self, form):
        if form in self.an_dict:
            return self.an_dict[form]
        else:
            return None

    def generate(self, lemma):
        if lemma in self.gen_dict:
            generated = self.gen_dict[lemma]
            output = [{"form": g[0], "full_tag": g[1]} for g in generated]
            return output 
        else:
            return [] 

