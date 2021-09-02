from flexer_module import Flexer
from morf_wrapper import MorfWrapper
from combo_wrapper import ComboWrapper
from pseudo_morph import PseudoMorph

morphology_file = "polish_flexer/pl_morph.json"
nlp = ComboWrapper("polish-herbert-base", use_translation=False)
inflection_dict = MorfWrapper()

#DICT MODE
#flexer = Flexer(nlp, morphology_file, inflection_dict, "pl_encoder.mdl", "pl_decoder.mdl", "DICT")

#NEURO MODE
flexer = Flexer(nlp, morphology_file, inflection_dict, "pl_encoder.mdl", "pl_decoder.mdl", "NEURO")

text = "pierwsza siedziba Piastów"
toks = nlp(text)

flexed = flexer.flex_phrase(toks, toks, "gen:pl")
print(flexed)
