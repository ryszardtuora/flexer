import re
import csv
from typing import Dict, List, Set, FrozenSet, Iterable, NamedTuple, NewType
from collections.abc import Set

Form = NewType("Form", str)
Lemma = NewType("Lemma", str)
UmTag = NewType("UmTag", str)
UdFeat = NewType("UdFeat", str)
UmFeat = NewType("UmFeat", str)
UmFeats = FrozenSet[UmFeat]
EMPTY_FEAT = UmFeat("_")
UD2UM_FILE = "UD-UniMorph.tsv"

POS_TAGS = ["ADV",
            "PRO",
            "ADP",
            "DET",
            "N",
            "ADJ",
            "CONJ",
            "NUM",
            "PROPN",
            "PART",
            "INTJ",
            "V",
            ]


class UdTag(Set):
    def __init__(self, tag: str) -> None:
        self.data = set(tag.split("|"))

    def __contains__(self, value):
        return value in self.data

    def __iter__(self):
        yield from self.data

    def __len__(self):
        return len(self.data)

class CoNLLRow(NamedTuple):
    id: str
    form: Form
    lemma: Lemma
    upostag: str
    xpostag: str
    feats: str
    head: str
    deprel: str
    deps: str
    misc: str

    @classmethod
    def make(cls, string: str) -> "CoNLLRow":
        return cls._make(string.split("\t"))

def _ud2um_mapping() -> Dict[UdFeat, UmFeat]:
    ud2um = {}
    with open(UD2UM_FILE, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            ud = UdFeat(row["UD"])
            um = UmFeat(row["UniMorph"] or "_")
            ud2um[ud] = um
    return ud2um


def handle_arguments(ud: UdTag) -> List[UmFeat]:
    def handle_argument(parts):
        parts = str(parts)
        if "[psed]" in parts or "[gram]" in parts:
            return "_"

        if "[erg]" in parts:
            kind = "ER"
        elif "[dat]" in parts:
            kind = "DA"
        elif "[abs]" in parts:
            kind = "AB"
        else:
            print(parts)
            raise AssertionError

        if "=Plur" in parts:
            number = "P"
        elif "=Sing" in parts:
            number = "S"
        elif "=Dual" in parts:
            number = "D"
        else:
            assert "Number" not in str(parts)
            number = ""

        if "=1" in parts:
            person = "1"
        elif "=2" in parts:
            person = "2"
        elif "=3" in parts:
            person = "3"
        else:
            assert "Person" not in str(parts)
            person = ""
        return f"ARG{kind}{person}{number}"

    arg_parts = [p for p in ud if "[" in p and "[psor]" not in p]
    if not arg_parts:
        return [EMPTY_FEAT]
    arg_re = re.compile(r"\[(.*?)\]")
    tags = {arg_re.search(p).group(1) for p in arg_parts}  # type: ignore
    contributions = []
    for tag in tags:
        arg = handle_argument([p for p in arg_parts if tag in p])
        if arg:
            contributions.append(arg)
    return contributions


def handle_possession(ud_tag: UdTag) -> UmFeat:
    psor_parts = [p for p in ud_tag if "[psor]" in p]
    if not psor_parts:
        return EMPTY_FEAT

    if "None" in str(psor_parts):
        return UmFeat("PSSD")

    try:
        assert len(psor_parts) <= 2
    except AssertionError:
        print(psor_parts)
        raise

    if "Number[psor]=Plur" in psor_parts:
        number = "P"
    elif "Number[psor]=Sing" in psor_parts:
        number = "S"
    elif "Number[psor]=Dual" in psor_parts:
        number = "D"
    else:
        assert "Number" not in str(psor_parts)
        number = ""

    if "Person[psor]=1" in psor_parts:
        person = "1"
    elif "Person[psor]=2" in psor_parts:
        person = "2"
    elif "Person[psor]=3" in psor_parts:
        person = "3"
    else:
        assert "Person" not in str(psor_parts)
        person = ""

    return UmFeat(f"PSS{person}{number}")

class Translator():
    def __init__(self):
        self.ud2um_mapping = _ud2um_mapping()

    def ud2um(self, ud_tag: UdTag) -> UmTag:
        um_tag: List[UmFeat] = []
        possession = handle_possession(ud_tag)
        um_tag.append(possession)
        arguments = handle_arguments(ud_tag)
        um_tag.extend(arguments)

        for part in ud_tag:
            if "," not in part:
                tag = self.process_tag(part)
                um_tag.append(tag)
            else:
                key, vals = part.split("=")
                vals = vals.split(",")
                all_parts = []
                for val in vals:
                    tag = self.process_tag(UdFeat(f"{key}={val}"))
                    all_parts.append(tag)
                all_parts = [p for p in all_parts if p != "_"]
                um_tag.append(
                    UmFeat(f"{{{'/'.join(all_parts)}}}") if all_parts else EMPTY_FEAT
                )
        um_tag = [f for f in um_tag if str(f) != "_"] or [EMPTY_FEAT]
        feats = ";".join([f for f in um_tag if f not in POS_TAGS])
        try:
            pos_tag = [f for f in um_tag if f in POS_TAGS][0]
        except IndexError:
            pos_tag = "X"
        return pos_tag, feats
    
    def process_tag(self, part: UdFeat) -> UmFeat:
        try:
            um_part = self.ud2um_mapping[part]
        except KeyError:
            # print("Couldn't find", part)
            return EMPTY_FEAT
        #     if part != "_":
        # if part in UD_valid_tags:
        #     self.no_match_tags.add(part)
        # else:
        #     self.invalid_tags.add(part)
        else:
            return um_part

    def translate(self, conllu_line):
        record = CoNLLRow.make(conllu_line) 
        ud_tag = UdTag(f"{record.upostag}|{record.feats}")
        pos_tag, feats = self.ud2um(ud_tag)
        return pos_tag, feats

