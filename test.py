import spacy
from spacy.compat import import_file


nlp = spacy.load("pl_spacy_model_morfeusz")
flexer = nlp.get_pipe("flexer")


doc = nlp("czarny but na obcasie")
flexed = flexer.flex_mwe(doc[1], "pl:dat")
print(flexed, flexed == "czarnym butom na obcasie")

doc = nlp("Dwoje dzieci w zimowych strojach")
flexed = flexer.flex_mwe(doc[1], "dat")
print(flexed, flexed == "Dwojgu dzieci w zimowych strojach") # tutaj jest kwestia kryterium rysowana strzałek

doc = nlp("mały ale wytrzymały samochód bojowy")
flexed = flexer.flex_mwe(doc[3], "pl:inst")
print(flexed, flexed == "małymi ale wytrzymałymi samochodami bojowymi") 

doc = nlp("trzy bardzo stare książki z Francji")
flexed = flexer.flex_mwe(doc[3], "pl:loc")
print(flexed, flexed == "trzech bardzo starych książkach z Francji") 

doc = nlp("mały chleb i świeża marchewka")
flexed = flexer.flex_mwe(doc[1], "pl:inst")
print(flexed, flexed == "małymi chlebami i świeżymi marchewkami") 

doc = nlp("biała koszula zapinane na guziki")
flexed = flexer.flex_mwe(doc[1], "pl:gen")
print(flexed, flexed == "białych koszul zapinanych na guziki")

doc = nlp("najstarszy syn Jana Kowalskiego")
flexed = flexer.flex_mwe(doc[1], "pl:loc")
print(flexed, flexed == "najstarszych synach Jana Kowalskiego")

doc = nlp("ładne i tanie")
flexed = flexer.flex_mwe(doc[0], "sup")
print(flexed, flexed == "najładniejsze i najtańsze")

doc = nlp("nazwany i opisany")
flexed = flexer.flex_mwe(doc[0], "neg")
print(flexed, flexed == "nienazwany i nieopisany") # to nie działa przez organizację zanegowanych przymiotników w Morfeuszu

doc = nlp("widziałem to")
flexed = flexer.flex_mwe(doc[0], "pl")
print(flexed, flexed == "widzieliśmy to")


