import spacy
from spacy.compat import import_file


nlp = spacy.load("pl_spacy_model_morfeusz")
flexer = nlp.get_pipe("flexer")

inflection_instances = [
  ("czarny but na obcasie", "pl:dat", "czarnym butom na obcasie"),
  ("dwoje dzieci w zimowych strojach", "dat", "dwojgu dzieci w zimowych strojach"), # tutaj jest kwestia kryterium rysowana strzałek
  ("mały ale wytrzymały samochód bojowy", "pl:inst", "małymi ale wytrzymałymi samochodami bojowymi"),
  ("trzy bardzo stare książki z Francji", "pl:loc", "trzech bardzo starych książkach z Francji"),
  ("mały chleb i świeża marchewka", "pl:inst", "małymi chlebami i świeżymi marchewkami"),
  ("Biała koszula zapinane na guziki", "pl:gen", "Białych koszul zapinanych na guziki"),
  ("najstarszy syn Jana Kowalskiego", "pl:loc", "najstarszych synach Jana Kowalskiego"),
  ("ładne i tanie", "sup", "najładniejsze i najtańsze"),
  ("nazwany i opisany", "neg", "nienazwany i nieopisany"), # to nie działa przez organizację zanegowanych przymiotników w Morfeuszu
  ("widziałem to", "pl", "widzieliśmy to"), # problem z fleksją subword units
]

lemmatization_instances = [
  ("sądach pierwszej instancji", "sąd pierwszej instancji"),
  ("generałem Janem Kowalskim", "generał Jan Kowalski"),
  ("międzynarodowym funduszem walutowym", "międzynarodowy fundusz walutowy"),
  ("najwyższej izby kontroli", "najwyższa izba kontroli"),
  ("mieście stołecznym Warszawie", "miasto stołeczne Warszawa"), # błąd parsera, powinna być apozycja?
  ("prezydentem miasta stołecznego Warszawy", "prezydent miasta stołecznego Warszawy"),
  ("pierwszej połowie dwudziestego wieku", "pierwsza połowa dwudziestego wieku"),
  ("ulicy Bohaterów Getta", "ulica Bohaterów Getta"),
  ("województwu kieleckiemu", "województwo kieleckie"),
  ("pierwszym zdobywcom złotego medalu", "pierwszy zdobywca złotego medalu"),
]

print("Inflection")
for source, pattern, target in inflection_instances:
  doc = nlp(source)
  head = [token for token in doc if token.dep_ == "ROOT"][0]
  flexed = flexer.flex_mwe(doc[head.i], pattern)
  print("Inflection: ", flexed, flexed == target)

for source, target in lemmatization_instances:
  doc = nlp(source)
  head = [token for token in doc if token.dep_ == "ROOT"][0]
  lemmatized = flexer.lemmatize_mwe(doc[head.i])
  print("Lemmatization: ", lemmatized, lemmatized == target)



# dla apozycji (i flatów?) lematyzować
# dla pozostałych podrzędników odmieniać do formy zlematyzowanego głównego członu
# Polem jest w clarinie

