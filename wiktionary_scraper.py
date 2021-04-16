import requests
import pandas
import concurrent.futures
from tqdm import tqdm
from bs4 import BeautifulSoup

WIKTIONARY = "https://pl.wiktionary.org"


def get_content(address):
  r = requests.get(address)
  soup = BeautifulSoup(r.content, 'html.parser')
  return soup

phrasal_verbs = WIKTIONARY + "/wiki/Indeks:Polski_-_Związki_frazeologiczne"
soup = get_content(phrasal_verbs)
list_items = soup.findAll("ul")


links = []
for li in list_items:
  if li.get("class") or li.get("id"):
    continue
  link_tags = li.findAll("li")
  for link in link_tags:
    anchor = link.find("a")
    if anchor:
      ref = anchor.get("href")
      if not ref.startswith("/wiki/Kategoria"):
        links.append(ref)


def get_inflection_table(link):
  soup = get_content(WIKTIONARY+link)
  inflection_table = soup.find("table", class_="odmiana")
  if inflection_table:
    df = pandas.read_html(str(inflection_table))
    return df



with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
  dfs = executor.map(get_inflection_table, links)

inf_tables = [d for d in dfs if d]
