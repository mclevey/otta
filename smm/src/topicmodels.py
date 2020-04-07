import dill
import pandas as pd
import otta_functions as of
from pprint import pprint
from corextopic import corextopic as ct
from corextopic import vis_topic as vt
import spacy
nlp = spacy.load('en_core_web_sm')


with open('../data/inter/sl_text', 'r') as dillfile:
    sl_text = dill.load(sl_text, dillfile)

with open('../data/inter/il_text', 'rb') as dillfile:
    il_text = dill.load(dillfile)

with open('../data/inter/dk_text', 'rb') as dillfile:
    dk_text = dill.load(dillfile)

with open('../data/inter/nl_text', 'rb') as dillfile:
    nl_text = dill.load(dillfile)

with open('../data/inter/no_text', 'rb') as dillfile:
    no_text = dill.load(dillfile)

print(len(no_text))
print(len(nl_text))
