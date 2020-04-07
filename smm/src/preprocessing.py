"""
This script pulls in the raw story data, puts it through a spaCy pipeline,
and then writes it out for the topic modelling step.

John
"""

import dill
import pandas as pd
import spacy
nlp = spacy.load('en_core_web_sm')

scotland = pd.read_csv('../data/raw/sl.csv')
norway = pd.read_csv('../data/raw/no.csv')
denmark = pd.read_csv('../data/raw/dk.csv')
iceland = pd.read_csv('../data/raw/il.csv')
newfoundland_labrador = pd.read_csv('../data/raw/nl.csv')

sl_text = scotland['content']
no_text = norway['content']
dk_text = denmark['content']
il_text = iceland['content']
nl_text = newfoundland_labrador['content']


def process_texts(texts):
    processed = []

    for doc in nlp.pipe(texts, disable=["tagger", "parser"]):
        pos_tags = ['NOUN', 'PROPN', 'ADJ', 'VERB', 'AUX']

        proc = []
        for token in doc:
            if token.is_alpha and len(token) > 1:
                if token.is_stop is False and token.pos_ in pos_tags:
                    proc.append(token.lemma_)
        proc = " ".join(proc)
        processed.append(proc)
    return processed


sl_text = process_texts(sl_text.tolist())
il_text = process_texts(il_text.tolist())
dk_text = process_texts(dk_text.tolist())
nl_text = process_texts(nl_text.tolist())
no_text = process_texts(no_text.tolist())

with open('../data/inter/sl_text', 'wb') as dillfile:
    dill.dump(sl_text, dillfile)

with open('../data/inter/il_text', 'wb') as dillfile:
    dill.dump(il_text, dillfile)

with open('../data/inter/dk_text', 'wb') as dillfile:
    dill.dump(dk_text, dillfile)

with open('../data/inter/nl_text', 'wb') as dillfile:
    dill.dump(nl_text, dillfile)

with open('../data/inter/no_text', 'wb') as dillfile:
    dill.dump(no_text, dillfile)
