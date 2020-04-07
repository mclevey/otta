# John McLevey
# March 2020

from tabulate import tabulate
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import scipy.sparse as ss
from sklearn.feature_extraction.text import CountVectorizer
import spacy
nlp = spacy.load('en_core_web_sm')


def context_term_matrix(text, skip_ints=True, min_df=0.05):
    """
    Produces matrices in the form required by COREX.
    """
    if skip_ints is True:
        vect = CountVectorizer(binary=True, stop_words="english",
                               lowercase=True,
                               strip_accents='ascii',
                               min_df=min_df,
                               token_pattern=r'\b[^\d\W]+\b')
    else:
        vect = CountVectorizer(binary=True,
                               stop_words="english",
                               lowercase=True,
                               strip_accents='ascii',
                               min_df=min_df)
    matrix = vect.fit_transform(text)
    matrix = ss.csr_matrix(matrix)
    words = vect.get_feature_names()

    return matrix, words


def print_topics(topics):
    """
    Just print the text of the topics to screen.
    """
    for topic_n, topic in enumerate(topics):
        words, mis = zip(*topic)
        print(f'Topic {topic_n}: {" ".join(words)}')


def topic_dot_plot(tm, tm_n, n_words):
    """
    Produce a dot plot for a single topic.
    """
    df = pd.DataFrame(tm.get_topics(topic=tm_n, n_words=n_words), columns=[
                      'Word', 'Score']).sort_values('Score')
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.hlines(y=df['Word'], xmin=0, xmax=df['Score'],
              color='#32363A', linewidth=2)
    ax.plot(df['Score'], df['Word'], "o", color='#32363A', markersize=8)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title(f'Topic {tm_n}')
    ax.set_xlabel('Mutual Information')
    ax.set_ylabel('')
    plt.show()


def get_topic_df(tm, topic_id, n_words=12):
    """
    Used in the case plot function.
    """
    df = pd.DataFrame(tm.get_topics(topic=topic_id, n_words=n_words), columns=[
                      'Word', 'Score']).sort_values('Score')
    return df


# def make_case_plot(case_tm, case_string):
#     fig = plt.figure(figsize=(12, 6))

#     ax = plt.subplot(231)
#     ax.set_title('Topic Rankings')
#     ax.plot(range(case_tm.n_hidden), case_tm.tcs, color='#32363A')
#     ax.scatter(range(case_tm.n_hidden), case_tm.tcs,
#                marker='o', color='#32363A', s=220)
#     for x, y in zip(range(case_tm.n_hidden), case_tm.tcs):
#         plt.text(x, y, f'T{x}', horizontalalignment='center',
#                  verticalalignment='center', color='white', fontsize=7)
#     # modify the axis labels
# #     a = [f'T{int(l)}' for l in ax.get_xticks().tolist()]
# #     ax.set_xticklabels(a)
#     ax.set_ylabel('MMI')
#     ax.set_xlabel('')

#     ax = plt.subplot(232)
#     ax.set_title('T0 Word Rankings')
#     t0 = get_topic_df(case_tm, 0)
#     ax.hlines(y=t0['Word'], xmin=0, xmax=t0['Score'],
#               color='#32363A', linewidth=2)
#     ax.plot(t0['Score'], t0['Word'], "o", color='#32363A', markersize=6)
#     ax.set_xlabel('')
#     ax.set_ylabel('')
#     ax.tick_params(axis='y', which='both', length=0)

#     ax = plt.subplot(233)
#     ax.set_title('T1 Word Rankings')
#     t1 = get_topic_df(case_tm, 1)
#     ax.hlines(y=t1['Word'], xmin=0, xmax=t1['Score'],
#               color='#32363A', linewidth=2)
#     ax.plot(t1['Score'], t1['Word'], "o", color='#32363A', markersize=8)
#     ax.set_xlabel('')
#     ax.set_ylabel('')
#     ax.tick_params(axis='y', which='both', length=0)

#     ax = plt.subplot(234)
#     ax.set_title('T2 Word Rankings')
#     t2 = get_topic_df(case_tm, 2)
#     ax.hlines(y=t2['Word'], xmin=0, xmax=t2['Score'],
#               color='#32363A', linewidth=2)
#     ax.plot(t2['Score'], t2['Word'], "o", color='#32363A', markersize=8)
#     ax.set_xlabel('')
#     ax.set_ylabel('')
#     ax.tick_params(axis='y', which='both', length=0)

#     ax = plt.subplot(235)
#     ax.set_title('T3 Word Rankings')
#     t3 = get_topic_df(case_tm, 3)
#     ax.hlines(y=t3['Word'], xmin=0, xmax=t3['Score'],
#               color='#32363A', linewidth=2)
#     ax.plot(t3['Score'], t3['Word'], "o", color='#32363A', markersize=8)
#     ax.set_xlabel('')
#     ax.set_ylabel('')
#     ax.tick_params(axis='y', which='both', length=0)

#     ax = plt.subplot(236)
#     ax.set_title('T4 Word Rankings')
#     t4 = get_topic_df(case_tm, 4)
#     ax.hlines(y=t4['Word'], xmin=0, xmax=t4['Score'],
#               color='#32363A', linewidth=2)
#     ax.plot(t4['Score'], t4['Word'], "o", color='#32363A', markersize=8)
#     ax.set_xlabel('')
#     ax.set_ylabel('')
#     ax.tick_params(axis='y', which='both', length=0)

#     plt.tight_layout()
#     plt.savefig(f'img/{case_string}.pdf')
#     plt.show()

def make_case_plot(case_tm, case_string, list_5_topics=[0, 1, 2, 3, 4, 5]):
    fig = plt.figure(figsize=(12, 6))

    # TOPIC RANKING PANEL
    ax = plt.subplot(231)
    ax.set_title('Topic Rankings')
    ax.plot(range(case_tm.n_hidden), case_tm.tcs, color='#32363A')
    ax.scatter(range(case_tm.n_hidden), case_tm.tcs,
               marker='o', color='#32363A', s=180)
    for x, y in zip(range(case_tm.n_hidden), case_tm.tcs):
        plt.text(x, y, f'T{x}', horizontalalignment='center',
                 verticalalignment='center', color='white', fontsize=7)
    ax.set_ylabel('MMI')
    ax.set_xlabel('')

    # TOPIC A PANEL
    topica = list_5_topics[0]
    ax = plt.subplot(232)
    ax.set_title(f'T{topica} Word Rankings')
    ta = get_topic_df(case_tm, topica, n_words=12)
    ax.hlines(y=ta['Word'], xmin=0, xmax=ta['Score'],
              color='#32363A', linewidth=2)
    ax.plot(ta['Score'], ta['Word'], "o", color='#32363A', markersize=6)
    ax.tick_params(axis='y', which='both', length=0)
    ax.set_xlabel('')
    ax.set_ylabel('')

    # TOPIC B PANEL
    ax = plt.subplot(233)
    topicb = list_5_topics[1]
    ax.set_title(f'T{topicb} Word Rankings')
    tb = get_topic_df(case_tm, topicb, n_words=12)
    ax.hlines(y=tb['Word'], xmin=0, xmax=tb['Score'],
              color='#32363A', linewidth=2)
    ax.plot(tb['Score'], tb['Word'], "o", color='#32363A', markersize=6)
    ax.tick_params(axis='y', which='both', length=0)
    ax.set_xlabel('')
    ax.set_ylabel('')

    # TOPIC C PANEL
    ax = plt.subplot(234)
    topicc = list_5_topics[2]
    ax.set_title(f'T{topicc} Word Rankings')
    tc = get_topic_df(case_tm, topicc, n_words=12)
    ax.hlines(y=tc['Word'], xmin=0, xmax=tc['Score'],
              color='#32363A', linewidth=2)
    ax.plot(tc['Score'], tc['Word'], "o", color='#32363A', markersize=6)
    ax.tick_params(axis='y', which='both', length=0)
    ax.set_xlabel('')
    ax.set_ylabel('')

    # TOPIC D PANEL
    ax = plt.subplot(235)
    topicd = list_5_topics[3]
    ax.set_title(f'T{topicd} Word Rankings')
    td = get_topic_df(case_tm, topicd, n_words=12)
    ax.hlines(y=td['Word'], xmin=0, xmax=td['Score'],
              color='#32363A', linewidth=2)
    ax.plot(td['Score'], td['Word'], "o", color='#32363A', markersize=6)
    ax.tick_params(axis='y', which='both', length=0)
    ax.set_xlabel('')
    ax.set_ylabel('')

    # TOPIC E PANEL
    ax = plt.subplot(236)
    topice = list_5_topics[4]
    ax.set_title(f'T{topice} Word Rankings')
    te = get_topic_df(case_tm, topice, n_words=12)
    ax.hlines(y=te['Word'], xmin=0, xmax=te['Score'],
              color='#32363A', linewidth=2)
    ax.plot(te['Score'], te['Word'], "o", color='#32363A', markersize=6)
    ax.tick_params(axis='y', which='both', length=0)
    ax.set_xlabel('')
    ax.set_ylabel('')

    plt.tight_layout()
    plt.savefig(f'{case_string}.pdf')
    plt.show()


def read(corex_model, topic_id, search_term, ot=True):
    top_docs = corex_model.get_top_docs(
        topic=3, n_docs=10, sort_by='log_prob')
    text = [str(each) for each in top_docs]
    search_term = f' {search_term} '
    docs = nlp.pipe(text, batch_size=10, n_threads=4)
    docs = [doc for doc in docs]
    for d in docs:
        sents = [str(t) for t in list(d.sents)]
        for sent in sents:
            if search_term in sent:
                print(sent.replace(search_term,
                                   f' ***{search_term.upper()}*** '), '\n')
                if ot is True:
                    if 'oil' in sent:
                        print(sent.replace('oil', '*** OIL ***'), '\n')
                    if 'tourism' in sent:
                        print(sent.replace('tourism', '*** TOURISM ***'), '\n')
                    if 'tourist' in sent:
                        print(sent.replace('tourist', '*** TOURIST ***'), '\n')
        print('\n------\n')


def get_top_words(corex_model, n_words=10):
    data = []
    for topic_n, topic in enumerate(corex_model.get_topics()):
        words, mis = zip(*topic)
        tab = [f'T{topic_n}', ", ".join(words[0:n_words])]
        data.append(tab)

    df = pd.DataFrame(data, columns=['Topic', 'Top Words'])

    print(tabulate(get_top_words(sl_tm)))
    return df
