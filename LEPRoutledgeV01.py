
# python -m streamlit run lepanV01R.py

vac_red = r"PolarIs1_VaccRed_up_ext.xlsx"   #r"PolarIs1_VaccRed.xlsx"
us16 = r"app_US2016_up.xlsx"  # app_US2016_up

vac_red_log = r"PolarIs1_Logos_Ann1.xlsx"
vac_red_log2 = r"PolarIs1-VaccRed_RawDataSegmented_ADU.xlsx"

us16_log = r"US2016r_Logos_Ann1.xlsx"
us16_log2 = r"US2016_reddit_RawData_ADU.xlsx"



colors_log = {'Ethos Attack':'#FF4444', 'No Ethos':'#022D96', 'Ethos Support':'#369C0E',
        'Logos Attack':'#FF4444', 'No Logos':'#022D96', 'Logos Support':'#369C0E',
        'Negative Pathos':'#FF4444', 'No Pathos':'#022D96', 'Positive Pathos':'#369C0E',
        'Ethos': '#FA9718', 'Logos': '#FA9718', 'Pathos': '#FA9718',
        }

# imports
import streamlit as st
from PIL import Image
from collections import Counter
import pandas as pd
pd.set_option("max_colwidth", 400)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
#plt.style.use("seaborn-talk")


import spacy
nlp = spacy.load('en_core_web_sm')

pd.options.mode.chained_assignment = None
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


import plotly.express as px
import plotly
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageFont

import wordcloud
from wordcloud import WordCloud, STOPWORDS

import nltk
from nltk.text import Text

# functions

ethos_mapping = {0: 'neutral', 1: 'support', 2: 'attack'}
valence_mapping = {0: 'neutral p', 1: 'positive', 2: 'negative'}


def clean_text(df, text_column, text_column_name = "content"):
  import re
  new_texts = []
  for text in df[text_column]:
    text_list = str(text).lower().split(" ")
    new_string_list = []
    for word in text_list:
      if 'http' in word:
        word = "url"
      elif ('@' in word) and (len(word) > 1):
        word = "@"
      if (len(word) > 1) and not word == 'amp' and not (word.isnumeric()):
        new_string_list.append(word)
    new_string = " ".join(new_string_list)
    new_string = re.sub("\d+", " ", new_string)
    new_string = new_string.replace('\n', ' ')
    new_string = new_string.replace('  ', ' ')
    new_string = new_string.strip()
    new_texts.append(new_string)
  df[text_column_name] = new_texts
  return df


def make_word_cloud(comment_words, width = 1100, height = 650, colour = "black", colormap = "brg", stops = True):
    stopwords = set(STOPWORDS)
    if stops:
            wordcloud = WordCloud(collocations=False, max_words=100, colormap=colormap, width = width, height = height,
                        background_color ='black',
                        min_font_size = 16, ).generate(comment_words) # , stopwords = stopwords
    else:
            wordcloud = WordCloud(collocations=False, max_words=100, colormap=colormap, width = width, height = height,
                        background_color ='black',
                        min_font_size = 16, stopwords = stopwords ).generate(comment_words) # , stopwords = stopwords

    fig, ax = plt.subplots(figsize = (width/ 100, height/100), facecolor = colour)
    ax.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    plt.show()
    return fig, wordcloud.words_.keys()


def prepare_cloud_lexeme_data(data_neutral, data_support, data_attack):

  # neutral df
  neu_text = " ".join(data_neutral['sentence_lemmatized'].apply(lambda x: " ".join(t for t in set(x.split()))).to_numpy())
  count_dict_df_neu_text = Counter(neu_text.split(" "))
  df_neu_text = pd.DataFrame( {"word": list(count_dict_df_neu_text.keys()),
                              'neutral #': list(count_dict_df_neu_text.values())} )
  df_neu_text.sort_values(by = 'neutral #', inplace=True, ascending=False)
  df_neu_text.reset_index(inplace=True, drop=True)
  #df_neu_text = df_neu_text[~(df_neu_text.word.isin(stops))]

  # support df
  supp_text = " ".join(data_support['sentence_lemmatized'].apply(lambda x: " ".join(t for t in set(x.split()))).to_numpy())
  count_dict_df_supp_text = Counter(supp_text.split(" "))
  df_supp_text = pd.DataFrame( {"word": list(count_dict_df_supp_text.keys()),
                              'support #': list(count_dict_df_supp_text.values())} )

  df_supp_text.sort_values(by = 'support #', inplace=True, ascending=False)
  df_supp_text.reset_index(inplace=True, drop=True)
  #df_supp_text = df_supp_text[~(df_supp_text.word.isin(stops))]

  merg = pd.merge(df_supp_text, df_neu_text, on = 'word', how = 'outer')

  #attack df
  att_text = " ".join(data_attack['sentence_lemmatized'].apply(lambda x: " ".join(t for t in set(x.split()))).to_numpy())
  count_dict_df_att_text = Counter(att_text.split(" "))
  df_att_text = pd.DataFrame( {"word": list(count_dict_df_att_text.keys()),
                              'attack #': list(count_dict_df_att_text.values())} )

  df_att_text.sort_values(by = 'attack #', inplace=True, ascending=False)
  df_att_text.reset_index(inplace=True, drop=True)
  #df_att_text = df_att_text[~(df_att_text.word.isin(stops))]

  df2 = pd.merge(merg, df_att_text, on = 'word', how = 'outer')
  df2.fillna(0, inplace=True)
  df2['general #'] = df2['support #'] + df2['attack #'] + df2['neutral #']
  df2['word'] = df2['word'].str.replace("'", "_").replace("”", "_").replace("’", "_")
  return df2


import random
def wordcloud_lexeme(dataframe, lexeme_threshold = 90, analysis_for = 'support', cmap_wordcloud = 'Greens', stops = False):
  '''
  analysis_for:
  'support',
  'attack',
  'both' (both support and attack)

  cmap_wordcloud: best to choose from:
  gist_heat, flare_r, crest, viridis

  '''
  if analysis_for == 'attack':
    #print(f'Analysis for: {analysis_for} ')
    cmap_wordcloud = 'Reds' #gist_heat
    dataframe['precis'] = (round(dataframe['attack #'] / dataframe['general #'], 3) * 100).apply(float) # att
  elif analysis_for == 'both':
    #print(f'Analysis for: {analysis_for} ')
    cmap_wordcloud = 'autumn' #viridis
    dataframe['precis'] = (round((dataframe['support #'] + dataframe['attack #']) / dataframe['general #'], 3) * 100).apply(float) # both supp & att
  else:
    #print(f'Analysis for: {analysis_for} ')
    dataframe['precis'] = (round(dataframe['support #'] / dataframe['general #'], 3) * 100).apply(float) # supp

  dfcloud = dataframe[(dataframe['precis'] >= int(lexeme_threshold)) & (dataframe['general #'] > 3) & (dataframe.word.map(len)>3)]
  #print(f'There are {len(dfcloud)} words for the analysis of language {analysis_for} with precis threshold equal to {lexeme_threshold}.')
  n_words = dfcloud['word'].nunique()
  text = []
  for i in dfcloud.index:
    w = dfcloud.loc[i, 'word']
    w = str(w).strip()
    if analysis_for == 'both':
      n = int(dfcloud.loc[i, 'support #'] + dfcloud.loc[i, 'attack #'])
    else:
      n = int(dfcloud.loc[i, str(analysis_for)+' #']) #  + dfcloud.loc[i, 'attack #']   dfcloud.loc[i, 'support #']+  general
    l = np.repeat(w, n)
    text.extend(l)

  import random
  random.shuffle(text)
  #st.write(f"There are {n_words} words.")
  if n_words < 1:
      st.error('No words with a specified threshold. \n Try lower value of threshold.')
      st.stop()
  figure_cloud, figure_cloud_words = make_word_cloud(" ".join(text), 1000, 620, '#1E1E1E', str(cmap_wordcloud), stops = stops) #gist_heat / flare_r crest viridis
  return figure_cloud, dfcloud, figure_cloud_words



def transform_text(dataframe, text_column):
  data = dataframe.copy()
  pos_column = []

  for doc in nlp.pipe(data[text_column].apply(str)):
    pos_column.append(" ".join(list( token.pos_ for token in doc)))
  data["POS_tags"] = pos_column
  return data



def UserRhetStrategy(data_list):
    st.write("## Rhetoric Strategy")
    add_spacelines()
    rhetoric_dims = ['ethos', 'pathos']
    df = data_list[0].copy()
    user_stats_df = user_stats_app(data = df)
    user_stats_df.fillna(0, inplace=True)
    cc = ['size',
          'ethos_n', 'ethos_support_n', 'ethos_attack_n',
          'pathos_n', 'pathos_negative_n', 'pathos_positive_n',
          ]
    user_stats_df[cc] = user_stats_df[cc].astype('int')
    user_stats_df_desc = user_stats_df.describe().round(3)
    cols_strat_zip = [
                    ('ethos_support_percent', 'pathos_positive_percent'),
                    ('ethos_attack_percent', 'pathos_negative_percent'),
                        ]
    cols_strat = ['ethos_support_percent', 'pathos_positive_percent',
                'ethos_attack_percent', 'pathos_negative_percent'
                  ]
    user_stats_df[cols_strat] = user_stats_df[cols_strat].round(-1)
    user_stats_df[cols_strat] = user_stats_df[cols_strat].astype('int')
    range_list = []
    number_users = []
    rhetoric_list = []
    bin_low = [1, 11, 21, 31, 41, 51, 61, 71, 81, 91]
    bin_high = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    dimensions = ['ethos_support_percent', 'pathos_positive_percent']
    for dim in dimensions:
        for val in zip(bin_low, bin_high):
            rhetoric_list.append(dim)
            range_list.append(str(val))
            count_users = len(user_stats_df[ (user_stats_df[dim] >= int(val[0])) & (user_stats_df[dim] <= int(val[1]))])
            number_users.append(count_users)
    heat_df = pd.DataFrame({'range': range_list, 'values': number_users, 'dimension':rhetoric_list})
    heat_df['dimension'] = heat_df['dimension'].str.replace("_percent", "")

    heat_grouped = heat_df.pivot(index='range', columns='dimension', values='values')
    range_list_at = []
    number_users_at = []
    rhetoric_list_at = []
    dimensions_at = ['ethos_attack_percent', 'pathos_negative_percent']
    for dim in dimensions_at:
        for val in zip(bin_low, bin_high):
            rhetoric_list_at.append(dim)
            range_list_at.append(str(val))
            count_users = len(user_stats_df[ (user_stats_df[dim] >= int(val[0])) & (user_stats_df[dim] <= int(val[1]))])
            number_users_at.append(count_users)
    heat_df_at = pd.DataFrame({'range': range_list_at, 'values': number_users_at, 'dimension':rhetoric_list_at})
    heat_df_at['dimension'] = heat_df_at['dimension'].str.replace("_percent", "")
    heat_grouped_at = heat_df_at.pivot(index='range', columns='dimension', values='values')

    sns.set(style = 'whitegrid', font_scale=1.4)
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    sns.heatmap(heat_grouped_at, ax=axes[1], cmap='Reds', linewidths=0.1, annot=True)
    sns.heatmap(heat_grouped, ax=axes[0], cmap='Greens', linewidths=0.1, annot=True)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("range - percentage of texts %\n")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")
    plt.tight_layout(pad = 3)
    plt.show()
    st.write("#### Strategies - Overview")
    _, hsm, _ = st.columns([1, 15, 1])
    with hsm:
        st.pyplot(fig)
    add_spacelines(2)

    sns.set(style = 'whitegrid', font_scale=1.06)
    fig, axes = plt.subplots(2, 1, figsize=(10, 14))
    axes = axes.flatten()
    nz = 0
    for cz in cols_strat_zip[:1]:
        data_crosstab = pd.crosstab(user_stats_df[ user_stats_df[[cz[0], cz[1]]].all(axis=1) ][cz[0]],
                                    user_stats_df[ user_stats_df[[cz[0], cz[1]]].all(axis=1) ][cz[1]],
                                    margins = False)
        #data_crosstab = data_crosstab[data_crosstab[data_crosstab.columns].any(axis=1)]
        #data_crosstab = data_crosstab[data_crosstab[data_crosstab.columns].any(axis=0)]
        htmp = sns.heatmap(data_crosstab, ax=axes[nz], cmap='Greens', linewidths=0.1, annot=True)
        nz += 1

    for cz in cols_strat_zip[1:]:
        data_crosstab = pd.crosstab(user_stats_df[ user_stats_df[[cz[0], cz[1]]].all(axis=1) ][cz[0]],
                                    user_stats_df[ user_stats_df[[cz[0], cz[1]]].all(axis=1) ][cz[1]],
                                    margins = False)
        #data_crosstab = data_crosstab[data_crosstab[data_crosstab.columns].any(axis=1)]
        #data_crosstab = data_crosstab[data_crosstab[data_crosstab.columns].any(axis=0)]
        htmn = sns.heatmap(data_crosstab, ax=axes[nz], cmap='Reds', linewidths=0.1, annot=True)
        nz += 1
    htmp.set_yticklabels(htmp.get_yticklabels(), rotation=0)
    htmn.set_yticklabels(htmn.get_yticklabels(), rotation=0)
    plt.tight_layout(pad = 2.3)
    #plt.xticks(rotation = 0)
    #plt.yticks(rotation = 0)
    plt.show()

    # need to adjust it
    #st.write("#### Strategies - Cross View")
    #_, hsm2, _ = st.columns([1, 3, 1])
    #with hsm2:
        #st.pyplot(fig)
    #st.write('***********************************************************************')




def MainPage():
    #add_spacelines(2)
    with st.expander("Read abstract"):
        add_spacelines(2)
        st.write("""Digitalisation is rapidly transforming our societies, transforming the dynamics of
                    our interactions, transforming the culture of our debates. One of the major threats
                    associated with digitalisation – which manifests itself in online misbehaviour such
                    as hate speech, fake news, echo chambers, cyber tribalism, filter bubbles and so on -
                    is a violation of the basic condition for trusting and being trustworthy. Thus, when
                    we calibrate our focus on this critical requirement for constructive, reasonable and
                    responsible interactions, then ethos, including ethotic misbehaviour, becomes cen-
                    tral for the study of rhetoric. If we are furthermore keen to capture the dynamics
                    of scaled-up network of communication, which is the constitutive feature of inter-
                    actions in the digital society, then the new approach to the study of rhetoric: its
                    subject-matter, methodology and goals, has to be taken. This paper presents a new
                    research program, called The New Ethos, which employs AI-based technology to
                    investigate rhetoric at scale, that is, distributed and digitised communication net-
                    works in which volume of information and velocity of message proliferation take
                    on a hitherto unknown scale. We present Rhetoric Analytics, a suit of tools that
                    compute and visualise statistical patterns, trends and tendencies in rhetorical use
                    of language. This allows us to explore, for example, how Donald Trump and Hilary
                    Clinton build their communication to win elections and how social media users re-
                    act to their rhetorical strategies. This opens the path to comprehend the present
                    and future of human communication and human condition. By unifying philosophy,
                    linguistics and Artificial Intelligence, this goal becomes closer than ever before.""")

    with st.container():
        df_sum = pd.DataFrame(
                {
                        "Corpus": ['Covid', 'ElectionsSM'],
                        "# Words": [30014, 30099], 
                        "# ADU": [2706, 3827], 
                        "# Posts": [963, 1317], 
                        "# Speakers": [465, 1317], 
                }
        )

        df_iaa = pd.DataFrame(
                {
                        'Covid': [ 440, 59, 630, 1233, 653, 152, 0.752, 0.618, 0.417 ], 
                        'ElectionsSM' : [ 847, 492, 581, 1144, 1294, 190, 0.793, 0.817, 0.573 ],           
                        
                }, index = ["# Ethos attack", "# Ethos support",  "# Logos attack",  "# Logos support",  "# Pathos negative", "# Pathos positive", 'IAA Logos', 'IAA Ethos', 'IAA Pathos' ]
        )            
        st.write("**Data summary**")
        st.dataframe(df_sum.set_index("Corpus"))
        st.dataframe(df_iaa)
        add_spacelines(2)
        st.write("**[The New Ethos Lab](https://newethos.org/)**")
        #add_spacelines(1)
        st.write(" ************************************************************* ")
    st.write('<style>div.row-widget.stRadio > div{flex-direction:column;font-size=18px;}</style>', unsafe_allow_html=True)






#@st.cache
@st.cache_data
def PolarizingNetworksSub(df3):
    pos_n = []
    neg_n = []
    neu_n = []
    tuples_tree = {
     'pos': [],
     'neg': [],
     'neu': []}
    for i in df3.index:
     if df3.ethos_label.loc[i] == 'support':
       tuples_tree['pos'].append(tuple( [str(df3.source.loc[i]), str(df3.Target.loc[i]).replace("@", '')] ))
       pos_n.append(df3.source.loc[i])

     elif df3.ethos_label.loc[i] == 'attack':
       tuples_tree['neg'].append(tuple( [str(df3.source.loc[i]), str(df3.Target.loc[i]).replace("@", '')] ))
       neg_n.append(df3.source.loc[i])

     elif df3.ethos_label.loc[i] == 'neutral':
       tuples_tree['neu'].append(tuple( [str(df3.source.loc[i]), str(df3.Target.loc[i]).replace("@", '')] ))
       neu_n.append(df3.source.loc[i])

    G = nx.DiGraph()

    default_weight = 0.7
    for nodes in tuples_tree['neu']:
        n0 = nodes[0]
        n1 = nodes[1]
        if n0 != n1:
            if G.has_edge(n0,n1):
                if G[n0][n1]['weight'] <= 4:
                    G[n0][n1]['weight'] += default_weight
            else:
                G.add_edge(n0,n1, weight=default_weight, color='blue')

    default_weight = 0.9
    for nodes in tuples_tree['neg']:
        n0 = nodes[0]
        n1 = nodes[1]
        if n0 != n1:
            if G.has_edge(n0,n1):
                if G[n0][n1]['weight'] <= 4:
                    G[n0][n1]['weight'] += default_weight
            else:
                G.add_edge(n0,n1, weight=default_weight, color='red')

    default_weight = 0.9
    for nodes in tuples_tree['pos']:
        n0 = nodes[0]
        n1 = nodes[1]
        if n0 != n1:
            if G.has_edge(n0,n1):
                if G[n0][n1]['weight'] <= 4:
                    G[n0][n1]['weight'] += default_weight
            else:
                G.add_edge(n0,n1, weight=default_weight, color='green')

    colors_nx_node = {}
    for n0 in G.nodes():
        if not (n0 in neu_n or n0 in neg_n or n0 in pos_n):
            colors_nx_node[n0] = 'grey'
        elif n0 in neu_n and not (n0 in neg_n or n0 in pos_n):
            colors_nx_node[n0] = 'blue'
        elif n0 in pos_n and not (n0 in neg_n or n0 in neu_n):
            colors_nx_node[n0] = 'green'
        elif n0 in neg_n and not (n0 in neu_n or n0 in pos_n):
            colors_nx_node[n0] = 'red'
        else:
            colors_nx_node[n0] = 'gold'
    nx.set_node_attributes(G, colors_nx_node, name="color")
    return G



def transform_text(dataframe, text_column):
  data = dataframe.copy()
  pos_column = []

  for doc in nlp.pipe(data[text_column].apply(str)):
    pos_column.append(" ".join(list( token.pos_ for token in doc)))
  data["POS_tags"] = pos_column
  return data


def FellowsDevils(df_list):
    st.write("### Fellows - Devils")
    add_spacelines(1)
    meth_feldev = 'frequency' # st.radio("Choose a method of calculation", ('frequency', 'log-likelihood ratio') )
    add_spacelines(1)
    selected_rhet_dim = 'ethos'
    selected_rhet_dim = selected_rhet_dim+"_label"
    add_spacelines(1)
    df = df_list[0]
    df.source = df.source.astype('str')
    df.source = np.where(df.source == 'nan', 'user1', df.source)
    df.source = "@" + df.source

    df = df.drop_duplicates(subset = ['source', 'sentence'])
    df.Target = df.Target.str.replace('humans', 'people')
    src = df.source.unique()
    df.Target = df.Target.str.replace("@@", "")
    df.Target = df.Target.str.replace("@", "")
    df.source = df.source.str.replace("@@", "")
    df.source = df.source.str.replace("@", "")

    df['mentions'] = df.sentence.apply(lambda x: " ".join( w for w in str(x).split() if '@' in w ))
    #df['sentence_lemmatized'] = df.sentence.apply(lambda x: " ".join( str(w).replace("#", "") for w in str(x).split() if not '@' in w ))
    #df = lemmatization(df, 'sentence_lemmatized')
    #df = clean_text(df, 'sentence_lemmatized', text_column_name = "sentence_lemmatized")
    #df['sentence_lemmatized'] = df['sentence_lemmatized'].astype('str') + " " + df['mentions']
    df['sentence_lemmatized'] = df['Target']

    if not 'neutral' in df['ethos_label'].unique():
        df['ethos_label'] = df['ethos_label'].map(ethos_mapping)

    if not meth_feldev == 'frequency':
        ddmsc = ['support', 'attack']
        odds_list_of_dicts = []
        effect_list_of_dicts = []
        freq_list_of_dicts = []
        # 1 vs rest
        #num = np.floor( len(df) / 10 )
        for ii, ddmsc1 in enumerate(ddmsc):
            dict_1vsall_percent = {}
            dict_1vsall_effect_size = {}
            dict_1vsall_freq = {}

            ddmsc12 = set(ddmsc).difference([ddmsc1])
            #all100popular = Counter(" ".join( df.lemmatized.values ).split()).most_common(100)
            #all100popular = list(w[0] for w in all100popular)

            ddmsc1w = " ".join( df[df[selected_rhet_dim] == ddmsc1 ].sentence_lemmatized.fillna('').astype('str').values ).split() # sentence_lemmatized
            c = len(ddmsc1w)
            #ddmsc1w = list(w for w in ddmsc1w if not w in all100popular)
            ddmsc1w = Counter(ddmsc1w).most_common() # num
            if ddmsc1 in ['positive', 'support']:
                ddmsc1w = [w for w in ddmsc1w if w[1] >= 3 ]
            else:
                ddmsc1w = [w for w in ddmsc1w if w[1] > 3 ]
            #print('**********')
            #print(len(ddmsc1w), ddmsc1w)
            #print([w for w in ddmsc1w if w[1] > 2 ])
            #print(len([w for w in ddmsc1w if w[1] > 2 ]))
            ddmsc1w_word = dict(ddmsc1w)

            #st.write( list(ddmsc12)[0] )

            ddmsc2w = " ".join( df[ df[selected_rhet_dim] == list(ddmsc12)[0] ].sentence_lemmatized.fillna('').astype('str').values ).split() # sentence_lemmatized
            d = len(ddmsc2w)
            #ddmsc2w = list(w for w in ddmsc2w if not w in all100popular)
            ddmsc2w = Counter(ddmsc2w).most_common()
            ddmsc2w_word = dict(ddmsc2w)


            ddmsc1w_words = list( ddmsc1w_word.keys() )
            for n, dim in enumerate( ddmsc1w_words ):

                a = ddmsc1w_word[dim]
                try:
                    b = ddmsc2w_word[dim]
                except:
                    b = 0.5

                ca = c-a
                bd = d-b

                E1 = c*(a+b) / (c+d)
                E2 = d*(a+b) / (c+d)

                g2 = 2*((a*np.log(a/E1)) + (b* np.log(b/E2)))
                g2 = round(g2, 2)

                odds = round( (a*(d-b)) / (b*(c-a)), 2)

                if odds > 1:

                    if g2 > 10.83:
                        #print(f"{dim, g2, odds} ***p < 0.001 ")
                        dict_1vsall_percent[dim] = odds
                        dict_1vsall_effect_size[dim] = 0.001
                        dict_1vsall_freq[dim] = a
                    elif g2 > 6.63:
                        #print(f"{dim, g2, odds} **p < 0.01 ")
                        dict_1vsall_percent[dim] = odds
                        dict_1vsall_effect_size[dim] = 0.01
                        dict_1vsall_freq[dim] = a
                    elif g2 > 3.84:
                        #print(f"{dim, g2, odds} *p < 0.05 ")
                        dict_1vsall_percent[dim] = odds
                        dict_1vsall_effect_size[dim] = 0.05
                        dict_1vsall_freq[dim] = a
            #print(dict(sorted(dict_1vsall_percent.items(), key=lambda item: item[1])))
            odds_list_of_dicts.append(dict_1vsall_percent)
            effect_list_of_dicts.append(dict_1vsall_effect_size)
            freq_list_of_dicts.append(dict_1vsall_freq)

        df_odds_pos = pd.DataFrame({
                    'word':odds_list_of_dicts[0].keys(),
                    'odds':odds_list_of_dicts[0].values(),
                    'frequency':freq_list_of_dicts[0].values(),
                    'effect_size_p':effect_list_of_dicts[0].values(),
        })
        df_odds_pos['category'] = 'fellows'
        df_odds_neg = pd.DataFrame({
                    'word':odds_list_of_dicts[1].keys(),
                    'odds':odds_list_of_dicts[1].values(),
                    'frequency':freq_list_of_dicts[1].values(),
                    'effect_size_p':effect_list_of_dicts[1].values(),
        })
        df_odds_neg['category'] = 'devils'

        df_odds_neg = df_odds_neg.sort_values(by = 'odds', ascending = False)
        df_odds_neg = df_odds_neg[df_odds_neg.word != 'user']

        df_odds_pos = df_odds_pos.sort_values(by = 'odds', ascending = False)
        df_odds_pos = df_odds_pos[df_odds_pos.word != 'user']

    else:
        df = df[df.Target != 'nan']
        df_odds_pos = pd.DataFrame( df[df.ethos_label == 'support'].Target.value_counts() ).reset_index()
        df_odds_pos.columns = ['word', 'size']
        df_odds_pos['category'] = 'fellows'
        df_odds_pos.index += 1

        df_odds_neg = pd.DataFrame( df[df.ethos_label == 'attack'].Target.value_counts() ).reset_index()
        df_odds_neg.columns = ['word', 'size']
        df_odds_neg['category'] = 'devils'
        df_odds_neg.index += 1
        #st.write(df_odds_neg)

    #st.write( df[ (df.Target != 'nan' ) & (df.ethos_label == 'neutral') ] )
    #st.stop()


    df_odds_neg = transform_text(df_odds_neg, 'word')
    df_odds_pos = transform_text(df_odds_pos, 'word')
    df_odds_neg.loc[ df_odds_neg.word.str.startswith("@"), 'POS_tags' ]  = 'PROPN'
    df_odds_pos.loc[ df_odds_pos.word.str.startswith("@"), 'POS_tags' ]  = 'PROPN'
    pos_list = ['NOUN', 'PROPN']
    df_odds_neg = df_odds_neg[df_odds_neg.POS_tags.isin(pos_list)]
    df_odds_pos = df_odds_pos[df_odds_pos.POS_tags.isin(pos_list)]

    targets_list = df.Target.astype('str').unique()
    df_odds_pos = df_odds_pos[df_odds_pos.word.isin(targets_list)]
    df_odds_neg = df_odds_neg[df_odds_neg.word.isin(targets_list)]

    df_odds_neg = df_odds_neg.reset_index(drop=True)
    df_odds_pos = df_odds_pos.reset_index(drop=True)
    df_odds_pos.index += 1
    df_odds_neg.index += 1

    df_odds_pos_words = set(df_odds_pos.word.values)
    df_odds_neg_words = set(df_odds_neg.word.values)


    tab_odd, tab_fellow, tab_devil = st.tabs(['Tables', 'Fellows', 'Devils'])
    with tab_odd:
        oddpos_c, oddneg_c = st.columns(2, gap = 'large')
        cols_odds = ['source', 'sentence', 'ethos_label', 'Target']

        with oddpos_c:
            st.write(f'Number of **{df_odds_pos.category.iloc[0]}**: {len(df_odds_pos)} ')
            st.dataframe(df_odds_pos)
            add_spacelines(1)
            pos_list_freq = df_odds_pos.word.tolist()
            freq_word_pos = st.multiselect('Choose entities for network analytics', pos_list_freq, pos_list_freq[:3])
            df_odds_pos_words = set(freq_word_pos)
            df0p = df[df.Target.isin(df_odds_pos_words)]

            #pos_list_freq = df_odds_pos.word.tolist()
            #freq_word_pos = st.multiselect('Choose a word you would like to see data cases for', pos_list_freq, pos_list_freq[:2])
            #df_odds_pos_words = set(freq_word_pos)
            #df[df_odds_pos.category.iloc[0]] = df.Target.apply(lambda x: " ".join( set([x]).intersection(df_odds_pos_words) ))
            #df0p = df[ (df[df_odds_pos.category.iloc[0]].str.split().map(len) >= 1) ]
            #st.write(f'Cases with **{freq_word_pos}** :')
            #st.dataframe(df0p[cols_odds])

            #df0p = df[df.Target.isin(pos_list_freq)]
            #df0p = df0p.groupby(['Target', 'source'], as_index=False).size()
            #df0p.Target = np.where(df0p.Target.duplicated(), '', df0p.Target)
            #df0p = df0p.rename(columns = {'size':'# references'})
            #st.write(df0p)
            add_spacelines(1)


        with oddneg_c:
            st.write(f'Number of **{df_odds_neg.category.iloc[0]}**: {len(df_odds_neg)} ')
            st.dataframe(df_odds_neg)
            add_spacelines(1)
            neg_list_freq = df_odds_neg.word.tolist()
            freq_word_neg = st.multiselect('Choose entities for network analytics', neg_list_freq, neg_list_freq[:3])
            df_odds_neg_words = set(freq_word_neg)
            df0n = df[df.Target.isin(df_odds_neg_words)]

            #neg_list_freq = df_odds_neg.word.tolist()
            #freq_word_neg = st.multiselect('Choose a word you would like to see data cases for', neg_list_freq, neg_list_freq[:2])
            #df_odds_neg_words = set(freq_word_neg)
            #df[df_odds_neg.category.iloc[0]] = df.Target.apply(lambda x: " ".join( set([x]).intersection(df_odds_neg_words) ))
            #df0n = df[ (df[df_odds_neg.category.iloc[0]].str.split().map(len) >= 1) ]
            #st.write(f'Cases with **{freq_word_neg}** words:')
            #st.dataframe(df0n[cols_odds])

            #df0p = df[df.Target.isin(neg_list_freq)]
            #df0p = df0p.groupby(['Target', 'source'], as_index=False).size()
            #df0p.Target = np.where(df0p.Target.duplicated(), '', df0p.Target)
            #df0p = df0p.rename(columns = {'size':'# references'})
            #st.write(df0p)
            add_spacelines(1)


    with tab_fellow:

        st.write("")
        pos_tr = df_odds_pos_words #df_odds_pos.word.unique() #df.Target.unique()
        pos_sr = df[ (df.Target.isin(pos_tr)) & (df.ethos_label == 'support') ].source.unique() #df.source.unique()

        df0p = df[ (df.source.isin(pos_sr)) | (df.Target.isin(pos_sr)) ] #  | (df.source.isin(pos_tr)) | (df.Target.isin(pos_sr))
        df0p.Target = df0p.Target.astype('str')
        df0p = df0p[df0p.Target != 'nan']

        #st.write(df0p)
        #st.write(df0p.shape)

        df0p_graph = df0p.groupby(['source', 'Target'], as_index=False).size()
        #G_a = nx.from_pandas_edgelist(df0p_graph,
        #                    source='source',
        #                    target='Target',
        #                    edge_attr = 'size',
        #                    create_using=nx.DiGraph()) # edge_attr='emo_src',
        #edges_g = len(G_a.edges())
        node_s = 100

        sns.set(font_scale=1.35, style='whitegrid')
        #widths = np.asarray(list(nx.get_edge_attributes(G_a,'size').values()))/5
        #widths = [ w if w < 3 else 3 for w in widths ]

        fig1, ax1 = plt.subplots(figsize = (12, 10))
        #pos = graphviz_layout(G_a, prog='dot' )
        #nx.draw_networkx(G_a, with_labels=True, pos = pos,
        #      width=widths, font_size=8,
        #      alpha=0.75, node_size = 120)# edge_color=colors, node_color = colors_nodes,
        #plt.title('')
        #plt.draw()
        #plt.show()
        #st.pyplot(fig1)

        G = PolarizingNetworksSub(df0p)
        sns.set(font_scale=1.25, style='whitegrid')
        widths = list(nx.get_edge_attributes(G,'weight').values())
        widths = [ w - 0.2 if w < 2.5 else 2.5 for w in widths ]
        colors = list(nx.get_edge_attributes(G,'color').values())
        colors_nodes = list(nx.get_node_attributes(G, "color").values())

        fig2, ax2 = plt.subplots(figsize = (14, 13))
        pos = nx.drawing.layout.spring_layout(G, k=0.75, iterations=20, seed=6)

        nx.draw_networkx(G, with_labels=False, pos = pos,
               width=widths, edge_color=colors,
               alpha=0.75, node_color = colors_nodes, node_size = 450)

        font_names = ['Sawasdee', 'Gentium Book Basic', 'FreeMono']
        family_names = ['sans-serif', 'serif', 'fantasy', 'monospace']
        pos_tr = list( x.replace("@", "") for x in pos_tr )

        text = nx.draw_networkx_labels(G, pos, font_size=10,
                labels = { n:n if not (n in pos_tr or n in pos_sr) else '' for n in nx.nodes(G) } )

        for i, nodes in enumerate(pos_tr):
            # extract the subgraph
            g = G.subgraph(pos_tr[i])
            # draw on the labels with different fonts
            nx.draw_networkx_labels(g, pos, font_size=14.5, font_weight='bold', font_color = 'darkgreen')

        for i, nodes in enumerate(pos_sr):
            # extract the subgraph
            g = G.subgraph(pos_sr[i])
            # draw on the labels with different fonts
            nx.draw_networkx_labels(g, pos, font_size=10, font_weight='bold',)
        #for _, t in text.items():
            #t.set_rotation(0)

        import matplotlib.patches as mpatches
        # add legend
        att_users_only = mpatches.Patch(color='red', label='negative')
        both_users = mpatches.Patch(color='gold', label='ambivalent')
        sup_users_only = mpatches.Patch(color='green', label='positive')
        #neu_users_only = mpatches.Patch(color='blue', label='neutral')
        targ_only = mpatches.Patch(color='grey', label='target only')
        plt.legend(handles=[att_users_only, sup_users_only, both_users, targ_only],
                    loc = 'upper center', bbox_to_anchor = (0.5, 1.045), ncol = 5, title = f'{df.corpus.iloc[0].split()[0]} Network of Fellows')
        plt.draw()
        plt.show()
        st.pyplot(fig2)
        add_spacelines(2)
        df0p_graph = df0p.groupby(['source', 'Target', 'ethos_label'], as_index=False).size()
        #df0p_graph.ethos_label = df0p_graph.ethos_label.map({'attack':'negative', 'support':'positive'})
        st.dataframe(df0p_graph.rename( columns = {'size':"# references", "ethos_label":"reference"} ))



    with tab_devil:
        st.write("")

        neg_tr = df_odds_neg_words #df_odds_neg.word.unique() #df.Target.unique()
        neg_sr = df[(df.Target.isin(neg_tr)) & (df.ethos_label == 'attack') ].source.unique() #df.source.unique()

        df0p = df[ (df.source.isin(neg_sr)) | (df.Target.isin(neg_sr)) ] #  | (df.source.isin(neg_tr)) | (df.Target.isin(neg_sr))
        df0p.Target = df0p.Target.astype('str')
        df0p = df0p[df0p.Target != 'nan']

        #st.write(df0p)
        #st.write(df0p.shape)

        df0p_graph = df0p.groupby(['source', 'Target'], as_index=False).size()
        #G_a = nx.from_pandas_edgelist(df0p_graph,
        #                    source='source',
        #                    target='Target',
        #                    edge_attr = 'size',
        #                    create_using=nx.DiGraph()) # edge_attr='emo_src',
        #edges_g = len(G_a.edges())
        node_s = 100

        sns.set(font_scale=1.35, style='whitegrid')
        #widths = np.asarray(list(nx.get_edge_attributes(G_a,'size').values()))/5
        #widths = [ w if w < 2.5 else 2.5 for w in widths ]

        #fig1, ax1 = plt.subplots(figsize = (12, 10))
        #pos = graphviz_layout(G_a, prog='dot' )
        #nx.draw_networkx(G_a, with_labels=True, pos = pos,
        #      width=widths, font_size=8,
        #      alpha=0.75, node_size = 120)# edge_color=colors, node_color = colors_nodes,
        #plt.title('')
        #plt.draw()
        #plt.show()
        #st.pyplot(fig1)

        G = PolarizingNetworksSub(df0p)
        sns.set(font_scale=1.25, style='whitegrid')
        widths = list(nx.get_edge_attributes(G,'weight').values())
        widths = [ w - 0.2 if w < 2.5 else 2.5 for w in widths ]
        colors = list(nx.get_edge_attributes(G,'color').values())
        colors_nodes = list(nx.get_node_attributes(G, "color").values())

        #st.write(df.corpus.iloc[0])

        fig2, ax2 = plt.subplots(figsize = (12, 11))
        pos = nx.drawing.layout.spring_layout(G, k=0.5, iterations=25, seed=5)

        nx.draw_networkx(G, with_labels=False, pos = pos,
               width=widths, edge_color=colors,
               alpha=0.75, node_color = colors_nodes, node_size = 450)

        font_names = ['Sawasdee', 'Gentium Book Basic', 'FreeMono']
        family_names = ['sans-serif', 'serif', 'fantasy', 'monospace']
        neg_tr = list( x.replace("@", "") for x in neg_tr )

        text = nx.draw_networkx_labels(G, pos, font_size=10,
                labels = { n:n if not (n in neg_tr or n in neg_sr) else '' for n in nx.nodes(G) } )

        for i, nodes in enumerate(neg_tr):
            # extract the subgraph
            g = G.subgraph(neg_tr[i])
            # draw on the labels with different fonts
            nx.draw_networkx_labels(g, pos, font_size=14.5, font_weight='bold', font_color = 'darkred')

        for i, nodes in enumerate(neg_sr):
            # extract the subgraph
            g = G.subgraph(neg_sr[i])
            # draw on the labels with different fonts
            nx.draw_networkx_labels(g, pos, font_size=10, font_weight='bold',)
        #for _, t in text.items():
            #t.set_rotation(0)

        import matplotlib.patches as mpatches
        # add legend
        att_users_only = mpatches.Patch(color='red', label='negative')
        both_users = mpatches.Patch(color='gold', label='ambivalent')
        sup_users_only = mpatches.Patch(color='green', label='positive')
        #neu_users_only = mpatches.Patch(color='blue', label='neutral')
        targ_only = mpatches.Patch(color='grey', label='target only')
        plt.legend(handles=[att_users_only, sup_users_only, both_users, targ_only],
                    loc = 'upper center', bbox_to_anchor = (0.5, 1.045), ncol = 5, title = f'{df.corpus.iloc[0].split()[0]} Network of Devils')
        plt.draw()
        plt.show()
        st.pyplot(fig2)
        add_spacelines(2)
        df0p_graph = df0p.groupby(['source', 'Target', 'ethos_label'], as_index=False).size()
        #df0p_graph.ethos_label = df0p_graph.ethos_label.map({'attack':'negative', 'support':'positive'})
        st.dataframe(df0p_graph.rename( columns = {'size':"# references", "ethos_label":"reference"} ))



###################################################




def UserRhetStrategy1(data_list):
    st.write(f" ### Rhetoric Strategies")
    df = data_list[0].copy()
    if len(data_list) > 1:
        for i, d in enumerate(data_list):
            df = pd.concat( [df, data_list[i+1]], axis=0, ignore_index=True )
    add_spacelines(1)
    plot_type_strategy = st.radio("Type of the plot", ('heatmap', 'histogram'))
    add_spacelines(1)

    rhetoric_dims = ['ethos', 'pathos']
    pathos_cols = ['pathos_label']

    user_stats_df = user_stats_app(df)
    user_stats_df.fillna(0, inplace=True)
    for c in ['text_n', 'ethos_n', 'ethos_support_n', 'ethos_attack_n', 'pathos_n', 'pathos_negative_n', 'pathos_positive_n']:
           user_stats_df[c] = user_stats_df[c].apply(int)

    user_stats_df_desc = user_stats_df.describe().round(3)
    cols_strat = ['ethos_support_percent', 'ethos_attack_percent',
                  'pathos_positive_percent', 'pathos_negative_percent']
    if plot_type_strategy == 'histogram':
        def plot_strategies(data):
            i = 0
            for c in range(2):
                sns.set(font_scale=1, style='whitegrid')
                print(cols_strat[c+i], cols_strat[c+i+1])
                fig_stats, axs = plt.subplots(1, 2, figsize=(10, 3), sharey=True)
                axs[0].hist(data[cols_strat[c+i]], color='#009C6F')
                title_str0 = " ".join(cols_strat[c+i].split("_")[:-1]).capitalize()
                axs[0].set_title(title_str0)
                axs[0].set_ylabel('number of users\n')
                axs[0].set_xlabel('\npercentage of texts %')
                axs[0].set_xticks(np.arange(0, 101, 10))

                axs[1].hist(data[cols_strat[c+i+1]], color='#9F0155')
                title_str1 = " ".join(cols_strat[c+i+1].split("_")[:-1]).capitalize()
                axs[1].set_xlabel('\npercentage of texts %')
                axs[1].yaxis.set_tick_params(labelbottom=True)
                axs[1].set_title(title_str1)
                axs[1].set_xticks(np.arange(0, 101, 10))
                plt.show()
                i+=1
                st.pyplot(fig_stats)
                add_spacelines(2)
        plot_strategies(data = user_stats_df)

    elif plot_type_strategy == 'heatmap':
        range_list = []
        number_users = []
        rhetoric_list = []
        bin_low = [0, 11, 21, 31, 41, 51, 61, 71, 81, 91]
        bin_high = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        dimensions = ['ethos_support_percent', 'pathos_positive_percent']
        for dim in dimensions:
            for val in zip(bin_low, bin_high):
                rhetoric_list.append(dim)
                range_list.append(str(val))
                count_users = len(user_stats_df[ (user_stats_df[dim] >= int(val[0])) & (user_stats_df[dim] <= int(val[1]))])
                number_users.append(count_users)
        heat_df = pd.DataFrame({'range': range_list, 'values': number_users, 'dimension':rhetoric_list})
        heat_df['dimension'] = heat_df['dimension'].str.replace("_percent", "")
        heat_grouped = heat_df.pivot(index='range', columns='dimension', values='values')

        range_list_at = []
        number_users_at = []
        rhetoric_list_at = []
        dimensions_at = ['ethos_attack_percent', 'pathos_negative_percent']
        for dim in dimensions_at:
            for val in zip(bin_low, bin_high):
                rhetoric_list_at.append(dim)
                range_list_at.append(str(val))
                count_users = len(user_stats_df[ (user_stats_df[dim] >= int(val[0])) & (user_stats_df[dim] <= int(val[1]))])
                number_users_at.append(count_users)
        heat_df_at = pd.DataFrame({'range': range_list_at, 'values': number_users_at, 'dimension':rhetoric_list_at})
        heat_df_at['dimension'] = heat_df_at['dimension'].str.replace("_percent", "")
        heat_grouped_at = heat_df_at.pivot(index='range', columns='dimension', values='values')

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        sns.heatmap(heat_grouped_at, ax=axes[1], cmap='Reds', linewidths=0.1, annot=True)
        sns.heatmap(heat_grouped, ax=axes[0], cmap='Greens', linewidths=0.1, annot=True)
        axes[0].set_xlabel("")
        axes[0].set_ylabel("range - percentage of texts %\n")
        axes[1].set_xlabel("")
        axes[1].set_ylabel("")
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.3)
        plt.show()
        st.pyplot(fig)
        add_spacelines(2)

    with st.container():
        ethos_strat = user_stats_df[(user_stats_df.ethos_percent > user_stats_df.ethos_percent.std()+user_stats_df.ethos_percent.mean()) & \
                (user_stats_df.pathos_percent < user_stats_df.pathos_percent.std()+user_stats_df.pathos_percent.mean())]

        pathos_strat = user_stats_df[(user_stats_df.ethos_percent < user_stats_df.ethos_percent.std()+user_stats_df.ethos_percent.mean()) & \
                (user_stats_df.pathos_percent > user_stats_df.pathos_percent.std()+user_stats_df.pathos_percent.mean())]

        col1, col2, col3 = st.columns([1, 3, 3])
        with col1:
            st.write('')
        with col2:
            st.write(f"Dominant **ethos** strategy ")
            col2.metric(str(ethos_strat.shape[0]) + " users", str(round(ethos_strat.shape[0] / len(user_stats_df) * 100, 1)) + "%")

        with col3:
            st.write(f"Dominant **pathos** strategy ")
            col3.metric(str(pathos_strat.shape[0]) + " users", str(round(pathos_strat.shape[0] / len(user_stats_df) * 100, 1)) + "%")
        #add_spacelines(2)
        #dominant_percent_strategy = round(pathos_strat.shape[0] / len(user_stats_df) * 100, 1) + round(ethos_strat.shape[0] / len(user_stats_df) * 100, 1)
        #col2.write(f"##### **{round(dominant_percent_strategy, 1)}%** of users have one dominant rhetoric strategy.")
        add_spacelines(2)



emosn = ['sadness', 'anger', 'fear', 'disgust']
emosp = ['joy'] # 'surprise'
emos_map = {'joy':'emotion_positive', 'surprise':2, 'sadness':'emotion_negative', 'anger':'emotion_negative',
            'fear':'emotion_negative', 'disgust':'emotion_negative', 'neutral':'emotion_neutral'}

# app version
#@st.cache
def user_stats_app(data, source_column = 'source', ethos_column = 'ethos_label', emotion_column = 'pathos_label'):
  dataframe = data.copy() # data_list[0].copy()
  dataframe[source_column] = dataframe[source_column].astype('str')

  if not 'neutral' in dataframe[ethos_column]:
      dataframe[ethos_column] = dataframe[ethos_column].map(ethos_mapping)
  if not 'neutral' in dataframe[emotion_column]:
      dataframe[emotion_column] = dataframe[emotion_column].map(valence_mapping)

  sources_list = dataframe[dataframe[source_column] != 'nan'][source_column].unique()
  dataframe = dataframe[dataframe[source_column].isin(sources_list)]
  dataframe = dataframe.rename(columns = {'sentence':'text'})
  df = pd.DataFrame(columns = ['user', 'text_n',
                               'ethos_n', 'ethos_support_n', 'ethos_attack_n',
                               'pathos_n', 'pathos_negative_n', 'pathos_positive_n',
                             'ethos_percent', 'ethos_support_percent', 'ethos_attack_percent',
                             'pathos_percent', 'pathos_negative_percent', 'pathos_positive_percent',
                             ])
  users_list = []
  d1 = dataframe.groupby(source_column, as_index=False).size()
  d1 = d1[ d1['size'] > 1]
  sources_list = d1[source_column].unique()
  dataframe = dataframe[dataframe[source_column].isin(sources_list)]

  d2 = dataframe.groupby([source_column, ethos_column], as_index=False)['text'].size()
  d2 = d2.pivot(index = source_column, columns = ethos_column, values = 'size')
  d2 = d2.fillna(0).reset_index()
  d2.columns = ['ethos_'+c+"_n" if i >= 1 else c for i, c in enumerate(d2.columns) ]

  d22 = pd.DataFrame(dataframe.groupby(source_column)[ethos_column].value_counts(normalize=True).round(3)*100)
  d22.columns = ['percent']
  d22 = d22.reset_index()
  d22 = d22.pivot(index = source_column, columns = ethos_column, values = 'percent')
  d22 = d22.fillna(0).reset_index()
  d22.columns = ['ethos_'+c+"_percent" if i >= 1 else c for i, c in enumerate(d22.columns) ]
  #st.dataframe(d2)
  #st.dataframe(d22)
  d3 = dataframe.groupby([source_column, emotion_column], as_index=False)['text'].size()
  d3 = d3.pivot(index = source_column, columns = emotion_column, values = 'size')
  d3 = d3.fillna(0).reset_index()
  d3 = d3[[source_column,  'negative', 'positive']]
  d3.columns = ['pathos_'+c+"_n" if i >= 1 else c for i, c in enumerate(d3.columns) ]
  #st.dataframe(d3)
  d32 = pd.DataFrame(dataframe.groupby(source_column)[emotion_column].value_counts(normalize=True).round(3)*100)
  d32.columns = ['percent']
  d32 = d32.reset_index()
  d32 = d32.pivot(index = source_column, columns = emotion_column, values = 'percent')
  d32 = d32.fillna(0).reset_index()
  d32 = d32[[source_column,  'negative', 'positive']]
  d32.columns = ['pathos_'+c+"_percent" if i >= 1 else c for i, c in enumerate(d32.columns) ]

  df = d1.merge(d2, on = source_column, how = 'left')
  df = df.merge(d22, on = source_column, how = 'left')
  df = df.merge(d3, on = source_column, how = 'left')
  df = df.merge(d32, on = source_column, how = 'left')
  #df = df.fillna(0)
  #st.dataframe(df)
  df['pathos_n'] = df.pathos_negative_n + df.pathos_positive_n
  df['ethos_n'] = df.ethos_attack_n + df.ethos_support_n
  return df


def standardize(data):
  from sklearn.preprocessing import StandardScaler
  scaler = StandardScaler()
  data0 = data.copy()
  scaled_values = scaler.fit_transform(data0)
  data0.loc[:, :] = scaled_values
  return data0


def user_rhetoric_v2(data, source_column = 'source', ethos_col = 'ethos_label',
                  pathos_col = 'pathos_label'):

  import warnings
  dataframe = data.copy()

  dataframe[source_column] = dataframe[source_column].apply(str)
  sources_list = dataframe[ ~(dataframe[source_column].isin(['nan', ''])) ][source_column].unique()
  metric_value = []
  users_list = []
  if not 'neutral' in dataframe[ethos_col]:
      dataframe[ethos_col] = dataframe[ethos_col].map(ethos_mapping)
  if not 'neutral' in dataframe[pathos_col]:
      dataframe[pathos_col] = dataframe[pathos_col].map(valence_mapping)

  map_ethos_weight = {'attack':-1, 'neutral':0, 'support':1}
  map_pathos_weight = {'negative':-1, 'neutral':0, 'positive':1}
  for u in sources_list:
    users_list.append(str(u))
    df_user = dataframe[dataframe[source_column] == u]
    ethos_pathos_user = 0
    df_user_rhetoric = df_user.groupby([str(pathos_col), str(ethos_col)], as_index=False).size()
    # map weights
    df_user_rhetoric[pathos_col] = df_user_rhetoric[pathos_col].map(map_pathos_weight)
    df_user_rhetoric[ethos_col] = df_user_rhetoric[ethos_col].map(map_ethos_weight)

    ethos_pathos_sum_ids = []

    for id in df_user_rhetoric.index:
      ethos_pathos_val = np.sum(df_user_rhetoric.loc[id, str(pathos_col):str(ethos_col)].to_numpy())
      ethos_pathos_val = ethos_pathos_val * df_user_rhetoric.loc[id, 'size']
      ethos_pathos_sum_ids.append(ethos_pathos_val)

    ethos_pathos_user = np.sum(ethos_pathos_sum_ids)
    try:
        metric_value.append(int(ethos_pathos_user))
    except:
        metric_value.append(0)
  df = pd.DataFrame({'user': users_list, 'rhetoric_metric': metric_value})
  return df




def add_spacelines(number_sp=2):
    for xx in range(number_sp):
        st.write("\n")


@st.cache_data#(allow_output_mutation=True)
def load_data(file_path, indx = True, indx_col = 0):
  '''Parameters:
  file_path: path to your excel or csv file with data,

  indx: boolean - whether there is index column in your file (usually it is the first column) --> default is True

  indx_col: int - if your file has index column, specify column number here --> default is 0 (first column)
  '''
  if indx == True and file_path.endswith(".xlsx"):
    data = pd.read_excel(file_path, index_col = indx_col)
  elif indx == False and file_path.endswith(".xlsx"):
    data = pd.read_excel(file_path)

  elif indx == True and file_path.endswith(".csv"):
    data = pd.read_csv(file_path, index_col = indx_col)
  elif indx == False and file_path.endswith(".csv"):
    data = pd.read_csv(file_path)
  return data


@st.cache_data
def lemmatization(dataframe, text_column = 'sentence', name_column = False):
  '''Parameters:
  dataframe: dataframe with your data,

  text_column: name of a column in your dataframe where text is located
  '''
  df = dataframe.copy()
  lemmas = []
  for doc in nlp.pipe(df[text_column].astype('str')):
    lemmas.append(" ".join([token.lemma_ for token in doc if (not token.is_punct and not token.is_stop and not token.like_num and len(token) > 1) ]))

  if name_column:
      df[text_column] = lemmas
  else:
      df[text_column+"_lemmatized"] = lemmas
  return df



def ttr_lr(t, n, definition = 'TTR'):
    '''
    https://core.ac.uk/download/pdf/82620241.pdf
    Torruella, J., & Capsada, R. (2013). Lexical statistics and tipological structures: a measure of lexical richness. Procedia-Social and Behavioral Sciences, 95, 447-454.

    definition:
    TTR (type-token ratio) (1957, Templin),
    RTTR (root type-token ratio) (1960, Giraud),
    CTTR (corrected type-token ratio) (1964, Carrol),
    H (1960, Herdan),
    M (1966, Mass),
    '''
    definition = str(definition).upper()
    if definition == 'TTR':
        coeff = round(t / n, 2)
    elif definition == 'RTTR':
        coeff = round(t / np.sqrt(n), 2)
    elif definition == 'CTTR':
        coeff = round(t / np.sqrt(n*2), 2)
    elif definition == 'H':
        coeff = round(np.log(t) / np.log(n), 2)
    elif definition == 'M':
        coeff = round((np.log(n) - np.log(t)) / np.log2(n), 2)
    return coeff



def compnwords(dataframe, column_name = 'sentence'):
    data = dataframe.copy()
    data['nwords'] = data[column_name].astype('str').str.split().map(len)
    return data





def StatsLog(df_list, an_type = 'ADU-based'):
    #st.write("#### Sentence Length Analysis")
    add_spacelines(2)

    st.write("#### ADU-based analytics")
    conn_list = ['Logos Attack', 'Logos Support']
    map_naming = {'attack':'Ethos Attack', 'neutral':'Neutral', 'support':'Ethos Support',
            'Default Conflict': 'Logos Attack',
            'Default Rephrase' : ' Neutral',
            'Default Inference' : 'Logos Support'}
    rhetoric_dims = ['ethos', 'logos']
    df_list_et = df_list[0]
    df_list_et['nwords'] = df_list_et['sentence'].str.split().map(len)
    if not 'neutral' in df_list_et['ethos_label'].unique():
        df_list_et['ethos_label'] = df_list_et['ethos_label'].map(ethos_mapping).map(map_naming)
    df_list_log = df_list[1]
    import re
    df_list_log['locution_conclusion'] = df_list_log.locution_conclusion.apply(lambda x: " ".join( str(x).split(':')[1:]) )
    df_list_log['locution_premise'] = df_list_log.locution_premise.apply(lambda x: " ".join( str(x).split(':')[1:]) )
    df_list_log['sentence'] = df_list_log.locution_premise.astype('str')# + " " + df_list_log.locution_conclusion.astype('str')
    df_list_log['nwords_conclusion'] = df_list_log['locution_conclusion'].str.split().map(len)
    df_list_log['nwords_premise'] = df_list_log['locution_premise'].str.split().map(len)
    df_list_log['nwords'] = df_list_log[['nwords_conclusion', 'nwords_premise']].mean(axis=1).round(2)

    df_list_log.connection = df_list_log.connection.map(map_naming)
    df_list_log_stats = df_list_log.groupby(['connection'], as_index=False)['nwords'].mean().round(2)
    log_all = df_list_log_stats[df_list_log_stats.connection.isin(['Logos Attack', 'Logos Support'])].nwords.mean().round(2)
    df_list_log_stats.loc[len(df_list_log_stats)] = [' Logos All', log_all]

    df_list_et_stats = df_list_et.groupby(['ethos_label'], as_index=False)['nwords'].mean().round(2)
    et_all = df_list_et_stats[df_list_et_stats.ethos_label.isin(['Ethos Support','Ethos Attack'])].nwords.mean().round(2)
    df_list_et_stats.loc[len(df_list_et_stats)] = ['Ethos All', et_all]
    #st.stop()

    if an_type == 'Text-based':
            df_list_log_stats = df_list_log.groupby(['id_connection', 'connection'], as_index=False)[['nwords_premise', 'nwords_conclusion']].sum().round(2)
            df_list_log_stats = df_list_log_stats.groupby(['connection'], as_index=False)[['nwords_premise', 'nwords_conclusion']].mean().round(2)
            df_list_log_stats['nwords'] = df_list_log_stats[['nwords_conclusion', 'nwords_premise']].mean(axis=1).round(2)
            log_all = df_list_log_stats[df_list_log_stats.connection.isin(['Logos Attack', 'Logos Support'])].nwords.mean().round(2)
            #st.write(df_list_log_stats)
            #df_list_log_stats.loc[len(df_list_log_stats)] = [' Logos All', log_all]

            #st.write(df_list_log_stats)

    #df_list_et = compnwords(df_list_et, column_name = 'sentence')
    #df_list_log_stats = compnwords(df_list_log_stats, column_name = 'sentence')

    cet_desc, c_log_stats_desc = st.columns(2)
    #df_list_et_desc = pd.DataFrame(df_list_et[df_list_et.ethos_label.isin(['Ethos Support','Ethos Attack'])].groupby('ethos_label').nwords.describe().round(2).iloc[:, 1:])
    df_list_et_desc = pd.DataFrame(df_list_et.groupby('ethos_label').nwords.describe().round(2).iloc[:, 1:])
    df_list_et_desc = df_list_et_desc.T
    with cet_desc:
        st.write("ADU Length for **Ethos**: ")
        st.write(df_list_et_desc)
    #df_list_log_stats_desc = pd.DataFrame(df_list_log_stats[df_list_log_stats.connection.isin(conn_list)].groupby('connection').nwords.describe().round(2).iloc[:, 1:])
    conn_list = ['Logos Attack', 'Logos Support', ' Neutral']
    df_list_log_stats_desc = pd.DataFrame(df_list_log[df_list_log.connection.isin(conn_list)].groupby('connection').nwords.describe().round(2).iloc[:, 1:])
    if an_type == 'Text-based':
        df_list_log_stats = df_list_log.groupby(['id_connection', 'connection'], as_index=False)[['nwords_premise', 'nwords_conclusion']].sum().round(2)
        df_list_log_stats_desc = pd.DataFrame(df_list_log_stats[df_list_log_stats.connection.isin(conn_list)].groupby('connection').nwords.describe().round(2).iloc[:, 1:])

    df_list_log_stats_desc = df_list_log_stats_desc.T
    with c_log_stats_desc:
        st.write("ADU Length for **Logos**: ")
        st.write(df_list_log_stats_desc)

    add_spacelines(1)
    #cstat1, cstat2, cstat3, cstat4 = st.columns(4)
    cstat1, cstat2, _, _ = st.columns(4)
    with cstat1:
        le = df_list_et_desc.loc['mean', 'Ethos Attack']
        ll = df_list_log_stats_desc.loc['mean', 'Logos Attack']
        lrel = round((le *100 / ll)- 100, 2)
        #st.write(le, ll, lrel)
        st.metric('Ethos Attack vs. Logos Attack', f" {le} vs. {ll} ", str(lrel)+'%')

    with cstat2:
        le = df_list_et_desc.loc['mean', 'Ethos Support']
        ll = df_list_log_stats_desc.loc['mean', 'Logos Support']
        lrel = round((le *100 / ll)- 100, 2)
        st.metric('Ethos Support vs. Logos Support', f" {le} vs. {ll} ", str(lrel)+'%')

    #with cstat3:
        #le = df_list_et_desc.loc['mean', 'Ethos Attack']
        #ll = df_list_log_stats_desc.loc['mean', 'Logos Attack']
        #lrel = round((ll *100 / le)- 100, 2)
        #st.metric('Logos Attack vs. Ethos Attack', f" {ll} vs. {le} ", str(lrel)+'%')

    #with cstat4:
        #le = df_list_et_desc.loc['mean', 'Ethos Support']
        #ll = df_list_log_stats_desc.loc['mean', 'Logos Support']
        #lrel = round((ll *100 / le)- 100, 2)
        #st.metric('Logos Support vs. Ethos Support', f" {ll} vs. {le} ", str(lrel)+'%')

    #st.write(df_list_log_stats)
    #st.write(df_list_et_stats)
    df_list_et_stats.columns = ['connection', 'nwords']
    df_list_desc = pd.concat( [df_list_log_stats,
                                df_list_et_stats], axis = 0, ignore_index=True )

    #df_list_desc = df_list_desc.reset_index()
    #st.write(df_list_desc)
    #st.stop()
    df_list_desc.columns = ['category', 'mean']
    df_list_desc.loc[:4, 'dimension'] = 'Logos'
    df_list_desc.loc[4:, 'dimension'] = 'Ethos'
    df_list_desc = df_list_desc.sort_values(by = ['dimension', 'category'])
    #df_list_desc['category'] = df_list_desc['category'].str.replace(' Ethos Neutral', 'Neutral').str.replace(' Logos  Neutral', ' Neutral')

    sns.set(font_scale = 1.4, style = 'whitegrid')
    f_desc = sns.catplot(data = df_list_desc, x = 'category', y = 'mean', col = 'dimension',
                kind = 'bar', palette = {'Ethos Attack':'#BB0000', 'Neutral':'#3B3591', 'Ethos Support':'#026F00', 'Ethos All':'#6C6C6E',
                        'Logos Attack':'#BB0000', ' Neutral':'#3B3591', 'Logos Support':'#026F00', ' Logos All':'#6C6C6E'},
                        height = 4, aspect = 1.4, sharex=False, legend = False)
    f_desc.set(xlabel = '', ylabel = 'mean ADU length', ylim = (0, np.max(df_list_desc['mean']+2)))
    f_desc.set_xticklabels(fontsize = 13)
    for ax in f_desc.axes.ravel():
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),
            ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')
    st.pyplot(f_desc)
    st.write("************************************************************************")

    st.write("#### Word-based analytics")
    df_list_et['nwords'] = df_list_et['sentence'].astype('str').apply(lambda x: np.mean( [ len(w)  for w in x.split()] ) )
    #df_list_log_stats['nwords'] = df_list_log_stats['sentence'].astype('str').apply(lambda x: np.mean( [ len(w)  for w in x.split()] ) )
    df_list_log['nwords_conclusion'] = df_list_log['locution_conclusion'].astype('str').apply(lambda x: np.mean( [ len(w)  for w in x.split()] ) )
    df_list_log['nwords_premise'] = df_list_log['locution_premise'].astype('str').apply(lambda x: np.mean( [ len(w)  for w in x.split()] ) )
    df_list_log['nwords'] = df_list_log[['nwords_conclusion', 'nwords_premise']].mean(axis=1).round(2)


    df_list_log_stats = df_list_log.groupby(['connection'], as_index=False)['nwords'].mean().round(2)
    log_all = df_list_log_stats[df_list_log_stats.connection.isin(['Logos Attack', 'Logos Support'])].nwords.mean().round(2)
    df_list_log_stats.loc[len(df_list_log_stats)] = [' Logos All', log_all]

    df_list_et_stats = df_list_et.groupby(['ethos_label'], as_index=False)['nwords'].mean().round(2)
    et_all = df_list_et_stats[df_list_et_stats.ethos_label.isin(['Ethos Support','Ethos Attack'])].nwords.mean().round(2)
    df_list_et_stats.loc[len(df_list_et_stats)] = ['Ethos All', et_all]
    #st.stop()

    if an_type == 'Text-based':
            df_list_log_stats = df_list_log.groupby(['id_connection', 'connection'], as_index=False)[['nwords_premise', 'nwords_conclusion']].sum().round(2)
            df_list_log_stats = df_list_log_stats.groupby(['connection'], as_index=False)['nwords_premise', 'nwords_conclusion'].mean().round(2)
            df_list_log_stats['nwords'] = df_list_log_stats[['nwords_conclusion', 'nwords_premise']].mean(axis=1).round(2)
            log_all = df_list_log_stats[df_list_log_stats.connection.isin(['Logos Attack', 'Logos Support'])].nwords.mean().round(2)
            df_list_log_stats.loc[len(df_list_log_stats)] = [' Logos All', log_all]


    #cet_desc, c_log_stats_desc = st.columns(2)
    #df_list_et_desc = pd.DataFrame(df_list_et.groupby('ethos_label').nwords.describe().round(2).iloc[:, 1:])
    #df_list_et_desc = df_list_et_desc.T
    #with cet_desc:
        #st.write("Word Length for **Ethos**: ")
        #st.write(df_list_et_desc)
    #df_list_log_stats_desc = pd.DataFrame(df_list_log_stats.groupby('connection').nwords.describe().round(2).iloc[:, 1:])
    #df_list_log_stats_desc = df_list_log_stats_desc.T
    #with c_log_stats_desc:
        #st.write("Word Length for **Logos**: ")
        #st.write(df_list_log_stats_desc)

    df_list_et_stats.columns = ['connection', 'nwords']
    add_spacelines(1)
    cstat12, cstat22 ,_, _= st.columns(4)
    with cstat12:
        le = df_list_et_stats[df_list_et_stats.connection == 'Ethos Attack'].nwords.iloc[0]
        ll =  df_list_log_stats[df_list_log_stats.connection == 'Logos Attack'].nwords.iloc[0]
        lrel = round((le *100 / ll)- 100, 2)
        st.metric('Ethos Attack vs. Logos Attack', f" {le} vs. {ll} ", str(lrel)+'%')

    with cstat22:
        le = df_list_et_stats[df_list_et_stats.connection == 'Ethos Support'].nwords.iloc[0]
        ll =  df_list_log_stats[df_list_log_stats.connection == 'Logos Support'].nwords.iloc[0]
        lrel = round((le *100 / ll)- 100, 2)
        st.metric('Ethos Support vs. Logos Support', f" {le} vs. {ll} ", str(lrel)+'%')


    df_list_desc = pd.concat( [df_list_log_stats,
                                df_list_et_stats], axis = 0, ignore_index=True )

    df_list_desc.columns = ['category', 'mean']
    df_list_desc.loc[:4, 'dimension'] = 'Logos'
    df_list_desc.loc[4:, 'dimension'] = 'Ethos'
    df_list_desc = df_list_desc.sort_values(by = ['dimension', 'category'])
    #df_list_desc['category'] = df_list_desc['category'].str.replace(' Ethos Neutral', 'Neutral').str.replace(' Logos  Neutral', ' Neutral')


    sns.set(font_scale = 1.4, style = 'whitegrid')
    f_desc2 = sns.catplot(data = df_list_desc, x = 'category', y = 'mean', col = 'dimension',
                kind = 'bar', palette = {'Ethos Attack':'#BB0000', 'Neutral':'#3B3591', 'Ethos Support':'#026F00', 'Ethos All':'#6C6C6E',
                        'Logos Attack':'#BB0000', ' Neutral':'#3B3591', 'Logos Support':'#026F00', ' Logos All':'#6C6C6E'},
                        height = 4, aspect = 1.4, sharex=False, legend = False )
    f_desc2.set(xlabel = '', ylabel = 'mean word length', ylim = (0, np.max(df_list_desc['mean']+2)))
    f_desc2.set_xticklabels(fontsize = 13)
    for ax in f_desc2.axes.ravel():
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),
            ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')
    st.pyplot(f_desc2)

    st.write("************************************************************************")

    lr_coeff = st.selectbox("Choose a measure of lexical richness",
                ['TTR', 'RTTR', 'CTTR', 'H', 'M'])

    with st.expander('Lexical richness measures'):
        st.write('''
        A first class of indices based on the direct relationship between the number of terms and words (type-token):
        **TTR**: (type-token ratio) (1957, Templin);\n


        TTR with corrections:
        **RTTR**:  (root type-token ratio) (1960, Giraud),
        **CTTR**: (corrected type-token ratio) (1964, Carrol);\n

        A second class of indices has been developed using formulae based on logarithmic function:
        **Herdan H**:  (1960, Herdan),
        **Mass M**:  (1966, Mass).
        ''')
    add_spacelines(1)
    dims_ttr = ['Ethos Attack', 'Ethos Support', 'Logos Attack', 'Logos Support']
    vals_ttr = []

    df_list_log['sentence'] = df_list_log['locution_premise'].astype('str') + " " + df_list_log['locution_conclusion'].astype('str')
    df_list_log_stats = lemmatization(df_list_log, 'sentence') # premise_lemmatized
    #st.write(df_list_log_stats)

    lr_def = str(lr_coeff)

    colttr1, colttr2, colttr3, colttr4 = st.columns(4)
    ttr_ea1 = " ".join( df_list_et[df_list_et['ethos_label'] == 'Ethos Attack']['sentence_lemmatized'].astype('str').str.lower().values )
    ttr_ea1 = ttr_ea1.split()
    df_list_et_targets = df_list_et.Target.dropna().str.lower().values
    ttr_ea1 = list(w for w in ttr_ea1 if not w in df_list_et_targets)
    ttr_ea1_token = len(ttr_ea1)
    ttr_ea1_type = len( set(ttr_ea1) )
    ttr_ea = ttr_lr(t = ttr_ea1_type, n = ttr_ea1_token, definition = lr_def)
    #ttr_ea = round(ttr_ea1_type / ttr_ea1_token, 2) # np.sqrt()  np.sqrt(ttr_ea1_token*2)
    vals_ttr.append(ttr_ea)
    with colttr1:
        st.metric('Lexical Richness of Ethos Attack', ttr_ea)


    ttr_es1 = " ".join( df_list_et[df_list_et['ethos_label'] == 'Ethos Support']['sentence_lemmatized'].astype('str').str.lower().values )
    ttr_es1 = ttr_es1.split()
    ttr_es1 = list(w for w in ttr_es1 if not w in df_list_et_targets)
    ttr_es1_token = len(ttr_es1)
    ttr_es1_type = len( set(ttr_es1) )
    ttr_es = ttr_lr(t = ttr_es1_type, n = ttr_es1_token, definition = lr_def)
    #ttr_es = round(ttr_es1_type / ttr_es1_token, 2)
    vals_ttr.append(ttr_es)
    with colttr2:
        st.metric('Lexical Richness of Ethos Support', ttr_es)

    ttr_ea1 = " ".join( df_list_log_stats[df_list_log_stats['connection'] == 'Logos Attack']['sentence_lemmatized'].astype('str').str.lower().values )
    ttr_ea1 = ttr_ea1.split()
    ttr_ea1 = list(w for w in ttr_ea1 if not w in df_list_et_targets)
    ttr_ea1_token = len(ttr_ea1)
    ttr_ea1_type = len( set(ttr_ea1) )
    ttr_ea = ttr_lr(t = ttr_ea1_type, n = ttr_ea1_token, definition = lr_def)
    #ttr_ea = round(ttr_ea1_type / ttr_ea1_token, 2)
    vals_ttr.append(ttr_ea)
    with colttr3:
        st.metric('Lexical Richness of Logos Attack', ttr_ea)


    ttr_es1 = " ".join( df_list_log_stats[df_list_log_stats['connection'] == 'Logos Support']['sentence_lemmatized'].astype('str').str.lower().values )
    ttr_es1 = ttr_es1.split()
    ttr_es1 = list(w for w in ttr_es1 if not w in df_list_et_targets)
    ttr_es1_token = len(ttr_es1)
    ttr_es1_type = len( set(ttr_es1) )
    ttr_es = ttr_lr(t = ttr_es1_type, n = ttr_es1_token, definition = lr_def)
    #ttr_es = round(ttr_es1_type / ttr_es1_token, 2)
    vals_ttr.append(ttr_es)
    with colttr4:
        st.metric('Lexical Richness of Logos Support', ttr_es)

    df_ttr_stats = pd.DataFrame({'category':dims_ttr, 'ttr ratio':vals_ttr})

    df_ttr_stats.loc[:1, 'dimension'] = 'Ethos'
    df_ttr_stats.loc[2:, 'dimension'] = 'Logos'
    df_ttr_stats = df_ttr_stats.sort_values(by = ['dimension', 'category'])
    val = round(np.max(df_ttr_stats['ttr ratio']) / 5, 2)

    sns.set(font_scale = 1.4, style = 'whitegrid')
    f_desc2 = sns.catplot(data = df_ttr_stats, x = 'category', y = 'ttr ratio', col = 'dimension',
                kind = 'bar', palette = {'Ethos Attack':'#BB0000', ' No Ethos':'#022D96', 'Ethos Support':'#026F00',
                        'Logos Attack':'#BB0000', ' Logos  Rephrase':'#D7A000', 'Logos Support':'#026F00'},
                        height = 4, aspect = 1.4, sharex=False, legend = False)
    f_desc2.set(xlabel = '', ylabel = lr_def,
                ylim = (0, np.max(df_ttr_stats['ttr ratio'])+val ) ) # 'TTR (type-token ratio)'  'CTTR \n(corrected type-token ratio)'
    for ax in f_desc2.axes.ravel():
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),
            ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')
    st.pyplot(f_desc2)
    st.stop()




def StatsLog_compare(df_list, an_type = 'ADU-based'):
    #st.write("#### Sentence Length Analysis")
    add_spacelines(2)
    conn_list = ['Logos Attack', 'Logos Support']
    map_naming = {'attack':'Ethos Attack', 'neutral':' No Ethos', 'support':'Ethos Support',
            'Default Conflict': 'Logos Attack',
            'Default Rephrase' : ' Logos  Rephrase',
            'Default Inference' : 'Logos Support'}
    rhetoric_dims = ['ethos', 'logos']
    df_list_et = pd.concat([df_list[0], df_list[-2]], axis=0, ignore_index = True)

    #df_list_et = df_list[0]
    if not 'neutral' in df_list_et['ethos_label'].unique():
        df_list_et['ethos_label'] = df_list_et['ethos_label'].map(ethos_mapping).map(map_naming)
    #df_list_log = df_list[1]
    df_list_log = pd.concat([df_list[1], df_list[-1]], axis=0, ignore_index = True)
    import re
    df_list_log['locution_conclusion'] = df_list_log.locution_conclusion.apply(lambda x: " ".join( str(x).split(':')[1:]) )
    df_list_log['locution_premise'] = df_list_log.locution_premise.apply(lambda x: " ".join( str(x).split(':')[1:]) )
    df_list_log['sentence'] = df_list_log.locution_premise.astype('str')# + " " + df_list_log.conclusion.astype('str')
    df_list_log_stats = df_list_log.groupby(['corpus', 'locution_premise', 'id_connection', 'connection'])['sentence'].apply(lambda x: " ".join(x)).reset_index()

    if an_type == 'Text-based':
            df_list_log['locution_premise'] = df_list_log['locution_premise'].astype('str')
            df_list_log['locution_conclusion'] = df_list_log['locution_conclusion'].astype('str')

            dfp = df_list_log.groupby(['corpus', 'id_connection', 'connection'])['locution_premise'].apply(lambda x: " ".join(x)).reset_index()
            #dfc = df_list_log.groupby(['id_connection', 'connection'])['locution_conclusion'].apply(lambda x: " ".join(x)).reset_index()
            #dfp = dfp.merge(dfc, on = ['id_connection', 'connection']) #pd.concat([dfp, dfc.iloc[:, -1:]], axis=1) #dfp.merge(dfc, on = ['id_connection', 'connection'])
            dfp = dfp.drop_duplicates()

            dfp['sentence'] = dfp.locution_premise.astype('str')#+ " " + dfp['conclusion'].astype('str')
            import re
            dfp['sentence'] = dfp['sentence'].apply(lambda x: re.sub(r"\W+", " ", str(x)))
            dfp['sentence'] = dfp['sentence'].astype('str').str.lower()
            df_list_log_stats = dfp.copy()

    df_list_log_stats.connection = df_list_log_stats.connection.map(map_naming)


    df_list_et = compnwords(df_list_et, column_name = 'sentence')
    df_list_log_stats = compnwords(df_list_log_stats, column_name = 'sentence')

    cet_desc, c_log_stats_desc = st.columns(2)
    df_list_et_desc = pd.DataFrame(df_list_et[df_list_et.ethos_label.isin(['Ethos Support','Ethos Attack'])].groupby(['corpus', 'ethos_label']).nwords.describe().round(2).iloc[:, 1:])
    #df_list_et_desc = df_list_et_desc.T
    with cet_desc:
        st.write("Sentence Length for **Ethos**: ")
        st.write(df_list_et_desc)
        #st.stop()

    df_list_log_stats_desc = pd.DataFrame(df_list_log_stats[df_list_log_stats.connection.isin(conn_list)].groupby(['corpus', 'connection']).nwords.describe().round(2).iloc[:, 1:])
    #df_list_log_stats_desc = df_list_log_stats_desc.T
    with c_log_stats_desc:
        st.write("Sentence Length for **Logos**: ")
        st.write(df_list_log_stats_desc)


    add_spacelines(1)
    df_list_log_stats_desc = df_list_log_stats_desc.reset_index()
    df_list_et_desc = df_list_et_desc.reset_index()
    #st.write(df_list_log_stats_desc.columns)
    coprs_names1 = df_list_log_stats_desc.corpus.iloc[0]
    coprs_names2 = df_list_log_stats_desc.corpus.iloc[2]
    coprs_names = [coprs_names1, coprs_names2]

    #cstat1, cstat2, cstat3, cstat4 = st.columns(4)
    cstat1, cstat2 = st.columns(2)
    for cname in coprs_names:
        with cstat1:
            st.write(f"**{cname}**")
            le = df_list_et_desc[(df_list_et_desc.corpus == cname) & (df_list_et_desc.ethos_label == 'Ethos Attack')]['mean'].iloc[0]
            ll = df_list_log_stats_desc[(df_list_log_stats_desc.corpus == cname) &\
                    (df_list_log_stats_desc.connection == 'Logos Attack')]['mean'].iloc[0]
            #ll = df_list_log_stats_desc[df_list_log_stats_desc.corpus == cname].loc['mean', 'Logos Attack']
            lrel = round((le *100 / ll)- 100, 2)
            #st.write(le, ll, lrel)
            st.metric('Ethos Attack vs. Logos Attack', f" {le} vs. {ll} ", str(lrel)+'%')

        with cstat2:
            st.write(f"**{cname}**")
            #le = df_list_et_desc.loc['mean', 'Ethos Support']
            #ll = df_list_log_stats_desc.loc['mean', 'Logos Support']
            le = df_list_et_desc[(df_list_et_desc.corpus == cname) & (df_list_et_desc.ethos_label == 'Ethos Support')]['mean'].iloc[0]
            ll = df_list_log_stats_desc[(df_list_log_stats_desc.corpus == cname) &\
                    (df_list_log_stats_desc.connection == 'Logos Support')]['mean'].iloc[0]
            lrel = round((le *100 / ll)- 100, 2)
            st.metric('Ethos Support vs. Logos Support', f" {le} vs. {ll} ", str(lrel)+'%')

    add_spacelines(2)
    df_list_log_stats_desc['dimension'] = 'Logos'
    df_list_log_stats_desc = df_list_log_stats_desc.rename(columns = {'connection':'category'})
    df_list_et_desc = df_list_et_desc.rename(columns = {'ethos_label':'category'})
    df_list_et_desc['dimension'] = 'Ethos'

    df_list_desc = pd.concat( [df_list_log_stats_desc, df_list_et_desc], axis = 0, ignore_index = True )
    #st.write(df_list_desc)
    df_list_desc = df_list_desc.sort_values(by = ['dimension', 'category'])

    sns.set(font_scale = 1.15, style = 'whitegrid')
    f_desc = sns.catplot(data = df_list_desc, x = 'category', y = 'mean', col = 'dimension', row = 'corpus',
                kind = 'bar', palette = {'Ethos Attack':'#BB0000', ' No Ethos':'#022D96', 'Ethos Support':'#026F00',
                        'Logos Attack':'#BB0000', ' Logos  Rephrase':'#D7A000', 'Logos Support':'#026F00'},
                        height = 4, aspect = 1.4, sharex=False)
    f_desc.set(xlabel = '', ylabel = 'mean sentence length', ylim = (0, np.max(df_list_desc['mean']+2)))
    for ax in f_desc.axes.ravel():
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),
            ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')

    plt.tight_layout(pad=2)
    st.pyplot(f_desc)
    st.write("************************************************************************")


    if an_type == 'ADU-based':
        #cet_desc, c_log_stats_desc = st.columns(2)
        #st.write(df_list_et)
        #n_words_all_list = df_list[-1]['premise'].tolist() + df_list[-1]['conclusion'].tolist() + df_list_et.sentence.tolist()
        vals = []
        cats_all = []
        corps_all = []

        for cname in coprs_names:
            n_words_all_list = df_list_log_stats[df_list_log_stats.corpus == cname].sentence.tolist() +\
                        df_list_et[df_list_et.corpus == cname].sentence.tolist()

            n_words_all_series = pd.DataFrame({'text':n_words_all_list})
            n_words_all_series = n_words_all_series.drop_duplicates()
            n_words_all_series['nl'] = n_words_all_series.text.str.split().map(len)
            n_words_all = n_words_all_series['nl'].mean().round(2)

            df_list_et_desc = pd.DataFrame(df_list_et[(df_list_et.ethos_label.isin(['Ethos Support','Ethos Attack'])) & (df_list_et.corpus == cname)].groupby('ethos_label').nwords.describe().round(2).iloc[:, 1:])
            df_list_et_desc = df_list_et_desc.T

            df_list_log_stats_desc = pd.DataFrame(df_list_log_stats[(df_list_log_stats.connection.isin(conn_list)) & (df_list_log_stats.corpus == cname)].groupby('connection').nwords.describe().round(2).iloc[:, 1:])
            df_list_log_stats_desc = df_list_log_stats_desc.T


            add_spacelines(1)
            n_words_all = round(n_words_all, 1)


            le = df_list_et_desc.loc['mean', 'Ethos Attack']
            lrel = round(le -n_words_all, 1)
            vals.append(n_words_all)
            cats_all.append('All')
            corps_all.append(cname)
            vals.append(le)
            cats_all.append('Ethos Attack')
            corps_all.append(cname)
            #st.write(le, ll, lrel)

            le = df_list_et_desc.loc['mean', 'Ethos Support']
            lrel = round(le -n_words_all, 1)
            #lrel = round((le *100 / ll)- 100, 2)
            vals.append(le)
            cats_all.append('Ethos Support')
            corps_all.append(cname)

            le = df_list_log_stats_desc.loc['mean', " Logos Attack"]
            lrel = round(le -n_words_all, 1)
            #st.write(le, ll, lrel)
            vals.append(le)
            cats_all.append(" Logos Attack")
            corps_all.append(cname)

            le = df_list_log_stats_desc.loc['mean', " Logos Support"]
            #ll = df_list_log_stats_desc.loc['mean', 'Logos Support']
            lrel = round(le -n_words_all, 1)
            vals.append(le)
            cats_all.append( " Logos Support")
            corps_all.append(cname)


        df_list_desc = pd.DataFrame([corps_all, cats_all, vals]).T
        df_list_desc.columns = ['corpus', 'category', 'mean']
        #add_spacelines(1)
        #st.write(df_list_desc)
        #st.stop()

        sns.set(font_scale = 1, style = 'whitegrid')
        f_desc = sns.catplot(data = df_list_desc, x = 'category', y = 'mean', col = 'corpus',
                    kind = 'bar', palette = {'Ethos Attack':'#BB0000', 'All':'#022D96', 'Ethos Support':'#026F00',
                            'Logos Attack':'#BB0000', ' Logos  Rephrase':'#D7A000', 'Logos Support':'#026F00'},
                            aspect = 1.65, sharex=False, height=4)
        f_desc.set(xlabel = '', ylabel = 'mean sentence length', ylim = (0, np.max(df_list_desc['mean']+1)))
        for ax in f_desc.axes.ravel():
            for p in ax.patches:
                ax.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')

        plt.tight_layout(pad=2)
        _, c_log_stats_desc_all, _ = st.columns([1,20,1])
        with c_log_stats_desc_all:
            st.pyplot(f_desc)
        st.write("************************************************************************")

    st.stop()






def OddsRatioLog_compare(df_list, selected_rhet_dim, an_type = 'ADU-based'):
    rhetoric_dims = ['ethos', 'logos']

    if selected_rhet_dim == 'ethos_label':
        df = df_list[0]
        if not 'neutral' in df['ethos_label'].unique():
            df['ethos_label'] = df['ethos_label'].map(ethos_mapping)
        df['sentence_lemmatized'] = df['sentence_lemmatized'].astype('str').str.lower().str.replace('ahould', 'should')

    if selected_rhet_dim == 'logos_label':
        df = df_list[1]
        #st.write(df)
        df['logos_label'] = df.connection.map({
                            'Default Conflict': 'attack',
                            'Default Rephrase' : 'Rephrase',
                            'Default Inference' : 'support'
        }).fillna('other')

        #df['sentence'] = df.premise
        df['sentence_lemmatized'] = df['premise'].astype('str') + " " + df['conclusion'].astype('str')
        if an_type != 'Text-based':
            df = lemmatization(df, 'sentence_lemmatized', name_column = True)
            df = clean_text(df, 'sentence_lemmatized', text_column_name = "sentence_lemmatized")
            df['sentence_lemmatized'] = df['sentence_lemmatized'].astype('str').str.lower().str.replace('ahould', 'should')

        elif an_type == 'Text-based':
            df['premise'] = df['premise'].astype('str')
            df['conclusion'] = df['conclusion'].astype('str')
            dfp = df.groupby(['id_connection', 'logos_label'])['premise'].apply(lambda x: " ".join(x)).reset_index()
            dfc = df.groupby(['id_connection', 'logos_label'])['conclusion'].apply(lambda x: " ".join(x)).reset_index()
            dfp = dfp.merge(dfc, on = ['id_connection', 'logos_label']) #pd.concat([dfp, dfc.iloc[:, -1:]], axis=1) #dfp.merge(dfc, on = ['id_connection', 'connection'])
            dfp = dfp.drop_duplicates()

            dfp['sentence_lemmatized'] = dfp.premise.astype('str')+ " " + dfc['conclusion'].astype('str')
            #st.write(dfp)
            import re
            dfp['sentence_lemmatized'] = dfp['sentence_lemmatized'].apply(lambda x: re.sub(r"\W+", " ", str(x)))
            dfp = lemmatization(dfp, 'sentence_lemmatized', name_column = True)
            df = dfp.copy()
            df = clean_text(df, 'sentence_lemmatized', text_column_name = "sentence_lemmatized")
            df['sentence_lemmatized'] = df['sentence_lemmatized'].astype('str').str.lower().str.replace('ahould', 'should')
        #if not 'sentence_lemmatized' in df.columns:
            #df = lemmatization(df, 'sentence')


    ddmsc = ['support', 'attack']
    if selected_rhet_dim == 'pathos_label':
        ddmsc = ['positive', 'negative']

    odds_list_of_dicts = []
    effect_list_of_dicts = []
    count_list_of_dicts = []
    # 1 vs rest
    #num = np.floor( len(df) / 10 )
    for ddmsc1 in ddmsc:
        dict_1vsall_percent = {}
        dict_1vsall_effect_size = {}
        dict_1vsall_count = {}
        #all100popular = Counter(" ".join( df.lemmatized.values ).split()).most_common(100)
        #all100popular = list(w[0] for w in all100popular)

        ddmsc1w = " ".join( df[df[selected_rhet_dim] == ddmsc1].sentence_lemmatized.fillna('').astype('str').values ).split()
        c = len(ddmsc1w)
        #ddmsc1w = list(w for w in ddmsc1w if not w in all100popular)
        ddmsc1w = Counter(ddmsc1w).most_common() # num
        ddmsc1w = [w for w in ddmsc1w if w[1] >= 3 ]

        #if ddmsc1 in ['positive', 'support']:
            #ddmsc1w = [w for w in ddmsc1w if w[1] >= 3 ]
        #else:
            #ddmsc1w = [w for w in ddmsc1w if w[1] > 3 ]

        ddmsc1w_word = dict(ddmsc1w)

        ddmsc2w = " ".join( df[df[selected_rhet_dim] != ddmsc1].sentence_lemmatized.fillna('').astype('str').values ).split()
        d = len(ddmsc2w)
        #ddmsc2w = list(w for w in ddmsc2w if not w in all100popular)
        ddmsc2w = Counter(ddmsc2w).most_common()
        ddmsc2w_word = dict(ddmsc2w)


        ddmsc1w_words = list( ddmsc1w_word.keys() )
        for n, dim in enumerate( ddmsc1w_words ):

            a = ddmsc1w_word[dim]
            try:
                b = ddmsc2w_word[dim]
            except:
                b = 0.5

            ca = c-a
            bd = d-b

            E1 = c*(a+b) / (c+d)
            E2 = d*(a+b) / (c+d)

            g2 = 2*((a*np.log(a/E1)) + (b* np.log(b/E2)))
            g2 = round(g2, 2)

            odds = round( (a*(d-b)) / (b*(c-a)), 2)

            if odds > 1 and len(dim) > 2:
                if g2 > 10.83:
                    #print(f"{dim, g2, odds} ***p < 0.001 ")
                    dict_1vsall_percent[dim] = odds
                    dict_1vsall_effect_size[dim] = 0.001
                    dict_1vsall_count[dim] = a
                elif g2 > 6.63:
                    #print(f"{dim, g2, odds} **p < 0.01 ")
                    dict_1vsall_percent[dim] = odds
                    dict_1vsall_effect_size[dim] = 0.01
                    dict_1vsall_count[dim] = a
                elif g2 > 3.84:
                    #print(f"{dim, g2, odds} *p < 0.05 ")
                    dict_1vsall_percent[dim] = odds
                    dict_1vsall_effect_size[dim] = 0.05
                    dict_1vsall_count[dim] = a
        #print(dict(sorted(dict_1vsall_percent.items(), key=lambda item: item[1])))
        odds_list_of_dicts.append(dict_1vsall_percent)
        effect_list_of_dicts.append(dict_1vsall_effect_size)
        count_list_of_dicts.append(dict_1vsall_count)

    df_odds_pos = pd.DataFrame({
                'word':odds_list_of_dicts[0].keys(),
                'odds':odds_list_of_dicts[0].values(),
                'effect_size_p':effect_list_of_dicts[0].values(),
                'frequency': count_list_of_dicts[0].values(),
    })
    df_odds_pos['category'] = ddmsc[0]
    df_odds_neg = pd.DataFrame({
                'word':odds_list_of_dicts[1].keys(),
                'odds':odds_list_of_dicts[1].values(),
                'effect_size_p':effect_list_of_dicts[1].values(),
                'frequency': count_list_of_dicts[1].values(),

    })
    df_odds_neg['category'] = ddmsc[1]
    df_odds_neg = df_odds_neg[df_odds_neg.word != 'bewp']
    df_odds_neg = df_odds_neg.sort_values(by = ['odds'], ascending = False)
    df_odds_pos = df_odds_pos.sort_values(by = ['odds'], ascending = False)


    df_odds_neg = transform_text(df_odds_neg, 'word')
    df_odds_pos = transform_text(df_odds_pos, 'word')
    pos_list = ['NOUN', 'VERB', 'NUM', 'PROPN', 'ADJ', 'ADV']
    df_odds_neg = df_odds_neg[df_odds_neg.POS_tags.isin(pos_list)]
    df_odds_pos = df_odds_pos[df_odds_pos.POS_tags.isin(pos_list)]
    df_odds_neg = df_odds_neg.reset_index(drop=True)
    df_odds_pos = df_odds_pos.reset_index(drop=True)
    df_odds_pos.index += 1
    df_odds_neg.index += 1

    df_odds_pos_tags_summ = df_odds_pos.POS_tags.value_counts(normalize = True).round(2)*100
    df_odds_neg_tags_summ = df_odds_neg.POS_tags.value_counts(normalize = True).round(2)*100
    df_odds_pos_tags_summ = df_odds_pos_tags_summ.reset_index()
    df_odds_pos_tags_summ.columns = ['POS_tags', 'percentage']

    df_odds_neg_tags_summ = df_odds_neg_tags_summ.reset_index()
    df_odds_neg_tags_summ.columns = ['POS_tags', 'percentage']

    df_odds_pos_tags_summ = df_odds_pos_tags_summ[df_odds_pos_tags_summ.percentage > 1]
    df_odds_neg_tags_summ = df_odds_neg_tags_summ[df_odds_neg_tags_summ.percentage > 1]

    oddpos_c, oddneg_c = st.columns(2)
    dimm = selected_rhet_dim.split("_")[0]
    with oddpos_c:
        st.write(f'Number of {dimm} {df_odds_pos.category.iloc[0]} words: {len(df_odds_pos)} ')
        st.dataframe(df_odds_pos)
        add_spacelines(1)
        st.dataframe(df_odds_pos_tags_summ)
        add_spacelines(1)

    with oddneg_c:
        st.write(f'Number of {dimm} {df_odds_neg.category.iloc[0]} words: {len(df_odds_neg)} ')
        st.dataframe(df_odds_neg)
        add_spacelines(1)
        st.dataframe(df_odds_neg_tags_summ)
        add_spacelines(1)







def OddsRatioLog(df_list, an_type = 'ADU-based'):
    st.write("### Lexical Analysis - Odds Ratio")
    add_spacelines(2)
    rhetoric_dims = ['ethos', 'logos']
    selected_rhet_dim = st.selectbox("Choose a rhetoric strategy for analysis", rhetoric_dims, index=0)
    selected_rhet_dim = selected_rhet_dim.replace('ethos', 'ethos_label').replace('logos', 'logos_label')
    add_spacelines(1)

    if selected_rhet_dim == 'ethos_label':
        df = df_list[0]
        if not 'neutral' in df['ethos_label'].unique():
            df['ethos_label'] = df['ethos_label'].map(ethos_mapping)

        df['sentence_lemmatized'] = df['sentence_lemmatized'].astype('str').str.lower().str.replace('ahould', 'should')
    if selected_rhet_dim == 'logos_label':
        df = df_list[1]
        #st.write(df)
        df['logos_label'] = df.connection.map({
                            'Default Conflict': 'attack',
                            'Default Rephrase' : 'Rephrase',
                            'Default Inference' : 'support'
        }).fillna('other')

        #df['sentence'] = df.premise
        df['sentence_lemmatized'] = df['premise'].astype('str') + " " + df['conclusion'].astype('str')
        if an_type != 'Text-based':
            df = lemmatization(df, 'sentence_lemmatized', name_column = True)
            df = clean_text(df, 'sentence_lemmatized', text_column_name = "sentence_lemmatized")
            df['sentence_lemmatized'] = df['sentence_lemmatized'].astype('str').str.lower().str.replace('ahould', 'should')

        elif an_type == 'Text-based':
            df['premise'] = df['premise'].astype('str')
            df['conclusion'] = df['conclusion'].astype('str')
            dfp = df.groupby(['id_connection', 'logos_label'])['premise'].apply(lambda x: " ".join(x)).reset_index()
            dfc = df.groupby(['id_connection', 'logos_label'])['conclusion'].apply(lambda x: " ".join(x)).reset_index()
            dfp = dfp.merge(dfc, on = ['id_connection', 'logos_label']) #pd.concat([dfp, dfc.iloc[:, -1:]], axis=1) #dfp.merge(dfc, on = ['id_connection', 'connection'])
            dfp = dfp.drop_duplicates()

            dfp['sentence_lemmatized'] = dfp.premise.astype('str')+ " " + dfc['conclusion'].astype('str')
            #st.write(dfp)
            import re
            dfp['sentence_lemmatized'] = dfp['sentence_lemmatized'].apply(lambda x: re.sub(r"\W+", " ", str(x)))
            dfp = lemmatization(dfp, 'sentence_lemmatized', name_column = True)
            df = dfp.copy()
            df = clean_text(df, 'sentence_lemmatized', text_column_name = "sentence_lemmatized")
            df['sentence_lemmatized'] = df['sentence_lemmatized'].astype('str').str.lower().str.replace('ahould', 'should')
        #if not 'sentence_lemmatized' in df.columns:
            #df = lemmatization(df, 'sentence')


    ddmsc = ['support', 'attack']
    if selected_rhet_dim == 'pathos_label':
        ddmsc = ['positive', 'negative']

    odds_list_of_dicts = []
    effect_list_of_dicts = []
    count_list_of_dicts = []
    # 1 vs rest
    #num = np.floor( len(df) / 10 )
    for ddmsc1 in ddmsc:
        dict_1vsall_percent = {}
        dict_1vsall_effect_size = {}
        dict_1vsall_count = {}
        #all100popular = Counter(" ".join( df.lemmatized.values ).split()).most_common(100)
        #all100popular = list(w[0] for w in all100popular)

        ddmsc1w = " ".join( df[df[selected_rhet_dim] == ddmsc1].sentence_lemmatized.fillna('').astype('str').values ).split()
        c = len(ddmsc1w)
        #ddmsc1w = list(w for w in ddmsc1w if not w in all100popular)
        ddmsc1w = Counter(ddmsc1w).most_common() # num
        ddmsc1w = [w for w in ddmsc1w if w[1] > 3 ]

        #if ddmsc1 in ['positive', 'support']:
            #ddmsc1w = [w for w in ddmsc1w if w[1] >= 3 ]
        #else:
            #ddmsc1w = [w for w in ddmsc1w if w[1] > 3 ]

        ddmsc1w_word = dict(ddmsc1w)

        ddmsc2w = " ".join( df[df[selected_rhet_dim] != ddmsc1].sentence_lemmatized.fillna('').astype('str').values ).split()
        d = len(ddmsc2w)
        #ddmsc2w = list(w for w in ddmsc2w if not w in all100popular)
        ddmsc2w = Counter(ddmsc2w).most_common()
        ddmsc2w_word = dict(ddmsc2w)


        ddmsc1w_words = list( ddmsc1w_word.keys() )
        for n, dim in enumerate( ddmsc1w_words ):

            a = ddmsc1w_word[dim]
            try:
                b = ddmsc2w_word[dim]
            except:
                b = 0.5

            ca = c-a
            bd = d-b

            E1 = c*(a+b) / (c+d)
            E2 = d*(a+b) / (c+d)

            g2 = 2*((a*np.log(a/E1)) + (b* np.log(b/E2)))
            g2 = round(g2, 2)

            odds = round( (a*(d-b)) / (b*(c-a)), 2)

            if odds > 1 and len(dim) > 2:
                if g2 > 10.83:
                    #print(f"{dim, g2, odds} ***p < 0.001 ")
                    dict_1vsall_percent[dim] = odds
                    dict_1vsall_effect_size[dim] = 0.001
                    dict_1vsall_count[dim] = a
                elif g2 > 6.63:
                    #print(f"{dim, g2, odds} **p < 0.01 ")
                    dict_1vsall_percent[dim] = odds
                    dict_1vsall_effect_size[dim] = 0.01
                    dict_1vsall_count[dim] = a
                elif g2 > 3.84:
                    #print(f"{dim, g2, odds} *p < 0.05 ")
                    dict_1vsall_percent[dim] = odds
                    dict_1vsall_effect_size[dim] = 0.05
                    dict_1vsall_count[dim] = a
        #print(dict(sorted(dict_1vsall_percent.items(), key=lambda item: item[1])))
        odds_list_of_dicts.append(dict_1vsall_percent)
        effect_list_of_dicts.append(dict_1vsall_effect_size)
        count_list_of_dicts.append(dict_1vsall_count)

    df_odds_pos = pd.DataFrame({
                'word':odds_list_of_dicts[0].keys(),
                'odds':odds_list_of_dicts[0].values(),
                'effect_size_p':effect_list_of_dicts[0].values(),
                'frequency': count_list_of_dicts[0].values(),
    })
    df_odds_pos['category'] = ddmsc[0]
    df_odds_neg = pd.DataFrame({
                'word':odds_list_of_dicts[1].keys(),
                'odds':odds_list_of_dicts[1].values(),
                'effect_size_p':effect_list_of_dicts[1].values(),
                'frequency': count_list_of_dicts[1].values(),

    })
    df_odds_neg['category'] = ddmsc[1]
    df_odds_neg = df_odds_neg[df_odds_neg.word != 'bewp']
    df_odds_neg = df_odds_neg.sort_values(by = ['odds'], ascending = False)
    df_odds_pos = df_odds_pos.sort_values(by = ['odds'], ascending = False)


    df_odds_neg = transform_text(df_odds_neg, 'word')
    df_odds_pos = transform_text(df_odds_pos, 'word')
    pos_list = ['NOUN', 'VERB', 'NUM', 'PROPN', 'ADJ', 'ADV']
    df_odds_neg = df_odds_neg[df_odds_neg.POS_tags.isin(pos_list)]
    df_odds_pos = df_odds_pos[df_odds_pos.POS_tags.isin(pos_list)]
    df_odds_neg = df_odds_neg.reset_index(drop=True)
    df_odds_pos = df_odds_pos.reset_index(drop=True)
    df_odds_pos.index += 1
    df_odds_neg.index += 1
    df_odds_neg['abusive'] = df_odds_neg.word.apply(lambda x: " ".join( set(x.lower().split()).intersection(abus_words)  ))
    df_odds_neg['abusive'] = np.where( df_odds_neg['abusive'].fillna('').astype('str').map(len) > 1 , 'abusive', 'non-abusive' )
    df_odds_pos['abusive'] = df_odds_pos.word.apply(lambda x: " ".join( set(x.lower().split()).intersection(abus_words)  ))
    df_odds_pos['abusive'] = np.where( df_odds_pos['abusive'].fillna('').astype('str').map(len) > 1, 'abusive', 'non-abusive' )


    df_odds_pos_tags_summ = df_odds_pos.POS_tags.value_counts(normalize = True).round(2)*100
    df_odds_neg_tags_summ = df_odds_neg.POS_tags.value_counts(normalize = True).round(2)*100
    df_odds_pos_tags_summ = df_odds_pos_tags_summ.reset_index()
    df_odds_pos_tags_summ.columns = ['POS_tags', 'percentage']
    df_odds_neg_tags_summ = df_odds_neg_tags_summ.reset_index()
    df_odds_neg_tags_summ.columns = ['POS_tags', 'percentage']

    df_odds_pos_tags_summ = df_odds_pos_tags_summ[df_odds_pos_tags_summ.percentage > 1]
    df_odds_neg_tags_summ = df_odds_neg_tags_summ[df_odds_neg_tags_summ.percentage > 1]

    df_odds_pos_abs= df_odds_pos.abusive.value_counts(normalize = True).round(3)*100
    df_odds_neg_abs = df_odds_neg.abusive.value_counts(normalize = True).round(3)*100
    df_odds_pos_abs = df_odds_pos_abs.reset_index()
    df_odds_pos_abs.columns = ['abusive', 'percentage']
    df_odds_neg_abs = df_odds_neg_abs.reset_index()
    df_odds_neg_abs.columns = ['abusive', 'percentage']

    df_odds_pos_words = set(df_odds_pos.word.values)
    df_odds_neg_words = set(df_odds_neg.word.values)

    df_odds = pd.concat( [df_odds_pos, df_odds_neg], axis = 0, ignore_index = True )
    df_odds = df_odds.sort_values(by = ['category', 'odds'], ascending = False)
    df['odds_words_'+df_odds_pos.category.iloc[0]] = df.sentence_lemmatized.apply(lambda x: " ".join( set(x.split()).intersection(df_odds_pos_words) ))
    df['odds_words_'+df_odds_neg.category.iloc[0]] = df.sentence_lemmatized.apply(lambda x: " ".join( set(x.split()).intersection(df_odds_neg_words) ))


    tab_odd, tab_pos, tab_abuse = st.tabs(['Odds', 'POS', 'Abusiveness'])
    with tab_odd:
        oddpos_c, oddneg_c = st.columns(2, gap = 'large')
        if selected_rhet_dim == 'ethos_label':
            cols_odds = ['source', 'sentence', 'ethos_label', 'Target',
                     'odds_words_'+df_odds_pos.category.iloc[0], 'odds_words_'+df_odds_neg.category.iloc[0]]

        elif selected_rhet_dim == 'pathos_label':
            cols_odds = ['source', 'sentence', 'pathos_label', 'Target',
                     'odds_words_'+df_odds_pos.category.iloc[0], 'odds_words_'+df_odds_neg.category.iloc[0]]

        elif selected_rhet_dim == 'logos_label':
            cols_odds = ['conclusion', 'premise', 'logos_label',
                     'odds_words_'+df_odds_pos.category.iloc[0], 'odds_words_'+df_odds_neg.category.iloc[0]]

        dimm = selected_rhet_dim.split("_")[0]
        with oddpos_c:
            st.write(f'Number of {dimm} {df_odds_pos.category.iloc[0]} words: {len(df_odds_pos)} ')
            st.dataframe(df_odds_pos)
            #add_spacelines(1)
            #st.dataframe(df_odds_pos_tags_summ)
            add_spacelines(1)
            st.write(f'Cases with **{df_odds_pos.category.iloc[0]}** words:')
            dfp = df[ df['odds_words_'+df_odds_pos.category.iloc[0]].str.split().map(len) >= 1 ][cols_odds]
            dfp = dfp[dfp[selected_rhet_dim].isin(['support', 'positive'])].drop_duplicates().reset_index(drop=True)
            st.dataframe(dfp) # .set_index('source')

        with oddneg_c:
            st.write(f'Number of {dimm} {df_odds_neg.category.iloc[0]} words: {len(df_odds_neg)} ')
            st.dataframe(df_odds_neg)
            #add_spacelines(1)
            #st.dataframe(df_odds_neg_tags_summ)
            add_spacelines(1)
            st.write(f'Cases with **{df_odds_neg.category.iloc[0]}** words:')
            dfn = df[ df['odds_words_'+df_odds_neg.category.iloc[0]].str.split().map(len) >= 1 ][cols_odds]
            dfn = dfn[dfn[selected_rhet_dim].isin(['attack', 'negative'])].drop_duplicates().reset_index(drop=True)
            st.dataframe(dfn) # .set_index('source')

    with tab_pos:
        sns.set(font_scale = 1.25, style = 'whitegrid')
        df_odds_pos_tags_summ['category'] = df_odds_pos.category.iloc[0]
        df_odds_neg_tags_summ['category'] = df_odds_neg.category.iloc[0]
        df_odds_pos = pd.concat([df_odds_pos_tags_summ, df_odds_neg_tags_summ], axis = 0, ignore_index=True)
        ffp = sns.catplot(kind='bar', data = df_odds_pos,
        y = 'POS_tags', x = 'percentage', hue = 'POS_tags', aspect = 1.3, height = 5, dodge=False,
        legend = False, col = 'category')
        ffp.set(ylabel = '')
        plt.tight_layout(w_pad=3)
        st.pyplot(ffp)
        add_spacelines(1)

        oddpos_cpos, oddneg_cpos = st.columns(2, gap = 'large')
        with oddpos_cpos:
            st.write(f'POS analysis of **{dimm} {df_odds_pos.category.iloc[0]}** words')
            add_spacelines(1)
            st.dataframe(df_odds_pos_tags_summ)
            add_spacelines(1)

        with oddneg_cpos:
            st.write(f'POS analysis of **{dimm} {df_odds_neg.category.iloc[0]}** words')
            add_spacelines(1)
            st.dataframe(df_odds_neg_tags_summ)
            add_spacelines(1)


    with tab_abuse:
        sns.set(font_scale = 1, style = 'whitegrid')
        df_odds_pos_abs['category'] = df_odds_pos.category.iloc[0]
        df_odds_neg_abs['category'] = df_odds_neg.category.iloc[0]
        df_odds_abs = pd.concat([df_odds_pos_abs, df_odds_neg_abs], axis = 0, ignore_index=True)
        ffp = sns.catplot(kind='bar', data = df_odds_abs,
        y = 'abusive', x = 'percentage', hue = 'abusive', aspect = 1.3, height = 3,dodge=False,
        palette = {'abusive':'darkred', 'non-abusive':'grey'}, legend = False, col = 'category')
        ffp.set(ylabel = '')
        plt.tight_layout(w_pad=3)
        st.pyplot(ffp)

        oddpos_cab, oddneg_cab = st.columns(2, gap = 'large')
        with oddpos_cab:
            st.write(f'Abusiveness analysis of **{dimm} {df_odds_pos.category.iloc[0]}** words')
            add_spacelines(1)
            st.dataframe(df_odds_pos_abs)
            add_spacelines(1)
            #ffp = sns.catplot(kind='bar', data = df_odds_pos_abs,
            #y = 'abusive', x = 'percentage', hue = 'abusive', aspect = 1.3, height = 3,dodge=False,
            #palette = {'abusive':'darkred', 'non-abusive':'grey'}, legend = False)
            #ffp.set(ylabel = '', title = f'Abusiveness of {dimm} {df_odds_pos.category.iloc[0]} words')
            #st.pyplot(ffp)
            add_spacelines(1)
            if df_odds_pos_abs.shape[0] > 1:
                st.write(df_odds_pos[df_odds_pos['abusive'] == 'abusive'])

        with oddneg_cab:
            st.write(f'Abusiveness analysis of **{dimm} {df_odds_neg.category.iloc[0]}** words')
            add_spacelines(1)
            st.dataframe(df_odds_neg_abs)
            add_spacelines(1)
            #ffn = sns.catplot(kind='bar', data = df_odds_neg_abs,
            #y = 'abusive', x = 'percentage', hue = 'abusive', aspect = 1.3, height = 3, dodge=False,
            #palette = {'abusive':'darkred', 'non-abusive':'grey'}, legend = False)
            #ffn.set(ylabel = '', title = f'Abusiveness of {dimm} {df_odds_neg.category.iloc[0]} words')
            #st.pyplot(ffn)
            add_spacelines(1)
            if df_odds_neg_abs.shape[0] > 1:
                st.write(df_odds_neg[df_odds_neg['abusive'] == 'abusive'])



@st.cache_data
def assignprons(data, col_take = 'sentence'):
    df = data.copy()
    prons = {'he', 'she', 'you', 'his', 'him', 'her', 'hers', 'your',
            'yours', 'herself', 'himself', 'yourself'}
    prons3rd = {'they', 'their', 'theirs', 'them', 'themselves'}
    prons_verbs = {'are', 'were', 'have been', "weren't", "aren't", "haven't"}
    import re
    df[col_take] = df[col_take].apply(lambda x: re.sub(r"\W+", " ", str(x)))
    df['pronouns_singular'] = df[col_take].apply(lambda x: " ".join( set(x.split()).intersection(prons) ) )
    df['pronouns_plural'] = df[col_take].apply(lambda x: " ".join( set(x.split()).intersection(prons3rd) ) )
    df['plural_TOBE_verbs'] = df[col_take].apply(lambda x: " ".join( set(x.split()).intersection(prons_verbs) ) )
    return df

@st.cache_data
def count_categories(dataframe, categories_column, spliting = False, prefix_txt = 'pos'):
  if spliting:
    dataframe[categories_column] = dataframe[categories_column].str.split()

  dataframe["merge_indx"] = range(0, len(dataframe))
  from collections import Counter
  dataframe = pd.merge(dataframe, pd.DataFrame([Counter(x) for x in dataframe[categories_column]]).fillna(0).astype(int).add_prefix(str(prefix_txt)), how='left', left_on="merge_indx", right_index=True)
  dataframe.drop(["merge_indx"], axis=1, inplace=True)
  if spliting:
    dataframe[categories_column] = dataframe[categories_column].apply(lambda x: " ".join(x))
  return dataframe


def PronousLoP(df_list):
    st.write("### Language of Polarization Cues")
    add_spacelines(2)

    radio_prons_cat_dict = {
            'singular pronouns':'pronouns_singular',
            'plural pronouns':'pronouns_plural',
            'plural TO BE verbs':'plural_TOBE_verbs'}
    radio_prons_cat = st.multiselect('Choose a method of searching for pronouns',
                    ['singular pronouns',
                    'plural pronouns'], ['plural pronouns'])
    df = df_list[0]

    if not 'neutral' in df['ethos_label'].unique():
        df['ethos_label'] = df['ethos_label'].map(ethos_mapping)
    if not 'negative' in df['pathos_label'].unique():
        df['pathos_label'] = df['pathos_label'].map(valence_mapping)

    df = clean_text(df, 'sentence', text_column_name = "sentence_lemmatized")
    df = assignprons(data = df, col_take = 'sentence_lemmatized')
    cols_odds1 = ['source', 'sentence_lemmatized',
                'pronouns_plural', 'plural_TOBE_verbs', 'pronouns_singular',
                'ethos_label', 'pathos_label', 'Target']

    df_pron = df.copy()
    df_pron_col0 = df_pron.columns
    for i, c in enumerate(radio_prons_cat):
        col_i = radio_prons_cat_dict[c]
        #df_pron = df_pron[ ( df_pron[col_i].str.split().map(len)>0 ) ] # | ( df.pronouns.str.split().map(len)>0 )
        df_pron = count_categories(dataframe = df_pron, categories_column = col_i, spliting = True, prefix_txt = '')

    df_pron['No_pronouns_plural'] = df_pron.pronouns_plural.str.split().map(len)
    df_pron['No_pronouns_singular'] = df_pron.pronouns_singular.str.split().map(len)
    df_pron_tab = df_pron.copy()
    for i, c in enumerate(radio_prons_cat):
        col_i = radio_prons_cat_dict[c]
        #df_pron = df_pron[ ( df_pron[col_i].str.split().map(len)>0 ) ] # | ( df.pronouns.str.split().map(len)>0 )
        df_pron_tab = df_pron_tab[ df_pron_tab[col_i].str.split().map(len) > 0]

    st.write(df_pron_tab[cols_odds1].set_index('source'))
    #st.write(df_pron.shape, df_pron_tab.shape)
    df_pron_plot_avg = df_pron.groupby('ethos_label', as_index = False)[['No_pronouns_plural', 'No_pronouns_singular']].mean()
    df_pron_plot_avg.No_pronouns_plural = df_pron_plot_avg.No_pronouns_plural.astype('float').round(3)*100
    df_pron_plot_avg.No_pronouns_singular = df_pron_plot_avg.No_pronouns_singular.astype('float').round(3)*100
    df_pron_plot_avg = df_pron_plot_avg.rename(columns = {'ethos_label':'ethos'})
    df_pron_plot_avg_melt = df_pron_plot_avg.melt('ethos')
    df_pron_plot_avg_melt['variable'] = df_pron_plot_avg_melt['variable'].str.replace("No_", "")
    #st.write(df_pron_plot_avg_melt)

    #st.stop()
    df_pron_col1 = df_pron.columns[:-2]
    df_pron_col2 = list( set(df_pron_col1).difference(set(df_pron_col0)) )
    df_pron_col2.extend(['ethos_label', 'pathos_label'])
    df_pron_plot = df_pron[df_pron_col2]
    #df_pron_plot_melt = df_pron_plot.melt(['ethos_label', 'pathos_label'])
    df_pron_plot_melt2 = df_pron_plot.groupby(['ethos_label', 'pathos_label'])[df_pron_col2[:-2]].mean().round(3)*100
    df_pron_plot_melt2 = df_pron_plot_melt2.reset_index()

    df_pron_plot_melt = df_pron_plot_melt2.melt(['ethos_label', 'pathos_label'])
    df_pron_plot_melt = df_pron_plot_melt.rename(columns = {'ethos_label':'ethos', 'pathos_label':'pathos'})
    max_val  = df_pron_plot_melt['value'].max()
    #st.write(df_pron_plot_melt2)
    add_spacelines(2)


    sns.set(font_scale=1.2, style='whitegrid')
    g = sns.catplot(data = df_pron_plot_melt, kind = 'bar', y = 'variable', x ='value',
        col = 'ethos', hue = 'variable',
        sharey=False, dodge=False, palette = 'hsv_r', legend=False, aspect = 1.2)
    plt.tight_layout(pad=3)
    #sns.move_legend(g, loc = 'lower left', bbox_to_anchor = (0.32, 0.98), ncol = 4, title = '')
    g.set(xticks = np.arange(0, max_val+31, 10), ylabel = 'pronoun', xlabel = 'percentage')
    plt.suptitle('Ethos')
    for ax in g.axes.flatten():
        ax.tick_params(labelbottom=True)
    plt.show()
    st.write("**Ethos**")
    st.pyplot(g)

    g = sns.catplot(data = df_pron_plot_avg_melt, kind = 'bar', y = 'variable', x ='value',
        col = 'ethos', hue = 'variable',
        sharey=False, dodge=False, palette = 'hsv_r', legend=False, aspect = 1.2)
    plt.tight_layout(pad=2)
    g.set(xticks = np.arange(0, 101, 20), xlabel = 'percentage', ylabel = 'pronouns type')
    st.pyplot(g)

    add_spacelines(2)

    sns.set(font_scale=1.2, style='whitegrid')
    g = sns.catplot(data = df_pron_plot_melt, kind = 'bar', y = 'variable', x ='value',
        col = 'pathos', hue = 'variable',
        sharey=False, dodge=False, palette = 'hsv_r', legend=False, aspect = 1.2)
    plt.tight_layout(pad=3)
    #sns.move_legend(g, loc = 'lower left', bbox_to_anchor = (0.32, 0.98), ncol = 4, title = '')
    plt.suptitle('Pathos')
    g.set(xticks = np.arange(0, max_val+31, 10), ylabel = 'pronoun', xlabel = 'percentage')
    for ax in g.axes.flatten():
        ax.tick_params(labelbottom=True)
    plt.show()
    st.write("**Pathos**")
    st.pyplot(g)

    df_pron_plot_avg = df_pron.groupby('pathos_label', as_index = False)[['No_pronouns_plural', 'No_pronouns_singular']].mean().round(3)
    df_pron_plot_avg.No_pronouns_plural = df_pron_plot_avg.No_pronouns_plural.astype('float').round(3)*100
    df_pron_plot_avg.No_pronouns_singular = df_pron_plot_avg.No_pronouns_singular.astype('float').round(3)*100
    df_pron_plot_avg = df_pron_plot_avg.rename(columns = {'pathos_label':'pathos'})
    df_pron_plot_avg_melt = df_pron_plot_avg.melt('pathos')
    df_pron_plot_avg_melt['variable'] = df_pron_plot_avg_melt['variable'].str.replace("No_", "")

    g = sns.catplot(data = df_pron_plot_avg_melt, kind = 'bar', y = 'variable', x ='value',
        col = 'pathos', hue = 'variable',
        sharey=False, dodge=False, palette = 'hsv_r', legend=False, aspect = 1.2)
    plt.tight_layout(pad=2)
    g.set(xticks = np.arange(0, 101, 20), xlabel = 'percentage', ylabel = 'pronouns type')
    st.pyplot(g)


    add_spacelines(2)

    sns.set(font_scale=1.75, style='whitegrid')
    g = sns.catplot(data = df_pron_plot_melt, kind = 'bar', y = 'variable', x ='value',
        col = 'ethos', row = 'pathos', hue = 'variable',
        sharey=False, dodge=False, palette = 'hsv_r', legend=False, aspect = 1.4)
    plt.tight_layout(pad=3)
    #sns.move_legend(g, loc = 'lower left', bbox_to_anchor = (0.32, 0.98), ncol = 4, title = '')
    g.set(xticks = np.arange(0, 101, 20), ylabel = 'pronoun', xlabel = 'percentage')
    for ax in g.axes.flatten():
        ax.tick_params(labelbottom=True)
    plt.show()
    st.write("**Ethos x Pathos**")
    st.pyplot(g)
    add_spacelines(2)



def FreqTables(df_list, rhetoric_dims = ['ethos', 'pathos']):
    st.write("### Word Frequency Tables")
    add_spacelines(2)

    selected_rhet_dim = st.selectbox("Choose a rhetoric strategy for analysis", rhetoric_dims, index=0)
    selected_rhet_dim = selected_rhet_dim+"_label"
    add_spacelines(1)
    df = df_list[0]
    df = clean_text(df, 'sentence_lemmatized', text_column_name = "sentence_lemmatized")
    #df = lemmatization(df, 'content')
    if not 'neutral' in df['ethos_label'].unique():
        df['ethos_label'] = df['ethos_label'].map(ethos_mapping)
    if not 'negative' in df['pathos_label'].unique():
        df['pathos_label'] = df['pathos_label'].map(valence_mapping)

    ddmsc = ['support', 'attack']
    if selected_rhet_dim == 'pathos_label':
        ddmsc = ['positive', 'negative']

    odds_list_of_dicts = []

    # 1 vs rest
    #num = np.floor( len(df) / 10 )
    for ddmsc1 in ddmsc:
        dict_1vsall_percent = {}
        dict_1vsall_effect_size = {}
        ddmsc2w = " ".join( df[df[selected_rhet_dim] == ddmsc1].sentence_lemmatized.fillna('').astype('str').values ).split()

        ddmsc2w = Counter(ddmsc2w).most_common()
        ddmsc2w_word = dict(ddmsc2w)
        odds_list_of_dicts.append(ddmsc2w_word)


    df_odds_pos = pd.DataFrame({
                'word':odds_list_of_dicts[0].keys(),
                'frequency':odds_list_of_dicts[0].values(),
    })
    df_odds_pos['category'] = ddmsc[0]
    df_odds_neg = pd.DataFrame({
                'word':odds_list_of_dicts[1].keys(),
                'frequency':odds_list_of_dicts[1].values(),
    })
    df_odds_neg['category'] = ddmsc[1]
    df_odds_neg = df_odds_neg.sort_values(by = ['frequency'], ascending = False)
    #df_odds_neg = df_odds_neg[df_odds_neg.frequency > 2]
    df_odds_pos = df_odds_pos.sort_values(by = ['frequency'], ascending = False)
    #df_odds_pos = df_odds_pos[df_odds_pos.frequency > 2]

    df_odds_neg = transform_text(df_odds_neg, 'word')
    df_odds_pos = transform_text(df_odds_pos, 'word')
    pos_list = ['NOUN', 'VERB', 'NUM', 'PROPN', 'ADJ', 'ADV']
    df_odds_neg['POS_tags']  = np.where(df_odds_neg.word == 'url', 'NOUN', df_odds_neg['POS_tags'])
    df_odds_pos['POS_tags']  = np.where(df_odds_pos.word == 'url', 'NOUN', df_odds_pos['POS_tags'])
    df_odds_neg = df_odds_neg[df_odds_neg.POS_tags.isin(pos_list)]
    df_odds_pos = df_odds_pos[df_odds_pos.POS_tags.isin(pos_list)]
    df_odds_neg = df_odds_neg.reset_index(drop=True)
    df_odds_pos = df_odds_pos.reset_index(drop=True)

    df_odds_neg['abusive'] = df_odds_neg.word.apply(lambda x: " ".join( set(x.lower().split()).intersection(abus_words)  ))
    df_odds_neg['abusive'] = np.where( df_odds_neg['abusive'].fillna('').astype('str').map(len) > 1 , 'abusive', 'non-abusive' )
    df_odds_pos['abusive'] = df_odds_pos.word.apply(lambda x: " ".join( set(x.lower().split()).intersection(abus_words)  ))
    df_odds_pos['abusive'] = np.where( df_odds_pos['abusive'].fillna('').astype('str').map(len) > 1, 'abusive', 'non-abusive' )

    df_odds_pos.index += 1
    df_odds_neg.index += 1

    if "sentence_lemmatized" in df.columns:
        df.sentence_lemmatized = df.sentence_lemmatized.str.replace(" pyro sick ", " pyro2sick ")

    import nltk
    oddpos_c, oddneg_c = st.columns(2, gap = 'large')
    dimm = selected_rhet_dim.split("_")[0]
    with oddpos_c:
        st.write(f'Number of **{dimm} {df_odds_pos.category.iloc[0]}** words: {len(df_odds_pos)} ')
        st.dataframe(df_odds_pos)
        add_spacelines(1)

        pos_list_freq = df_odds_pos.word.tolist()
        freq_word_pos = st.multiselect('Choose a word you would like to see data cases for', pos_list_freq, pos_list_freq[:2])
        df_odds_pos_words = set(freq_word_pos)
        df['freq_words_'+df_odds_pos.category.iloc[0]] = df.sentence_lemmatized.apply(lambda x: " ".join( set(x.split()).intersection(df_odds_pos_words) ))

        add_spacelines(1)
        cols_odds1 = ['source', 'sentence', 'ethos_label', 'pathos_label', 'Target',
                         'freq_words_'+df_odds_pos.category.iloc[0]]
        df01 = df[ (df['freq_words_'+df_odds_pos.category.iloc[0]].str.split().map(len) >= 1) & (df[selected_rhet_dim] == df_odds_pos.category.iloc[0]) ]
        txt_df01 = " ".join(df01.sentence_lemmatized.values)
        df['mentions'] = df.sentence.apply(lambda x: " ".join( w for w in str(x).split() if "@" in w ))

        df_targets = " ".join( df.mentions.dropna().str.replace("@", "").str.lower().unique() ).split()
        t = nltk.tokenize.WhitespaceTokenizer()
        #c = Text(t.tokenize(txt_df01))

        #st.write(freq_word_pos[0], txt_df01[:50])
        #st.write(c.concordance_list(freq_word_pos[0], width=51, lines=50))
        # Loading Libraries
        from nltk.collocations import TrigramCollocationFinder, BigramCollocationFinder
        from nltk.metrics import TrigramAssocMeasures, BigramAssocMeasures
        from nltk.corpus import stopwords
        stopset = set(stopwords.words('english'))
        filter_stops = lambda w: len(w) < 3 or w in stopset

        def get_keyword_collocations(corpus, keyword, windowsize=10, numresults=10):
            import string
            from nltk.tokenize import word_tokenize
            from nltk.collocations import BigramCollocationFinder
            from nltk.collocations import BigramAssocMeasures
            from nltk.corpus import stopwords
            nltk.download('punkt')
            #'''This function uses the Natural Language Toolkit to find collocations
            #for a specific keyword in a corpus. It takes as an argument a string that
            #contains the corpus you want to find collocations from. It prints the top
            #collocations it finds for each keyword.
            #https://github.com/ahegel/collocations/blob/master/get_collocations3.py
            #'''
            # convert the corpus (a string) into  a list of words
            tokens = word_tokenize(corpus)
            # initialize the bigram association measures object to score each collocation
            bigram_measures = BigramAssocMeasures()
            # initialize the bigram collocation finder object to find and rank collocations
            finder = BigramCollocationFinder.from_words(tokens, window_size=windowsize)
            # initialize a function that will narrow down collocates that don't contain the keyword
            keyword_filter = lambda *w: keyword not in w
            # apply a series of filters to narrow down the collocation results
            ignored_words = stopwords.words('english')
            finder.apply_word_filter(lambda w: len(w) < 2 or w.lower() in ignored_words)
            finder.apply_freq_filter(2)
            finder.apply_ngram_filter(keyword_filter)
            # calculate the top results by T-score
            # list of all possible measures: .raw_freq, .pmi, .likelihood_ratio, .chi_sq, .phi_sq, .fisher, .student_t, .mi_like, .poisson_stirling, .jaccard, .dice
            results = finder.nbest(bigram_measures.student_t, numresults)
            # print the results
            print("Top collocations for ", str(keyword), ":")
            collocations = ''
            for k, v in results:
                if k != keyword:
                    collocations += k + ' '
                else:
                    collocations += v + ' '
            #print(collocations, '\n')
            st.write(collocations, '\n')

        words_of_interest = list(freq_word_pos) # ["love", "die"]
        import nltk
        #from nltk.collocations
        bigram_measures = nltk.collocations.BigramAssocMeasures()
        finder = nltk.collocations.BigramCollocationFinder.from_words(txt_df01.split(), window_size=5)
        finder.nbest(bigram_measures.pmi, 10)
        finder.apply_freq_filter(2)
        results=finder.nbest(bigram_measures.pmi, 10)
        scores = finder.score_ngrams(bigram_measures.pmi)
        seed = "word"
        result_term = []
        result_pmi = []
        for terms, score in scores:
            if terms [0] in freq_word_pos or terms [1] in freq_word_pos:
                if not " ".join(terms) in result_term and not str(terms [1]) + " " + str(terms [0]) in result_term:
                    result_term.append(" ".join(terms))
                    result_pmi.append(score)

        df_bigrams0 = pd.DataFrame({'bi-grams':result_term, 'PMI':result_pmi})
        df_bigrams0 = df_bigrams0[df_bigrams0.PMI > 0].reset_index(drop=True)
        df_bigrams0.PMI = df_bigrams0.PMI.round(3)
        df_bigrams0.index += 1
        #st.write(df_bigrams0)


        #for word in words_of_interest:
            #get_keyword_collocations(txt_df01, word)

        st.write(f'Bi-gram collocations with **{freq_word_pos}**')
        colBigrams = list(nltk.ngrams(t.tokenize(txt_df01), 2))
        colBigrams2 = []
        for p in colBigrams:
            for w in freq_word_pos:
                if (w == p[0] or w == p[1]) and not (p[0] in df_targets or p[1] in df_targets):
                    colBigrams2.append(" ".join(p))
        df_bigrams = pd.DataFrame({'bi-grams':colBigrams2})
        df_bigrams = df_bigrams.drop_duplicates()
        df_bigrams = df_bigrams.groupby('bi-grams', as_index=False).size()
        df_bigrams.columns = ['bi-grams', 'frequency']
        df_bigrams = df_bigrams.sort_values(by = 'frequency', ascending = False).reset_index(drop=True)
        #df_bigrams = df_bigrams[df_bigrams.duplicated()].reset_index(drop=True)
        df_bigrams.index += 1
        st.write(df_bigrams0)# df_bigrams0 df_bigrams

        add_spacelines(1)
        st.write(f'Cases with **{freq_word_pos}** words:')
        st.dataframe(df01[cols_odds1].drop_duplicates().set_index('source'))


    with oddneg_c:
        st.write(f'Number of **{dimm} {df_odds_neg.category.iloc[0]}** words: {len(df_odds_neg)} ')
        st.dataframe(df_odds_neg)
        add_spacelines(1)

        neg_list_freq = df_odds_neg.word.tolist()
        freq_word_neg = st.multiselect('Choose a word you would like to see data cases for', neg_list_freq, neg_list_freq[:2])
        df_odds_neg_words = set(freq_word_neg)
        df['freq_words_'+df_odds_neg.category.iloc[0]] = df.sentence_lemmatized.apply(lambda x: " ".join( set(x.split()).intersection(df_odds_neg_words) ))
        df02 = df[ (df['freq_words_'+df_odds_neg.category.iloc[0]].str.split().map(len) >= 1) & (df[selected_rhet_dim] == df_odds_neg.category.iloc[0]) ]
        txt_df02 = " ".join(df02.sentence_lemmatized.values)

        finder = nltk.collocations.BigramCollocationFinder.from_words(txt_df02.split(), window_size=5)
        finder.nbest(bigram_measures.pmi, 10)
        finder.apply_freq_filter(2)
        results=finder.nbest(bigram_measures.pmi, 10)
        scores = finder.score_ngrams(bigram_measures.pmi)
        seed = "word"
        result_term = []
        result_pmi = []
        for terms, score in scores:
            if terms [0] in freq_word_neg or terms [1] in freq_word_neg:
                if not " ".join(terms) in result_term and not str(terms [1]) + " " + str(terms [0]) in result_term:
                    result_term.append(" ".join(terms))
                    result_pmi.append(score)

        df_bigrams0 = pd.DataFrame({'bi-grams':result_term, 'PMI':result_pmi})
        df_bigrams0 = df_bigrams0[df_bigrams0.PMI > 0].reset_index(drop=True)
        df_bigrams0.PMI = df_bigrams0.PMI.round(3)
        df_bigrams0.index += 1

        cols_odds2 = ['source', 'sentence', 'ethos_label', 'pathos_label', 'Target',
                         'freq_words_'+df_odds_neg.category.iloc[0]]
        add_spacelines(1)
        st.write(f'Bi-gram collocations with **{freq_word_neg}**')
        colBigrams = list(nltk.ngrams(t.tokenize(txt_df02), 2))
        colBigrams2 = []
        for p in colBigrams:
            for w in freq_word_neg:
                if (w == p[0] or w == p[1]) and not (p[0] in df_targets or p[1] in df_targets):
                    colBigrams2.append(" ".join(p))
        df_bigrams = pd.DataFrame({'bi-grams':colBigrams2})
        df_bigrams = df_bigrams.drop_duplicates()#.reset_index(drop=True)
        df_bigrams = df_bigrams.groupby('bi-grams', as_index=False).size()
        df_bigrams.columns = ['bi-grams', 'frequency']
        df_bigrams = df_bigrams.sort_values(by = 'frequency', ascending = False).reset_index(drop=True)
        #df_bigrams = df_bigrams[df_bigrams.duplicated()].reset_index(drop=True)
        df_bigrams.index += 1
        st.write(df_bigrams0)

        add_spacelines(1)
        st.write(f'Cases with **{freq_word_neg}** words:')
        st.dataframe(df02[cols_odds2].drop_duplicates().set_index('source'))




def FreqTablesLog(df_list, rhetoric_dims = ['ethos', 'logos']):
    st.write("### Word Frequency Tables")
    add_spacelines(2)

    selected_rhet_dim = st.selectbox("Choose a rhetoric strategy for analysis", rhetoric_dims, index=0)
    if selected_rhet_dim == 'logos':
        df = df_list[1]
        df['locution_conclusion'] = df.locution_conclusion.apply(lambda x: " ".join( str(x).split(':')[1:]) )
        df['locution_premise'] = df.locution_premise.apply(lambda x: " ".join( str(x).split(':')[1:]) )
        df['sentence'] = df['locution_premise'].astype('str') + " " + df['locution_conclusion'].astype('str')
        map_naming = {
                'Default Conflict': 'attack',
                'Default Rephrase' : 'neutral',
                'Default Inference' : 'support'}
        df.connection = df.connection.map(map_naming)

    else:
        df = df_list[0]
        if not 'neutral' in df['ethos_label'].unique():
            df['ethos_label'] = df['ethos_label'].map(ethos_mapping)

    selected_rhet_dim = selected_rhet_dim.replace('ethos', 'ethos_label').replace('logos', 'connection')
    add_spacelines(1)

    if 'sentence_lemmatized' in df.columns:
        df = clean_text(df, 'sentence_lemmatized', text_column_name = "sentence_lemmatized")
    else:
        df = lemmatization(df, 'sentence')
        df = clean_text(df, 'sentence_lemmatized', text_column_name = "sentence_lemmatized")


    ddmsc = ['support', 'attack']
    if selected_rhet_dim == 'pathos_label':
        ddmsc = ['positive', 'negative']

    odds_list_of_dicts = []

    # 1 vs rest
    #num = np.floor( len(df) / 10 )
    for ddmsc1 in ddmsc:
        dict_1vsall_percent = {}
        dict_1vsall_effect_size = {}
        ddmsc2w = " ".join( df[df[selected_rhet_dim] == ddmsc1].sentence_lemmatized.fillna('').astype('str').values ).split()

        ddmsc2w = Counter(ddmsc2w).most_common()
        ddmsc2w_word = dict(ddmsc2w)
        odds_list_of_dicts.append(ddmsc2w_word)


    df_odds_pos = pd.DataFrame({
                'word':odds_list_of_dicts[0].keys(),
                'frequency':odds_list_of_dicts[0].values(),
    })
    df_odds_pos['category'] = ddmsc[0]
    df_odds_neg = pd.DataFrame({
                'word':odds_list_of_dicts[1].keys(),
                'frequency':odds_list_of_dicts[1].values(),
    })
    df_odds_neg['category'] = ddmsc[1]
    df_odds_neg = df_odds_neg.sort_values(by = ['frequency'], ascending = False)
    #df_odds_neg = df_odds_neg[df_odds_neg.frequency > 2]
    df_odds_pos = df_odds_pos.sort_values(by = ['frequency'], ascending = False)
    #df_odds_pos = df_odds_pos[df_odds_pos.frequency > 2]

    df_odds_neg = transform_text(df_odds_neg, 'word')
    df_odds_pos = transform_text(df_odds_pos, 'word')
    pos_list = ['NOUN', 'VERB', 'NUM', 'PROPN', 'ADJ', 'ADV']
    df_odds_neg = df_odds_neg[df_odds_neg.POS_tags.isin(pos_list)]
    df_odds_pos = df_odds_pos[df_odds_pos.POS_tags.isin(pos_list)]

    df_odds_neg = df_odds_neg.reset_index(drop=True)
    df_odds_pos = df_odds_pos.reset_index(drop=True)
    df_odds_pos.index += 1
    df_odds_neg.index += 1

    df_odds_pos_10n = np.ceil(df_odds_pos.shape[0] * 0.1)
    df_odds_neg_10n = np.ceil(df_odds_neg.shape[0] * 0.1)

    df_odds_pos_tags_summ = df_odds_pos.iloc[:int(df_odds_pos_10n)].POS_tags.value_counts(normalize = True).round(2)*100
    df_odds_neg_tags_summ = df_odds_neg.iloc[:int(df_odds_neg_10n)].POS_tags.value_counts(normalize = True).round(2)*100
    df_odds_pos_tags_summ = df_odds_pos_tags_summ.reset_index()
    df_odds_pos_tags_summ.columns = ['POS_tags', 'percentage']

    df_odds_neg_tags_summ = df_odds_neg_tags_summ.reset_index()
    df_odds_neg_tags_summ.columns = ['POS_tags', 'percentage']

    df_odds_pos_tags_summ = df_odds_pos_tags_summ[df_odds_pos_tags_summ.percentage > 1]
    df_odds_neg_tags_summ = df_odds_neg_tags_summ[df_odds_neg_tags_summ.percentage > 1]

    if "sentence_lemmatized" in df.columns:
        df.sentence_lemmatized = df.sentence_lemmatized.str.replace(" pyro sick ", " pyro2sick ")

    import nltk
    oddpos_c, oddneg_c = st.columns(2, gap = 'large')
    dimm = selected_rhet_dim.split("_")[0]
    dimm = str(dimm).replace('connection', 'logos')
    with oddpos_c:
        st.write(f'Number of **{dimm} {df_odds_pos.category.iloc[0]}** words: {len(df_odds_pos)} ')
        st.dataframe(df_odds_pos)
        add_spacelines(1)
        st.write("Part-of-Speech analysis for the top 10% of words in the table")
        st.write(df_odds_pos_tags_summ)
        add_spacelines(1)

        pos_list_freq = df_odds_pos.word.tolist()
        freq_word_pos = st.multiselect('Choose a word you would like to see data cases for', pos_list_freq, pos_list_freq[:2])
        df_odds_pos_words = set(freq_word_pos)
        df['freq_words_'+df_odds_pos.category.iloc[0]] = df.sentence_lemmatized.apply(lambda x: " ".join( set(x.split()).intersection(df_odds_pos_words) ))

        add_spacelines(1)

        df01 = df[ (df['freq_words_'+df_odds_pos.category.iloc[0]].str.split().map(len) >= 1) & (df[selected_rhet_dim] == df_odds_pos.category.iloc[0]) ]
        txt_df01 = " ".join(df01.sentence_lemmatized.values)
        df['mentions'] = df.sentence.apply(lambda x: " ".join( w for w in str(x).split() if "@" in w ))

        df_targets = " ".join( df.mentions.dropna().str.replace("@", "").str.lower().unique() ).split()

        words_of_interest = list(freq_word_pos) # ["love", "die"]
        import nltk
        t = nltk.tokenize.WhitespaceTokenizer()
        #from nltk.collocations
        bigram_measures = nltk.collocations.BigramAssocMeasures()
        finder = nltk.collocations.BigramCollocationFinder.from_words(txt_df01.split(), window_size=5)
        finder.nbest(bigram_measures.pmi, 10)
        finder.apply_freq_filter(2)
        results=finder.nbest(bigram_measures.pmi, 10)
        scores = finder.score_ngrams(bigram_measures.pmi)
        seed = "word"
        result_term = []
        result_pmi = []
        for terms, score in scores:
            if terms [0] in freq_word_pos or terms [1] in freq_word_pos:
                if not " ".join(terms) in result_term and not str(terms [1]) + " " + str(terms [0]) in result_term:
                    result_term.append(" ".join(terms))
                    result_pmi.append(score)

        df_bigrams0 = pd.DataFrame({'bi-grams':result_term, 'PMI':result_pmi})
        df_bigrams0 = df_bigrams0[df_bigrams0.PMI > 0].reset_index(drop=True)
        df_bigrams0.PMI = df_bigrams0.PMI.round(3)
        df_bigrams0.index += 1

        st.write(f'Bi-gram collocations with **{freq_word_pos}**')
        colBigrams = list(nltk.ngrams(t.tokenize(txt_df01), 2))
        colBigrams2 = []
        for p in colBigrams:
            for w in freq_word_pos:
                if (w == p[0] or w == p[1]) and not (p[0] in df_targets or p[1] in df_targets):
                    colBigrams2.append(" ".join(p))
        df_bigrams = pd.DataFrame({'bi-grams':colBigrams2})
        df_bigrams = df_bigrams.drop_duplicates()
        df_bigrams = df_bigrams.groupby('bi-grams', as_index=False).size()
        df_bigrams.columns = ['bi-grams', 'frequency']
        df_bigrams = df_bigrams.sort_values(by = 'frequency', ascending = False).reset_index(drop=True)
        #df_bigrams = df_bigrams[df_bigrams.duplicated()].reset_index(drop=True)
        df_bigrams.index += 1
        st.write(df_bigrams0)# df_bigrams0 df_bigrams

        add_spacelines(1)
        st.write(f'Cases with **{freq_word_pos}** words:')
        st.dataframe(df01)


    with oddneg_c:
        st.write(f'Number of **{dimm} {df_odds_neg.category.iloc[0]}** words: {len(df_odds_neg)} ')
        st.dataframe(df_odds_neg)
        add_spacelines(1)
        st.write("Part-of-Speech analysis for the top 10% of words in the table")
        st.write(df_odds_neg_tags_summ)
        add_spacelines(1)

        neg_list_freq = df_odds_neg.word.tolist()
        freq_word_neg = st.multiselect('Choose a word you would like to see data cases for', neg_list_freq, neg_list_freq[:2])
        df_odds_neg_words = set(freq_word_neg)
        df['freq_words_'+df_odds_neg.category.iloc[0]] = df.sentence_lemmatized.apply(lambda x: " ".join( set(x.split()).intersection(df_odds_neg_words) ))
        df02 = df[ (df['freq_words_'+df_odds_neg.category.iloc[0]].str.split().map(len) >= 1) & (df[selected_rhet_dim] == df_odds_neg.category.iloc[0]) ]
        txt_df02 = " ".join(df02.sentence_lemmatized.values)

        finder = nltk.collocations.BigramCollocationFinder.from_words(txt_df02.split(), window_size=5)
        finder.nbest(bigram_measures.pmi, 10)
        finder.apply_freq_filter(2)
        results=finder.nbest(bigram_measures.pmi, 10)
        scores = finder.score_ngrams(bigram_measures.pmi)
        seed = "word"
        result_term = []
        result_pmi = []
        for terms, score in scores:
            if terms [0] in freq_word_neg or terms [1] in freq_word_neg:
                if not " ".join(terms) in result_term and not str(terms [1]) + " " + str(terms [0]) in result_term:
                    result_term.append(" ".join(terms))
                    result_pmi.append(score)

        df_bigrams0 = pd.DataFrame({'bi-grams':result_term, 'PMI':result_pmi})
        df_bigrams0 = df_bigrams0[df_bigrams0.PMI > 0].reset_index(drop=True)
        df_bigrams0.PMI = df_bigrams0.PMI.round(3)
        df_bigrams0.index += 1

        add_spacelines(1)
        st.write(f'Bi-gram collocations with **{freq_word_neg}**')
        colBigrams = list(nltk.ngrams(t.tokenize(txt_df02), 2))
        colBigrams2 = []
        for p in colBigrams:
            for w in freq_word_neg:
                if (w == p[0] or w == p[1]) and not (p[0] in df_targets or p[1] in df_targets):
                    colBigrams2.append(" ".join(p))
        df_bigrams = pd.DataFrame({'bi-grams':colBigrams2})
        df_bigrams = df_bigrams.drop_duplicates()#.reset_index(drop=True)
        df_bigrams = df_bigrams.groupby('bi-grams', as_index=False).size()
        df_bigrams.columns = ['bi-grams', 'frequency']
        df_bigrams = df_bigrams.sort_values(by = 'frequency', ascending = False).reset_index(drop=True)
        #df_bigrams = df_bigrams[df_bigrams.duplicated()].reset_index(drop=True)
        df_bigrams.index += 1
        st.write(df_bigrams0)

        add_spacelines(1)
        st.write(f'Cases with **{freq_word_neg}** words:')
        st.dataframe(df02)





def OddsRatio(df_list):
    st.write("### Lexical Analysis - Odds Ratio")
    add_spacelines(2)
    rhetoric_dims = ['ethos', 'pathos']
    selected_rhet_dim = st.selectbox("Choose a rhetoric strategy for analysis", rhetoric_dims, index=0)
    selected_rhet_dim = selected_rhet_dim+"_label"
    add_spacelines(1)
    df = df_list[0]
    df = clean_text(df, 'sentence_lemmatized', text_column_name = "sentence_lemmatized")
    #df = lemmatization(df, 'content')
    if not 'neutral' in df['ethos_label'].unique():
        df['ethos_label'] = df['ethos_label'].map(ethos_mapping)
    if not 'negative' in df['pathos_label'].unique():
        df['pathos_label'] = df['pathos_label'].map(valence_mapping)

    ddmsc = ['support', 'attack']
    if selected_rhet_dim == 'pathos_label':
        ddmsc = ['positive', 'negative']

    odds_list_of_dicts = []
    effect_list_of_dicts = []
    freq_list_of_dicts = []
    # 1 vs rest
    #num = np.floor( len(df) / 10 )
    for ddmsc1 in ddmsc:
        dict_1vsall_percent = {}
        dict_1vsall_effect_size = {}
        dict_1vsall_freq = {}
        #all100popular = Counter(" ".join( df.lemmatized.values ).split()).most_common(100)
        #all100popular = list(w[0] for w in all100popular)

        ddmsc1w = " ".join( df[df[selected_rhet_dim] == ddmsc1].sentence_lemmatized.fillna('').astype('str').values ).split()
        c = len(ddmsc1w)
        #ddmsc1w = list(w for w in ddmsc1w if not w in all100popular)
        ddmsc1w = Counter(ddmsc1w).most_common() # num
        if ddmsc1 in ['positive', 'support']:
            ddmsc1w = [w for w in ddmsc1w if w[1] >= 3 ]
        else:
            ddmsc1w = [w for w in ddmsc1w if w[1] > 3 ]
        #print('**********')
        #print(len(ddmsc1w), ddmsc1w)
        #print([w for w in ddmsc1w if w[1] > 2 ])
        #print(len([w for w in ddmsc1w if w[1] > 2 ]))
        ddmsc1w_word = dict(ddmsc1w)

        ddmsc2w = " ".join( df[df[selected_rhet_dim] != ddmsc1].sentence_lemmatized.fillna('').astype('str').values ).split()
        d = len(ddmsc2w)
        #ddmsc2w = list(w for w in ddmsc2w if not w in all100popular)
        ddmsc2w = Counter(ddmsc2w).most_common()
        ddmsc2w_word = dict(ddmsc2w)


        ddmsc1w_words = list( ddmsc1w_word.keys() )
        for n, dim in enumerate( ddmsc1w_words ):

            a = ddmsc1w_word[dim]
            try:
                b = ddmsc2w_word[dim]
            except:
                b = 0.5

            ca = c-a
            bd = d-b

            E1 = c*(a+b) / (c+d)
            E2 = d*(a+b) / (c+d)

            g2 = 2*((a*np.log(a/E1)) + (b* np.log(b/E2)))
            g2 = round(g2, 2)

            odds = round( (a*(d-b)) / (b*(c-a)), 2)

            if odds > 1:

                if g2 > 10.83:
                    #print(f"{dim, g2, odds} ***p < 0.001 ")
                    dict_1vsall_percent[dim] = odds
                    dict_1vsall_effect_size[dim] = 0.001
                    dict_1vsall_freq[dim] = a
                elif g2 > 6.63:
                    #print(f"{dim, g2, odds} **p < 0.01 ")
                    dict_1vsall_percent[dim] = odds
                    dict_1vsall_effect_size[dim] = 0.01
                    dict_1vsall_freq[dim] = a
                elif g2 > 3.84:
                    #print(f"{dim, g2, odds} *p < 0.05 ")
                    dict_1vsall_percent[dim] = odds
                    dict_1vsall_effect_size[dim] = 0.05
                    dict_1vsall_freq[dim] = a
        #print(dict(sorted(dict_1vsall_percent.items(), key=lambda item: item[1])))
        odds_list_of_dicts.append(dict_1vsall_percent)
        effect_list_of_dicts.append(dict_1vsall_effect_size)
        freq_list_of_dicts.append(dict_1vsall_freq)

    df_odds_pos = pd.DataFrame({
                'word':odds_list_of_dicts[0].keys(),
                'odds':odds_list_of_dicts[0].values(),
                'frquency':freq_list_of_dicts[0].values(),
                'effect_size_p':effect_list_of_dicts[0].values(),
    })
    df_odds_pos['category'] = ddmsc[0]
    df_odds_neg = pd.DataFrame({
                'word':odds_list_of_dicts[1].keys(),
                'odds':odds_list_of_dicts[1].values(),
                'frquency':freq_list_of_dicts[1].values(),
                'effect_size_p':effect_list_of_dicts[1].values(),
    })
    df_odds_neg['category'] = ddmsc[1]
    df_odds_neg = df_odds_neg.sort_values(by = ['odds'], ascending = False)
    df_odds_pos = df_odds_pos.sort_values(by = ['odds'], ascending = False)


    df_odds_neg = transform_text(df_odds_neg, 'word')
    df_odds_pos = transform_text(df_odds_pos, 'word')
    pos_list = ['NOUN', 'VERB', 'NUM', 'PROPN', 'ADJ', 'ADV']
    df_odds_neg = df_odds_neg[df_odds_neg.POS_tags.isin(pos_list)]
    df_odds_pos = df_odds_pos[df_odds_pos.POS_tags.isin(pos_list)]
    df_odds_neg = df_odds_neg.reset_index(drop=True)
    df_odds_pos = df_odds_pos.reset_index(drop=True)
    df_odds_pos.index += 1
    df_odds_neg.index += 1

    df_odds_neg['abusive'] = df_odds_neg.word.apply(lambda x: " ".join( set(x.lower().split()).intersection(abus_words)  ))
    df_odds_neg['abusive'] = np.where( df_odds_neg['abusive'].fillna('').astype('str').map(len) > 1 , 'abusive', 'non-abusive' )
    df_odds_pos['abusive'] = df_odds_pos.word.apply(lambda x: " ".join( set(x.lower().split()).intersection(abus_words)  ))
    df_odds_pos['abusive'] = np.where( df_odds_pos['abusive'].fillna('').astype('str').map(len) > 1, 'abusive', 'non-abusive' )

    df_odds_pos_tags_summ = df_odds_pos.POS_tags.value_counts(normalize = True).round(2)*100
    df_odds_neg_tags_summ = df_odds_neg.POS_tags.value_counts(normalize = True).round(2)*100
    df_odds_pos_tags_summ = df_odds_pos_tags_summ.reset_index()
    df_odds_pos_tags_summ.columns = ['POS_tags', 'percentage']

    df_odds_neg_tags_summ = df_odds_neg_tags_summ.reset_index()
    df_odds_neg_tags_summ.columns = ['POS_tags', 'percentage']

    df_odds_pos_tags_summ = df_odds_pos_tags_summ[df_odds_pos_tags_summ.percentage > 1]
    df_odds_neg_tags_summ = df_odds_neg_tags_summ[df_odds_neg_tags_summ.percentage > 1]

    df_odds_pos_words = set(df_odds_pos.word.values)
    df_odds_neg_words = set(df_odds_neg.word.values)

    df_odds = pd.concat( [df_odds_pos, df_odds_neg], axis = 0, ignore_index = True )
    df_odds = df_odds.sort_values(by = ['category', 'odds'], ascending = False)
    df['odds_words_'+df_odds_pos.category.iloc[0]] = df.sentence_lemmatized.apply(lambda x: " ".join( set(x.split()).intersection(df_odds_pos_words) ))
    df['odds_words_'+df_odds_neg.category.iloc[0]] = df.sentence_lemmatized.apply(lambda x: " ".join( set(x.split()).intersection(df_odds_neg_words) ))

    df_odds_pos_abs= df_odds_pos.abusive.value_counts(normalize = True).round(3)*100
    df_odds_neg_abs = df_odds_neg.abusive.value_counts(normalize = True).round(3)*100
    df_odds_pos_abs = df_odds_pos_abs.reset_index()
    df_odds_pos_abs.columns = ['abusive', 'percentage']
    df_odds_neg_abs = df_odds_neg_abs.reset_index()
    df_odds_neg_abs.columns = ['abusive', 'percentage']

    tab_odd, tab_pos, tab_abuse = st.tabs(['Odds', 'POS', 'Abusiveness'])
    with tab_odd:
        oddpos_c, oddneg_c = st.columns(2, gap = 'large')
        cols_odds = ['source', 'sentence', 'ethos_label', 'pathos_label', 'Target',
                     'odds_words_'+df_odds_pos.category.iloc[0], 'odds_words_'+df_odds_neg.category.iloc[0]]
        dimm = selected_rhet_dim.split("_")[0]
        with oddpos_c:
            st.write(f'Number of **{dimm} {df_odds_pos.category.iloc[0]}** words: {len(df_odds_pos)} ')
            st.dataframe(df_odds_pos)
            #add_spacelines(1)
            #st.dataframe(df_odds_pos_tags_summ)
            add_spacelines(1)
            st.write(f'Cases with **{df_odds_pos.category.iloc[0]}** words:')
            st.dataframe(df[ (df['odds_words_'+df_odds_pos.category.iloc[0]].str.split().map(len) >= 1) &\
                                (df[selected_rhet_dim] == df_odds_pos.category.iloc[0])  ][cols_odds].set_index('source').drop_duplicates('sentence'))

        with oddneg_c:
            st.write(f'Number of **{dimm} {df_odds_neg.category.iloc[0]}** words: {len(df_odds_neg)} ')
            st.dataframe(df_odds_neg)
            #add_spacelines(1)
            #st.dataframe(df_odds_neg_tags_summ)
            add_spacelines(1)
            st.write(f'Cases with **{df_odds_neg.category.iloc[0]}** words:')
            st.dataframe(df[ (df['odds_words_'+df_odds_neg.category.iloc[0]].str.split().map(len) >= 1) &\
                                (df[selected_rhet_dim] == df_odds_neg.category.iloc[0]) ][cols_odds].set_index('source').drop_duplicates('sentence'))

    with tab_pos:
        sns.set(font_scale = 1.25, style = 'whitegrid')
        df_odds_pos_tags_summ['category'] = df_odds_pos.category.iloc[0]
        df_odds_neg_tags_summ['category'] = df_odds_neg.category.iloc[0]
        df_odds_pos = pd.concat([df_odds_pos_tags_summ, df_odds_neg_tags_summ], axis = 0, ignore_index=True)
        ffp = sns.catplot(kind='bar', data = df_odds_pos,
        y = 'POS_tags', x = 'percentage', hue = 'POS_tags', aspect = 1.3, height = 5, dodge=False,
        legend = False, col = 'category')
        ffp.set(ylabel = '')
        plt.tight_layout(w_pad=3)
        st.pyplot(ffp)
        add_spacelines(1)

        oddpos_cpos, oddneg_cpos = st.columns(2, gap = 'large')
        with oddpos_cpos:
            st.write(f'POS analysis of **{dimm} {df_odds_pos.category.iloc[0]}** words')
            add_spacelines(1)
            st.dataframe(df_odds_pos_tags_summ)
            add_spacelines(1)

        with oddneg_cpos:
            st.write(f'POS analysis of **{dimm} {df_odds_neg.category.iloc[0]}** words')
            add_spacelines(1)
            st.dataframe(df_odds_neg_tags_summ)
            add_spacelines(1)


    with tab_abuse:
        sns.set(font_scale = 1, style = 'whitegrid')
        df_odds_pos_abs['category'] = df_odds_pos.category.iloc[0]
        df_odds_neg_abs['category'] = df_odds_neg.category.iloc[0]
        df_odds_abs = pd.concat([df_odds_pos_abs, df_odds_neg_abs], axis = 0, ignore_index=True)
        ffp = sns.catplot(kind='bar', data = df_odds_abs,
        y = 'abusive', x = 'percentage', hue = 'abusive', aspect = 1.3, height = 3,dodge=False,
        palette = {'abusive':'darkred', 'non-abusive':'grey'}, legend = False, col = 'category')
        ffp.set(ylabel = '')
        plt.tight_layout(w_pad=3)
        st.pyplot(ffp)

        oddpos_cab, oddneg_cab = st.columns(2, gap = 'large')
        with oddpos_cab:
            st.write(f'Abusiveness analysis of **{dimm} {df_odds_pos.category.iloc[0]}** words')
            add_spacelines(1)
            st.dataframe(df_odds_pos_abs)
            add_spacelines(1)
            #ffp = sns.catplot(kind='bar', data = df_odds_pos_abs,
            #y = 'abusive', x = 'percentage', hue = 'abusive', aspect = 1.3, height = 3,dodge=False,
            #palette = {'abusive':'darkred', 'non-abusive':'grey'}, legend = False)
            #ffp.set(ylabel = '', title = f'Abusiveness of {dimm} {df_odds_pos.category.iloc[0]} words')
            #st.pyplot(ffp)
            add_spacelines(1)
            if df_odds_pos_abs.shape[0] > 1:
                st.write(df_odds_pos[df_odds_pos['abusive'] == 'abusive'])

        with oddneg_cab:
            st.write(f'Abusiveness analysis of **{dimm} {df_odds_neg.category.iloc[0]}** words')
            add_spacelines(1)
            st.dataframe(df_odds_neg_abs)
            add_spacelines(1)
            #ffn = sns.catplot(kind='bar', data = df_odds_neg_abs,
            #y = 'abusive', x = 'percentage', hue = 'abusive', aspect = 1.3, height = 3, dodge=False,
            #palette = {'abusive':'darkred', 'non-abusive':'grey'}, legend = False)
            #ffn.set(ylabel = '', title = f'Abusiveness of {dimm} {df_odds_neg.category.iloc[0]} words')
            #st.pyplot(ffn)
            add_spacelines(1)
            if df_odds_neg_abs.shape[0] > 1:
                st.write(df_odds_neg[df_odds_neg['abusive'] == 'abusive'])


def generateWordCloud_log():
    selected_rhet_dim = st.selectbox("Choose a rhetoric category for a WordCloud", rhetoric_dims, index=0)
    add_spacelines(1)
    if selected_rhet_dim == 'pathos':
        label_cloud = st.radio("Choose a label for words in WordCloud", ('negative', 'positive'))
        selected_rhet_dim = selected_rhet_dim.replace("ethos", "ethos_label").replace("pathos", "pathos_label")
        label_cloud = label_cloud.replace("negative", "attack").replace("positive", "support")
    else:
        label_cloud = st.radio("Choose a label for words in WordCloud", ('attack', 'support'))
        selected_rhet_dim = selected_rhet_dim.replace("ethos", "ethos_label")
        label_cloud = label_cloud.replace("attack / negative", "attack").replace("support / positive", "support")

    add_spacelines(1)
    threshold_cloud = st.slider('Select a precision value (threshold) for words in WordCloud', 0, 100, 80)
    st.info(f'Selected precision: **{threshold_cloud}**')
    add_spacelines(1)
    st.write("**Processing the output ...**")


    generateWordCloudc1, generateWordCloudc2 = st.columns(2)
    with generateWordCloudc1:
        st.write(f"##### {corpora_list[0].corpus.iloc[0]}")
        add_spacelines(1)
        generateWordCloud_sub_log(corpora_list[:2], rhetoric_dims = ['ethos', 'logos'], an_type = contents_radio_an_cat,
            selected_rhet_dim = selected_rhet_dim, label_cloud=label_cloud, threshold_cloud=threshold_cloud)
    with generateWordCloudc2:
        st.write(f"##### {corpora_list[-1].corpus.iloc[0]}")
        add_spacelines(1)
        generateWordCloud_sub_log(corpora_list[2:], rhetoric_dims = ['ethos', 'logos'], an_type = contents_radio_an_cat,
            selected_rhet_dim = selected_rhet_dim, label_cloud=label_cloud, threshold_cloud=threshold_cloud)



def generateWordCloud_sub_log(df_list,
        selected_rhet_dim, label_cloud, threshold_cloud,
        rhetoric_dims = ['ethos', 'pathos'], an_type = 'ADU-based'):

    df = df_list[0]
    #st.write(df)
    add_spacelines(1)
    if selected_rhet_dim != 'logos':
        df = clean_text(df, 'sentence_lemmatized', text_column_name = "sentence_lemmatized")
        if not 'neutral' in df['ethos_label'].unique():
            df['ethos_label'] = df['ethos_label'].map(ethos_mapping)
        if not 'negative' in df['pathos_label'].unique():
            df['pathos_label'] = df['pathos_label'].map(valence_mapping)

    elif selected_rhet_dim == 'logos':
        df = df_list[-1] #pd.concat(df_list, axis=0, ignore_index=True)
        #st.write(df)
        df = df.dropna(subset = 'premise')
        df['sentence_lemmatized'] = df['premise'].astype('str') + " " + df['conclusion'].astype('str')

        if an_type != 'Text-based':
            df = lemmatization(df, 'sentence_lemmatized', name_column = True)
            df['sentence_lemmatized'] = df['sentence_lemmatized'].astype('str').str.lower().str.replace('ahould', 'should')

        elif an_type == 'Text-based':
            df['premise'] = df['premise'].astype('str')
            df['conclusion'] = df['conclusion'].astype('str')
            df = df.reset_index()

            dfp = df.groupby(['id_connection', 'connection'])['premise'].apply(lambda x: " ".join(x)).reset_index()
            #st.write(dfp)
            #dfp['sentence_lemmatized'] = dfp['sentence_lemmatized'].astype('str')
            dfc = df.groupby(['id_connection', 'connection'])['conclusion'].apply(lambda x: " ".join(x)).reset_index()
            #st.write(dfc)

            dfp = dfp.merge(dfc, on = ['id_connection', 'connection']) #pd.concat([dfp, dfc.iloc[:, -1:]], axis=1) #dfp.merge(dfc, on = ['id_connection', 'connection'])
            dfp = dfp.drop_duplicates()
            #st.write(dfp)
            #st.write(dfp[dfp.id_connection == 185352])
            #st.stop()
            dfp['sentence_lemmatized'] = dfp.premise.astype('str')+ " " + dfc['conclusion'].astype('str')
            #st.write(dfp)
            import re
            dfp['sentence_lemmatized'] = dfp['sentence_lemmatized'].apply(lambda x: re.sub(r"\W+", " ", str(x)))
            dfp = lemmatization(dfp, 'sentence_lemmatized', name_column = True)
            dfp['sentence_lemmatized'] = dfp['sentence_lemmatized'].astype('str').str.lower().str.replace('ahould', 'should')
            df = dfp.copy()
            #st.write(dfc.shape, dfp.shape, df.shape)

    st.write(df.corpus.iloc[0])

    if (selected_rhet_dim == 'ethos_label'):
         df_for_wordcloud = prepare_cloud_lexeme_data(df[df[str(selected_rhet_dim)] == 'neutral'],
         df[df[str(selected_rhet_dim)] == 'support'],
         df[df[str(selected_rhet_dim)] == 'attack'])

    elif selected_rhet_dim == 'logos':
         df_for_wordcloud = prepare_cloud_lexeme_data(df[ ~(df['connection'].isin(['Default Inference', 'Default Conflict'])) ],
         df[df['connection'] == 'Default Inference'],
         df[df['connection'] == 'Default Conflict'])
    else:
        df_for_wordcloud = prepare_cloud_lexeme_data(df[df[str(selected_rhet_dim)] == 'neutral'],
        df[df[str(selected_rhet_dim)] == 'positive'],
        df[df[str(selected_rhet_dim)] == 'negative'])

    fig_cloud1, df_cloud_words1, figure_cloud_words1 = wordcloud_lexeme(df_for_wordcloud, lexeme_threshold = threshold_cloud, analysis_for = str(label_cloud))

    #_, cw2, _ = st.columns([1, 6, 1])
    #with cw2:
    st.pyplot(fig_cloud1)

    add_spacelines(2)

    st.write(f'WordCloud frequency table: ')
    if selected_rhet_dim == 'pathos_label':

        label_cloud = label_cloud.replace('attack', 'negative').replace('support', 'positive')
        df_cloud_words1 = df_cloud_words1.rename(columns = {
        'precis':'precision',
        'attack #':'negative #',
        'general #':'overall #',
        'support #':'positive #',
        })
    else:
        df_cloud_words1 = df_cloud_words1.rename(columns = {'general #':'overall #', 'precis':'precision'})

    df_cloud_words1 = df_cloud_words1.sort_values(by = 'precision', ascending = False)
    df_cloud_words1 = df_cloud_words1.reset_index(drop = True)
    df_cloud_words1.index += 1
    st.write(df_cloud_words1)


    cols_odds1 = ['source', 'sentence', 'ethos_label', 'pathos_label', 'Target',
                         'freq_words_'+label_cloud]

    if selected_rhet_dim == 'logos':
        df = df.rename(columns = {'connection':'logos'})
        #cols_odds1 = ['locution_conclusion', 'locution_premise', 'logos', 'argument_linked', 'freq_words_'+label_cloud]
        cols_odds1 = ['premise', 'conclusion', 'sentence_lemmatized', 'logos', 'freq_words_'+label_cloud]
        df['sentence_lemmatized'] = df['sentence_lemmatized'].astype('str')
        df['logos'] = df['logos'].map({'Default Inference':'support', 'Default Conflict':'attack'})

    pos_list_freq = df_cloud_words1.word.tolist()
    freq_word_pos = st.multiselect('Choose word(s) you would like to see data cases for', pos_list_freq, pos_list_freq[:2])
    df_odds_pos_words = set(freq_word_pos)
    df['freq_words_'+label_cloud] = df.sentence_lemmatized.apply(lambda x: " ".join( set(x.split()).intersection(df_odds_pos_words) ))
    #st.write(df)
    add_spacelines(1)
    st.write(f'Cases with **{freq_word_pos}** words:')
    st.dataframe(df[ (df['freq_words_'+label_cloud].str.split().map(len) >= 1) & (df[selected_rhet_dim] == label_cloud) ][cols_odds1])# .set_index('source')



def AntiHeroWordCloud_compare(df_list, selected_rhet_dim, label_cloud, threshold_cloud, box_stopwords,
                                rhetoric_dims = ['ethos', 'pathos'], targeted = False, selected_target = 'government' ):
    add_spacelines(1)

    if selected_rhet_dim != 'logos':
        df = df_list[0]
        #st.write(df)
        corp = df.corpus.iloc[0]
        if not 'neutral' in df['ethos_label'].unique():
            df['ethos_label'] = df['ethos_label'].map(ethos_mapping)
        if not 'negative' in df['pathos_label'].unique():
            df['pathos_label'] = df['pathos_label'].map(valence_mapping)

        if targeted:
            df = df[ (df.Target == selected_target) ]

        df = clean_text(df, 'sentence_lemmatized', text_column_name = "sentence_lemmatized")


    heroes_tab1, heroes_tab2, heroes_tab_explore = st.tabs(['Plot', 'Tables', 'Cases'])


    if "&" in df.corpus.iloc[0]:
        ds = "Covid & ElectionsSM"
        df['corpus'] = ds


    with heroes_tab1:
        add_spacelines(1)
        st.write(corp)
        #df = df0[ df0['corpus'] == corp ]
        df = clean_text(df, 'sentence_lemmatized', text_column_name = "sentence_lemmatized")
        data_neutral = df[df.ethos_label != label_cloud]
        neu_text = " ".join(data_neutral['sentence_lemmatized'].apply(lambda x: " ".join(t for t in set(x.split()))).to_numpy())
        count_dict_df_neu_text = Counter(neu_text.split(" "))

        df_neu_text = pd.DataFrame( {"word": list(count_dict_df_neu_text.keys()),
                                    'other #': list(count_dict_df_neu_text.values())} )

        data_attack = df[df.ethos_label == label_cloud]

        att_text = " ".join(data_attack['sentence_lemmatized'].apply(lambda x: " ".join(t for t in set(x.split()))).to_numpy())
        count_dict_df_att_text = Counter(att_text.split(" "))
        df_att_text = pd.DataFrame( {"word": list(count_dict_df_att_text.keys()),
                                    label_cloud+' #': list(count_dict_df_att_text.values())} )


        df_for_wordcloud = pd.merge(df_att_text, df_neu_text, on = 'word', how = 'outer')
        df_for_wordcloud.fillna(0, inplace=True)
        #st.write(df_for_wordcloud)

        df_for_wordcloud['general #'] = df_for_wordcloud['other #'].astype('int') + df_for_wordcloud[label_cloud+' #'].astype('int')
        df_for_wordcloud['word'] = df_for_wordcloud['word'].str.replace("'", "_").replace("”", "_").replace("’", "_")
        df_for_wordcloud.sort_values(by = label_cloud + ' #', inplace=True, ascending=False)
        df_for_wordcloud.reset_index(inplace=True, drop=True)
        #st.write(df_for_wordcloud)

        analysis_for = label_cloud
        df_for_wordcloud['precis'] = (round(df_for_wordcloud[label_cloud+' #'] / df_for_wordcloud['general #'], 3) * 100).apply(float) # att

        #fig_cloud1, df_cloud_words1, figure_cloud_words1 = wordcloud_lexeme(df_for_wordcloud, lexeme_threshold = threshold_cloud, analysis_for = str(label_cloud))
        if label_cloud == 'attack':
          #print(f'Analysis for: {analysis_for} ')
          cmap_wordcloud = 'Reds' #gist_heat
        elif label_cloud == 'both':
          #print(f'Analysis for: {analysis_for} ')
          cmap_wordcloud = 'autumn' #viridis
        else:
          #print(f'Analysis for: {analysis_for} ')
          cmap_wordcloud = 'Greens'


        dfcloud = df_for_wordcloud[(df_for_wordcloud['precis'] >= int(threshold_cloud)) & (df_for_wordcloud['general #'] > 1) & (df_for_wordcloud.word.map(len)>3)]
        n_words = dfcloud['word'].nunique()

        if n_words < 1:
            st.error('No words with a specified threshold. \n Try lower value of threshold.')
            dfcloud = df_for_wordcloud[ (df_for_wordcloud['general #'] > 1) & (df_for_wordcloud.word.map(len)>3) ]
            df_cloud_words1 = dfcloud.copy()
            df_cloud_words1 = df_cloud_words1.rename(columns = {'general #':'overall #', 'precis':'precision'})
            df_cloud_words1 = df_cloud_words1.sort_values(by = 'precision', ascending = False)
            df_cloud_words1 = df_cloud_words1.reset_index(drop = True)
            df_cloud_words1.index += 1
            #st.stop()

        else:
            text = []
            for i in dfcloud.index:
              w = dfcloud.loc[i, 'word']
              w = str(w).strip()
              if analysis_for == 'both':
                n = int(dfcloud.loc[i, 'support #'] + dfcloud.loc[i, 'attack #'])
              else:
                n = int(dfcloud.loc[i, str(analysis_for)+' #']) #  + dfcloud.loc[i, 'attack #']   dfcloud.loc[i, 'support #']+  general
              l = np.repeat(w, n)
              text.extend(l)

            import random
            random.shuffle(text)

            figure_cloud, figure_cloud_words = make_word_cloud(" ".join(text), 1000, 620, '#1E1E1E', str(cmap_wordcloud), stops = box_stopwords)
            #st.write(f"There are {len(figure_cloud_words)} words.")
            #st.pyplot(figure_cloud)

            df_cloud_words1 = dfcloud.copy()
            df_cloud_words1 = df_cloud_words1.rename(columns = {'general #':'overall #', 'precis':'precision'})
            df_cloud_words1 = df_cloud_words1.sort_values(by = 'precision', ascending = False)
            df_cloud_words1 = df_cloud_words1.reset_index(drop = True)
            df_cloud_words1.index += 1
            #st.write(df_cloud_words1)

            st.write(f"There are {len(figure_cloud_words)} words.")
            st.pyplot(figure_cloud)


    with heroes_tab2:
        add_spacelines(1)
        st.write(corp)
        st.write(df_cloud_words1)


    with heroes_tab_explore:
        add_spacelines()
        st.write(corp)
        if n_words < 1:
            st.error('No words with a specified threshold. \n Try lower value of threshold.')
            st.stop()

        cols_odds1 = ['source', 'sentence', 'ethos_label',  'Target',  'freq_words_'+label_cloud, 'corpus']

        if selected_rhet_dim == 'logos':
            df = df.rename(columns = {'connection':'logos'})
            #cols_odds1 = ['locution_conclusion', 'locution_premise', 'logos', 'argument_linked', 'freq_words_'+label_cloud]
            cols_odds1 = ['locution_conclusion', 'locution_premise','premise', 'conclusion', 'sentence_lemmatized', 'logos', 'argument_linked', 'freq_words_'+label_cloud]
            df['sentence_lemmatized'] = df['sentence_lemmatized'].astype('str')
            df['logos'] = df['logos'].map({'Default Inference':'support', 'Default Conflict':'attack'})

        pos_list_freq = df_cloud_words1.word.tolist()
        freq_word_pos = st.multiselect('Choose word(s) you would like to see data cases for', pos_list_freq, pos_list_freq[:2])
        df_odds_pos_words = set(freq_word_pos)
        df['freq_words_'+label_cloud] = df.sentence_lemmatized.apply(lambda x: " ".join( set(x.split()).intersection(df_odds_pos_words) ))
        #st.write(df)
        if "&" in df.corpus.iloc[0]:
            ds = "Covid & ElectionsSM"
            df['corpus'] = ds
        add_spacelines(1)
        st.write(f'Cases with **{freq_word_pos}** words:')
        st.dataframe(df[ (df['freq_words_'+label_cloud].str.split().map(len) >= 1) ].dropna(axis=1, how='all')[cols_odds1])# .set_index('source')




def generateWordCloud_compare(df_list, selected_rhet_dim, label_cloud, threshold_cloud, rhetoric_dims = ['ethos', 'pathos']):
    add_spacelines(1)

    if selected_rhet_dim != 'logos':
        df = df_list[0]
        df = clean_text(df, 'sentence_lemmatized', text_column_name = "sentence_lemmatized")
        st.write( f" **{df.corpus.iloc[0]}** " )
        if not 'neutral' in df['ethos_label'].unique():
            df['ethos_label'] = df['ethos_label'].map(ethos_mapping)
        if not 'negative' in df['pathos_label'].unique():
            df['pathos_label'] = df['pathos_label'].map(valence_mapping)

    elif selected_rhet_dim == 'logos':
        df = df_list[-1] #pd.concat(df_list, axis=0, ignore_index=True)
        st.write( f" **{df.corpus.iloc[0]}** " )
        df = df.dropna(subset = 'premise')
        df['sentence_lemmatized'] = df['premise'].astype('str').apply(lambda x: re.sub(r"\W+", " ", str(x)))
        df = lemmatization(df, 'sentence_lemmatized', name_column = True)
        df['sentence_lemmatized'] = df['sentence_lemmatized'].astype('str').str.lower().str.replace('ahould', 'should')


    if (selected_rhet_dim == 'ethos_label'):
         df_for_wordcloud = prepare_cloud_lexeme_data(df[df[str(selected_rhet_dim)] == 'neutral'],
         df[df[str(selected_rhet_dim)] == 'support'],
         df[df[str(selected_rhet_dim)] == 'attack'])

    elif selected_rhet_dim == 'logos':
         df_for_wordcloud = prepare_cloud_lexeme_data(df[ ~(df['connection'].isin(['Default Inference', 'Default Conflict'])) ],
         df[df['connection'] == 'Default Inference'],
         df[df['connection'] == 'Default Conflict'])
    else:
        df_for_wordcloud = prepare_cloud_lexeme_data(df[df[str(selected_rhet_dim)] == 'neutral'],
        df[df[str(selected_rhet_dim)] == 'positive'],
        df[df[str(selected_rhet_dim)] == 'negative'])

    fig_cloud1, df_cloud_words1, figure_cloud_words1 = wordcloud_lexeme(df_for_wordcloud, lexeme_threshold = threshold_cloud, analysis_for = str(label_cloud))
    cc = df.corpus.iloc[0]

    #st.pyplot(fig_cloud1)

    add_spacelines(2)

    #st.write(f'WordCloud frequency table: ')
    if selected_rhet_dim == 'pathos_label':

        label_cloud = label_cloud.replace('attack', 'negative').replace('support', 'positive')
        df_cloud_words1 = df_cloud_words1.rename(columns = {
        'precis':'precision',
        'attack #':'negative #',
        'general #':'overall #',
        'support #':'positive #',
        })
    else:
        df_cloud_words1 = df_cloud_words1.rename(columns = {'general #':'overall #', 'precis':'precision'})

    df_cloud_words1 = df_cloud_words1.sort_values(by = 'precision', ascending = False)
    df_cloud_words1 = df_cloud_words1.reset_index(drop = True)
    df_cloud_words1.index += 1
    #st.write(df_cloud_words1)


    cols_odds1 = ['source', 'sentence', 'ethos_label', 'pathos_label', 'Target',
                         'freq_words_'+label_cloud]

    if selected_rhet_dim == 'logos':
        df = df.rename(columns = {'connection':'logos'})
        #cols_odds1 = ['locution_conclusion', 'locution_premise', 'logos', 'argument_linked', 'freq_words_'+label_cloud]
        cols_odds1 = ['premise', 'conclusion', 'sentence_lemmatized', 'logos', 'freq_words_'+label_cloud]
        df['sentence_lemmatized'] = df['sentence_lemmatized'].astype('str')
        df['logos'] = df['logos'].map({'Default Inference':'support', 'Default Conflict':'attack'})

    pos_list_freq = df_cloud_words1.word.tolist()
    freq_word_pos = st.multiselect('Choose word(s) you would like to see data cases for', pos_list_freq, pos_list_freq[:2])
    df_odds_pos_words = set(freq_word_pos)
    df['freq_words_'+label_cloud] = df.sentence_lemmatized.apply(lambda x: " ".join( set(x.split()).intersection(df_odds_pos_words) ))
    #st.write(df)
    add_spacelines(1)
    #st.write(f'Cases with **{freq_word_pos}** words:')
    dd = df[ (df['freq_words_'+label_cloud].str.split().map(len) >= 1) & (df[selected_rhet_dim] == label_cloud) ][cols_odds1]
    #st.dataframe(dd)# .set_index('source')
    return fig_cloud1, df_cloud_words1, freq_word_pos, dd, cc




def generateWordCloud(df_list, rhetoric_dims = ['ethos', 'pathos'], an_type = 'ADU-based'):
    #st.header(f" Text-Level Analytics ")

    selected_rhet_dim = st.selectbox("Choose a rhetoric category for a WordCloud", rhetoric_dims, index=0)
    add_spacelines(1)
    if selected_rhet_dim == 'pathos':
        label_cloud = st.radio("Choose a label for words in WordCloud", ('negative', 'positive'))
        selected_rhet_dim = selected_rhet_dim.replace("ethos", "ethos_label").replace("pathos", "pathos_label")
        label_cloud = label_cloud.replace("negative", "attack").replace("positive", "support")
    else:
        label_cloud = st.radio("Choose a label for words in WordCloud", ('attack', 'support'))
        selected_rhet_dim = selected_rhet_dim.replace("ethos", "ethos_label")
        label_cloud = label_cloud.replace("attack / negative", "attack").replace("support / positive", "support")

    add_spacelines(1)
    threshold_cloud = st.slider('Select a precision value (threshold) for words in WordCloud', 0, 100, 80)
    st.info(f'Selected precision: **{threshold_cloud}**')
    box_stopwords = st.checkbox( "Enable stop words", value = False )


    if selected_rhet_dim != 'logos':
        df = df_list[0]
        df = clean_text(df, 'sentence_lemmatized', text_column_name = "sentence_lemmatized")
        df = df.drop_duplicates()
        st.write( f" Corpus: **{df.corpus.iloc[0]}** " )
        add_spacelines(1)

        if not 'neutral' in df['ethos_label'].unique():
            df['ethos_label'] = df['ethos_label'].map(ethos_mapping)
        if not 'negative' in df['pathos_label'].unique():
            df['pathos_label'] = df['pathos_label'].map(valence_mapping)

    elif selected_rhet_dim == 'logos':
        df = df_list[-1] #pd.concat(df_list, axis=0, ignore_index=True)
        st.write( f" Corpus: **{df.corpus.iloc[0]}** " )
        add_spacelines(1)
        df = df.dropna(subset = 'premise').drop_duplicates()
        df['sentence_lemmatized'] = df['premise'].astype('str').apply(lambda x: re.sub(r"\W+", " ", str(x)))
        df = lemmatization(df, 'sentence_lemmatized', name_column = True)
        df['sentence_lemmatized'] = df['sentence_lemmatized'].astype('str').str.lower().str.replace('ahould', 'should')


    if (selected_rhet_dim == 'ethos_label'):
         df_for_wordcloud = prepare_cloud_lexeme_data(df[df[str(selected_rhet_dim)] == 'neutral'],
         df[df[str(selected_rhet_dim)] == 'support'],
         df[df[str(selected_rhet_dim)] == 'attack'])

    elif selected_rhet_dim == 'logos':
         df_for_wordcloud = prepare_cloud_lexeme_data(df[ ~(df['connection'].isin(['Default Inference', 'Default Conflict'])) ],
         df[df['connection'] == 'Default Inference'],
         df[df['connection'] == 'Default Conflict'])
    else:
        df_for_wordcloud = prepare_cloud_lexeme_data(df[df[str(selected_rhet_dim)] == 'neutral'],
        df[df[str(selected_rhet_dim)] == 'positive'],
        df[df[str(selected_rhet_dim)] == 'negative'])

    fig_cloud1, df_cloud_words1, figure_cloud_words1 = wordcloud_lexeme(df_for_wordcloud, lexeme_threshold = threshold_cloud, analysis_for = str(label_cloud), stops = box_stopwords)

    plot_tab, table_tab, explore_tab = st.tabs( ['Plot', 'Table', 'Cases'] )

    with plot_tab:
        add_spacelines(1)
        st.write(f"There are {len(figure_cloud_words1)} words.")
        st.pyplot(fig_cloud1)


    with table_tab:
        add_spacelines(1)
        st.write(f'WordCloud frequency table: ')

        if selected_rhet_dim == 'pathos_label':
            label_cloud = label_cloud.replace('attack', 'negative').replace('support', 'positive')
            df_cloud_words1 = df_cloud_words1.rename(columns = {
            'precis':'precision',
            'attack #':'negative #',
            'general #':'overall #',
            'support #':'positive #',
            })
        else:
            df_cloud_words1 = df_cloud_words1.rename(columns = {'general #':'overall #', 'precis':'precision'})

        df_cloud_words1 = df_cloud_words1.sort_values(by = 'precision', ascending = False).drop_duplicates()
        df_cloud_words1 = df_cloud_words1.reset_index(drop = True)
        df_cloud_words1.index += 1
        st.write(df_cloud_words1)


    with explore_tab:
        add_spacelines(1)
        cols_odds1 = ['source', 'sentence', 'ethos_label', 'pathos_label', 'Target', 'freq_words_'+label_cloud]

        if selected_rhet_dim == 'logos':
            df = df.rename(columns = {'connection':'logos'})
            #cols_odds1 = ['locution_conclusion', 'locution_premise', 'logos', 'argument_linked', 'freq_words_'+label_cloud]
            cols_odds1 = ['locution_conclusion', 'locution_premise','premise', 'conclusion', 'sentence_lemmatized', 'logos', 'argument_linked', 'freq_words_'+label_cloud]
            df['sentence_lemmatized'] = df['sentence_lemmatized'].astype('str')
            df['logos'] = df['logos'].map({'Default Inference':'support', 'Default Conflict':'attack'})

        pos_list_freq = df_cloud_words1.word.tolist()
        freq_word_pos = st.multiselect('Choose word(s) you would like to see data cases for', pos_list_freq, pos_list_freq[:2])
        df_odds_pos_words = set(freq_word_pos)
        df['freq_words_'+label_cloud] = df.sentence_lemmatized.apply(lambda x: " ".join( set(x.split()).intersection(df_odds_pos_words) ))
        #st.write(df)
        add_spacelines(1)
        st.write(f'Cases with **{freq_word_pos}** words:')
        st.dataframe(df[ (df['freq_words_'+label_cloud].str.split().map(len) >= 1) & (df[selected_rhet_dim] == label_cloud) ].drop_duplicates()[cols_odds1])# .set_index('source')




def ProfilesEntity_compare(data_list, selected_rhet_dim):

        up_data_dict = {}
        up_data_dicth = {}
        up_data_dictah = {}
        target_shared = {}
        up_data_dict_hist = {}
        #n = 0
        #for data in data_list:
        #heroes_tab1, heroes_tab2 = st.tabs(['Overview', 'Single Case Analysis'])
        #with heroes_tab1:
        df = data_list[0].copy()
        #st.dataframe(df)
        ds = df['corpus'].iloc[0]
        add_spacelines(1)
        st.write("##### Profiles Overview")
        dds = df.groupby('source', as_index=False).size()
        dds = dds[dds['size']>2]
        ddt = df.groupby('Target', as_index=False).size()
        ddt = ddt[ddt['size']>2]

        df = df[(df.source.isin(dds.source.values)) | (df.Target.isin(ddt.Target.values))]

        if not 'attack' in df['ethos_label'].unique():
            df['ethos_label'] = df['ethos_label'].map(ethos_mapping)
        if not 'pathos_label' in df.columns:
            df['pathos_label'] = 'neutral'
        if not 'neutral' in df['pathos_label'].unique():
            df['pathos_label'] = df['pathos_label'].map(valence_mapping)

        df["source"] = df["source"].astype('str').str.replace("@", "")
        df["Target"] = df["Target"].astype('str').str.replace("@", "")
        #df = df[ (df.Target != 'nan') & (df.Target != '') & (df.ethos_label != 'neutral') ]
        df["Target"] = df["Target"].str.replace('Government', 'government')
        selected_rhet_dim_n_cats = {'pathos_label':2, 'ethos_label':2, 'sentiment':2}
        selected_rhet_dim_subcats = {'pathos_label':['negative', 'positive'],
                                    'ethos_label': ['attack', 'support'],
                                     'sentiment':['negative', 'positive']}

        dd2_sizet = pd.DataFrame(df[df[selected_rhet_dim] != 'neutral'].groupby(['Target'])[selected_rhet_dim].value_counts(normalize=True).round(3)*100)
        #dd2_sizet = pd.DataFrame(df.groupby(['Target'])[selected_rhet_dim].value_counts(normalize=True).round(3)*100)
        dd2_sizet.columns = ['percentage']
        dd2_sizet = dd2_sizet.reset_index()
        adj_target = dd2_sizet.Target.unique()
        for t in adj_target:
            dd_adj_target =  dd2_sizet[dd2_sizet.Target == t]
            if dd_adj_target.shape[0] != selected_rhet_dim_n_cats[selected_rhet_dim]:
                if 'support' in dd_adj_target[selected_rhet_dim].values and not 'attack' in dd_adj_target[selected_rhet_dim].values:
                    dd2_sizet.loc[len(dd2_sizet)] = [t, 'attack', 0]
                elif not 'support' in dd_adj_target[selected_rhet_dim].values and 'attack' in dd_adj_target[selected_rhet_dim].values:
                    dd2_sizet.loc[len(dd2_sizet)] = [t, 'support', 0]
                elif not 'negative' in dd_adj_target[selected_rhet_dim].values and ('positive' in dd_adj_target[selected_rhet_dim].values):
                    dd2_sizet.loc[len(dd2_sizet)] = [t, 'negative', 0]
                elif not 'positive' in dd_adj_target[selected_rhet_dim].values and ('negative' in dd_adj_target[selected_rhet_dim].values):
                    dd2_sizet.loc[len(dd2_sizet)] = [t, 'positive', 0]
                #elif not 'neutral' in dd_adj_target[selected_rhet_dim].values and ('positive' in dd_adj_target[selected_rhet_dim].values or 'negative' in dd_adj_target[selected_rhet_dim].values):
                    #dd2_sizet.loc[len(dd2_sizet)] = [t, 'neutral', 0]


        #if selected_rhet_dim == 'ethos_label':
            #dd2_sizes = pd.DataFrame(df[df[selected_rhet_dim] != 'neutral'].groupby(['source'])[selected_rhet_dim].value_counts(normalize=True).round(3)*100)
        #else:
            #dd2_sizes = pd.DataFrame(df.groupby(['source'])[selected_rhet_dim].value_counts(normalize=True).round(3)*100)
        dd2_sizes = pd.DataFrame(df[df[selected_rhet_dim] != 'neutral'].groupby(['source'])[selected_rhet_dim].value_counts(normalize=True).round(3)*100)

        dd2_sizes.columns = ['percentage']
        dd2_sizes = dd2_sizes.reset_index()
        adj_source = dd2_sizes.source.unique()
        for t in adj_source:
            dd_adj_source =  dd2_sizes[dd2_sizes.source == t]
            if dd_adj_source.shape[0] != selected_rhet_dim_n_cats[selected_rhet_dim]:
                if 'support' in dd_adj_source[selected_rhet_dim].values and not 'attack' in dd_adj_source[selected_rhet_dim].values:
                    dd2_sizes.loc[len(dd2_sizes)] = [t, 'attack', 0]
                elif not 'support' in dd_adj_source[selected_rhet_dim].values and 'attack' in dd_adj_source[selected_rhet_dim].values:
                    dd2_sizes.loc[len(dd2_sizes)] = [t, 'support', 0]
                elif not 'negative' in dd_adj_source[selected_rhet_dim].values and ('positive' in dd_adj_source[selected_rhet_dim].values or 'neutral' in dd_adj_source[selected_rhet_dim].values):
                    dd2_sizes.loc[len(dd2_sizes)] = [t, 'negative', 0]
                elif not 'positive' in dd_adj_source[selected_rhet_dim].values and ('negative' in dd_adj_source[selected_rhet_dim].values or 'neutral' in dd_adj_source[selected_rhet_dim].values):
                    dd2_sizes.loc[len(dd2_sizes)] = [t, 'positive', 0]
                elif not 'neutral' in dd_adj_source[selected_rhet_dim].values and ('positive' in dd_adj_source[selected_rhet_dim].values or 'negative' in dd_adj_source[selected_rhet_dim].values):
                    dd2_sizes.loc[len(dd2_sizes)] = [t, 'neutral', 0]

        dd2_sizet.columns = ['entity', 'category', 'percentage']
        dd2_sizes.columns = ['entity', 'category', 'percentage']
        dd2_sizet['role'] = 'passive'
        dd2_sizes['role'] = 'active'

        #if selected_rhet_dim == 'ethos_label':
        dd2_sizet = dd2_sizet[dd2_sizet.category != 'neutral']
        dd2_sizes = dd2_sizes[dd2_sizes.category != 'neutral']

        dd2_size = pd.concat([dd2_sizes, dd2_sizet], axis = 0, ignore_index = True)

        #st.write(dd2_size)

        #dd2_size = pd.pivot_table(dd2_size, values='percentage', index=['entity', 'role'], columns=['category'], aggfunc=np.sum)
        cat_neg = selected_rhet_dim_subcats[selected_rhet_dim][0]
        cat_pos = selected_rhet_dim_subcats[selected_rhet_dim][1]
        plt1 = {cat_neg:'darkred', cat_pos:'darkgreen'}
        dd2_size.percentage = dd2_size.percentage.round(-1).astype('int')

        #dd2_size['profile'] = np.where( (dd2_size['role'] = 'passive') & (dd2_size[cat_neg] > dd2_size[cat_pos]) )
        #fig_pr = sns.catplot(kind = 'count', data = dd2_size, y = 'percentage',
        #        col = 'role', hue = 'category', palette = plt1, aspect = 1.3)
        #st.pyplot(fig_pr)

        dd2_size = pd.pivot_table(dd2_size, values='percentage', index=['entity', 'role'], columns=['category'], aggfunc=np.sum)
        dd2_size = dd2_size.reset_index()
        #st.write(dd2_size)


        #dd2_sizet.columns = ['entity', 'category', 'percentage']
        dd2_sizet = pd.pivot_table(dd2_sizet, values='percentage', index=['entity', 'role'], columns=['category'], aggfunc=np.sum).reset_index()
        #dd2_sizes.columns = ['entity', 'category', 'percentage']
        dd2_sizes = pd.pivot_table(dd2_sizes, values='percentage', index=['entity', 'role'], columns=['category'], aggfunc=np.sum).reset_index()
        #st.write(dd2_sizes)
        dd2_sizet['role'] = 1
        dd2_sizes['role'] = 1

        dd2_sizes = dd2_sizes.rename(columns = {'role':'active', cat_neg:cat_neg+'_active', cat_pos:cat_pos+'_active'})
        dd2_sizet = dd2_sizet.rename(columns = {'role':'passive', cat_neg:cat_neg+'_passive', cat_pos:cat_pos+'_passive'})

        dd2_size_2 = dd2_sizes.merge(dd2_sizet, on = 'entity')
        #st.write(dd2_size_2)

        dd2_size_2['profile'] = 'other'
        dd2_size_2['profile'] = np.where((dd2_size_2[cat_neg+'_active'] > dd2_size_2[cat_pos+'_active']) & (dd2_size_2[cat_neg+'_passive'] > dd2_size_2[cat_pos+'_passive']), 'angry man', dd2_size_2['profile'] )
        dd2_size_2['profile'] = np.where((dd2_size_2[cat_neg+'_active'] < dd2_size_2[cat_pos+'_active']) & (dd2_size_2[cat_neg+'_passive'] < dd2_size_2[cat_pos+'_passive']), 'positive soul', dd2_size_2['profile'] )
        dd2_size_2['profile'] = np.where((dd2_size_2[cat_neg+'_active'] > dd2_size_2[cat_pos+'_active']) & (dd2_size_2[cat_neg+'_passive'] <= dd2_size_2[cat_pos+'_passive']), 'attacker', dd2_size_2['profile'] )
        dd2_size_2['profile'] = np.where((dd2_size_2[cat_neg+'_active'] < dd2_size_2[cat_pos+'_active']) & (dd2_size_2[cat_neg+'_passive'] >=  dd2_size_2[cat_pos+'_passive']), 'supporter', dd2_size_2['profile'] )

        dd2_size_2['profile'] = np.where((dd2_size_2[cat_neg+'_active'] == dd2_size_2[cat_pos+'_active']) & (dd2_size_2[cat_neg+'_passive'] > dd2_size_2[cat_pos+'_passive']), 'negative undecided', dd2_size_2['profile'] )
        dd2_size_2['profile'] = np.where((dd2_size_2[cat_neg+'_active'] == dd2_size_2[cat_pos+'_active']) & (dd2_size_2[cat_neg+'_passive'] < dd2_size_2[cat_pos+'_passive']), 'positive undecided', dd2_size_2['profile'] )

        dd2_size_2_pr = pd.DataFrame( dd2_size_2['profile'].value_counts(normalize = True).round(3)*100)
        dd2_size_2_pr = dd2_size_2_pr.reset_index()
        dd2_size_2_pr.columns = ['profile', 'percentage']
        plt2 = {'angry man':'darkred', 'positive soul':'darkgreen', 'attacker':'darkred', 'supporter':'darkgreen',
        'negative undecided':'red', 'positive undecided':'green', 'other':'grey'}
        if dd2_size_2_pr.shape[0] != len(plt2.keys()):
            miss_pr =  set(plt2.keys()).difference( set(dd2_size_2_pr.profile.values))
            for p in miss_pr:
                dd2_size_2_pr.loc[len(dd2_size_2_pr)] = [p, 0]
        #st.write(dd2_size_2_pr)
        dd2_size_2_pr = dd2_size_2_pr.sort_values(by = 'profile')

        #st.write(dd2_size_2[dd2_size_2['profile'] == 'other'])
        fig_pr = sns.catplot(kind = 'bar', data = dd2_size_2_pr, x = 'percentage', y = 'profile',
                aspect = 1.65, palette = plt2)
        titl_fig = selected_rhet_dim.replace('_label', '')
        fig_pr.set(title = f'{titl_fig.capitalize()} profiles in {ds}', xlim = (0,100), xticks = np.arange(0, 100, 15))
        st.pyplot(fig_pr)


        dd2_size_2[[cat_pos+'_active', cat_pos+'_passive', cat_neg+'_active', cat_neg+'_passive']] = dd2_size_2[[cat_pos+'_active', cat_pos+'_passive', cat_neg+'_active', cat_neg+'_passive']].round(-1)
        #cat_pos+'_active', cat_pos+'_passive'
        dd2_size_2_melt_ac = dd2_size_2[['entity', cat_pos+'_active', cat_neg+'_active']].melt('entity')
        dd2_size_2_melt_ac['role'] = 'active'
        dd2_size_2_melt_ac['category'] = dd2_size_2_melt_ac['category'].str.replace('_active', '')

        dd2_size_2_melt_ps = dd2_size_2[['entity', cat_pos+'_passive', cat_neg+'_passive']].melt('entity')
        dd2_size_2_melt_ps['role'] = 'passive'
        dd2_size_2_melt_ps['category'] = dd2_size_2_melt_ps['category'].str.replace('_passive', '')

        dd2_size_2_melt = pd.concat([dd2_size_2_melt_ps, dd2_size_2_melt_ac], axis=0, ignore_index = True)
        dd2_size_2_melt_dd = pd.DataFrame(dd2_size_2_melt.groupby(['role', 'category'])['value'].value_counts(normalize = True).round(3)*100)
        dd2_size_2_melt_dd.columns = ['percentage']
        dd2_size_2_melt_dd = dd2_size_2_melt_dd.reset_index()
        for r in dd2_size_2_melt_dd.role.unique():
            for c in dd2_size_2_melt_dd.category.unique():
                for v in np.arange(0, 101, 10):
                    if not v in dd2_size_2_melt_dd[ (dd2_size_2_melt_dd.role == r) & (dd2_size_2_melt_dd.category == c) ]['value'].unique():
                        dd2_size_2_melt_dd.loc[len(dd2_size_2_melt_dd)] = [r, c, v, 0]
        dd2_size_2_melt_dd = dd2_size_2_melt_dd.sort_values(by = ['role', 'category', 'value'])
        dd2_size_2_melt_dd['value'] = dd2_size_2_melt_dd['value'].astype('int').astype('str')


        #st.write(dd2_size_2)
        dd2_size_2_melt['category'] = np.where(dd2_size_2_melt['value'] == 50, cat_pos + " & " + cat_neg, dd2_size_2_melt['category'])
        dd2_size_2_melt = dd2_size_2_melt.sort_values(by = ['entity', 'role', 'value'])
        dd2_size_2_melt2 = dd2_size_2_melt.drop_duplicates(subset = ['entity', 'role'], keep = 'last')
        dd2_size_2_melt2_grp = pd.DataFrame(dd2_size_2_melt2.groupby([ 'role' ]).category.value_counts(normalize = True).round(3)*100)
        dd2_size_2_melt2_grp.columns = ['percentage']
        dd2_size_2_melt2_grp = dd2_size_2_melt2_grp.reset_index()
        #st.write(dd2_size_2_melt2_grp)
        #st.write(dd2_size_2_melt2)
        colors[cat_pos + " & " + cat_neg] = '#CB7200'

        st.write("##### Roles Overview")
        sns.set(font_scale=1.6, style='whitegrid')
        fig_ac_pas = sns.catplot(kind = 'bar', data = dd2_size_2_melt2_grp, y = 'category', x = 'percentage',
                    col = 'role', hue = 'category', palette = colors,
                    dodge=False, aspect = 1.3, height = 6, legend = False)

        plt.tight_layout(pad=2)
        if max(dd2_size_2_melt2_grp.percentage) == 100:
            mm = 100
        else:
            mm = max(dd2_size_2_melt2_grp.percentage)+11

        fig_ac_pas.set(xticks = np.arange(0, mm, 10))
        st.pyplot(fig_ac_pas)

        #st.stop()
        #add_spacelines()
        st.write("*********************************************************")


        st.write("##### Single Case Analysis")
        dd2_size_2[[cat_pos+'_active', cat_pos+'_passive', cat_neg+'_active', cat_neg+'_passive']] = dd2_size_2[[cat_pos+'_active', cat_pos+'_passive', cat_neg+'_active', cat_neg+'_passive']].applymap(lambda x: 1 if x == 0 else x)
        #with heroes_tab2:
        dd2_size_2 = dd2_size_2.drop(columns = ['active', 'passive'], axis = 1)
        dd2_size_2.sort_values(by = [cat_neg+'_active', cat_pos+'_active', cat_neg+'_passive', cat_pos+'_passive'], ascending = False)
        vals = list(set(dd2_size_2.entity.values))
        select_box_prof = st.multiselect("Select an entity", vals, vals[:3])
        #st.write(dd2_size_2)
        add_spacelines(2)

        dd2_size_2_s = dd2_size_2[dd2_size_2.entity.isin(select_box_prof)]
        dd2_size_2_s = dd2_size_2_s.iloc[:, :-1].melt('entity', var_name = 'category', value_name = 'percentage')
        dd2_size_2_s['role'] = np.where(dd2_size_2_s.category.isin([cat_neg+'_active', cat_pos+'_active']), 'active', 'passive')
        dd2_size_2_s['category'] = dd2_size_2_s['category'].str.replace('_active', '').str.replace('_passive', '')

        plt3 = {cat_neg+"_active":'darkred', cat_pos+"_active":'darkgreen', cat_pos+"_passive":'green', cat_neg+"_passive":'red'}
        sns.set(font_scale=1.35, style='whitegrid')

        #st.write(dd2_size_2_s)

        fig_pr = sns.catplot(kind = 'bar', data = dd2_size_2_s, y = 'category', x = 'percentage',
                col = 'role', hue = 'category', row = 'entity', alpha = 0.9,
                palette = plt1, aspect = 1.3, dodge=False)
        for ax in fig_pr.axes.flatten():
            ax.tick_params(labelbottom=True, bottom=True)
        #labelbottom
        plt.tight_layout(pad=2.2)
        sns.move_legend(fig_pr, loc = 'upper right', bbox_to_anchor = (0.63, 1.04), ncol = 3)
        st.pyplot(fig_pr)
        add_spacelines(2)
        dd2_size_2_s2 = dd2_size_2[dd2_size_2.entity.isin(select_box_prof)]
        dd2_size_2_s2[[cat_neg+'_active', cat_pos+'_active', cat_neg+'_passive', cat_pos+'_passive']] = dd2_size_2_s2[[cat_neg+'_active', cat_pos+'_active', cat_neg+'_passive', cat_pos+'_passive']].applymap(lambda x: 0 if x == 1 else x)
        #dd2_size_2_s.percentage = np.where(dd2_size_2_s.percentage == 1, 0, dd2_size_2_s.percentage)
        st.write(dd2_size_2_s2.set_index('entity'))
        #st.write(dd2_size_2_s.sort_values(by = 'entity').set_index('entity'))
        #st.stop()
        add_spacelines(2)

        with st.expander('Profile names'):
            st.write('**Angry man**')
            st.write("""If negativity of an entity dominates her both active and passive roles.
            That is, the entity is both negative towards others (i.e., uses ethotic attacks/negative emotions) and others are negative towards this entity (i.e., others also use ethotic attacks/negative emotions when mentioning this entity).""")

            st.write('**Attacker**')
            st.write("""If negativity of an entity dominates her active role; a passive role is either positive or ambivalent (both positive and negative).
            That is, the entity is negative towards others (i.e., uses ethotic attacks/negative emotions) but others are positive or both positive and negative towards this entity (e.g., someone attacks her and someone supports her).""")

            st.write('**Negative undecided**')
            st.write("""If the active role of an entity is equally negative and positive; a passive role is negative.
            That is, the entity is negative towards some people and positive towards other but others are negative towards this entity (e.g., others attack her).""")

            st.write('**Positive soul**')
            st.write("""If positivity of an entity dominates her both active and passive roles.
            That is, the entity is both positive towards others (i.e., uses ethotic supports/positive emotions) and others are positive towards this entity (i.e., others also use ethotic supports/positive emotions when mentioning this entity).""")

            st.write('**Supporter**')
            st.write("""If positivity of an entity dominates her both active role; a passive role is either positive or ambivalent (both positive and negative).
            That is, the entity is positive towards others (i.e., uses ethotic supports/positive emotions) but others are positive or both positive and negative towards this entity (e.g., someone attacks her and someone supports her).""")

            st.write('**Positive undecided**')
            st.write("""If the active role of an entity is equally negative and positive; a passive role is positive.
            That is, the entity is negative towards some people and positive towards other but others are positive towards this entity (e.g., others support her).""")

            st.write('**Other**')
            st.write("""If an entity could not be assigned to any of the above categories.""")





def TargetHeroScores_compare(data_list, singl_an = True):
    st.write("### Villains & heroes")
    add_spacelines(1)
    contents_radio_heroes = st.radio("Category of the target of ethotic statements", ("both", "direct ethos", "3rd party ethos"))
    contents_radio_unit = st.radio("Unit of analysis", ("score", "number"))

    up_data_dict = {}
    up_data_dicth = {}
    up_data_dictah = {}
    target_shared = {}
    up_data_dict_hist = {}

    n = 0
    for data in data_list:
        df = data.copy()
        ds = df['corpus'].iloc[0]
        if not 'attack' in df['ethos_label'].unique():
            df['ethos_label'] = df['ethos_label'].map(ethos_mapping)
        df["Target"] = df["Target"].astype('str')
        df = df[ (df.Target != 'nan') & (df.Target != '') & (df.ethos_label != 'neutral') ]
        df["Target"] = df["Target"].str.replace('Government', 'government')
        target_shared[n] = set(df["Target"].unique())

        if contents_radio_heroes == "direct ethos":
            targets_limit = df['Target'].dropna().unique()
            targets_limit = [t for t in targets_limit if "@" in t]
            df = df[df.Target.isin(targets_limit)]
            if len(targets_limit) < 2:
                st.error(f'No cases of **{contents_radio_heroes}** found in the chosen corpora.')
                st.stop()
        elif contents_radio_heroes == "3rd party ethos":
            targets_limit = df['Target'].dropna().unique()
            targets_limit = [t for t in targets_limit if not "@" in t]
            df = df[df.Target.isin(targets_limit)]
            if len(targets_limit) < 2:
                st.error(f'No cases of **{contents_radio_heroes}** found in the chosen corpora.')
                st.stop()

        dd2_size = df.groupby(['Target'], as_index=False).size()
        dd2_size = dd2_size[dd2_size['size'] > 1]
        adj_target = dd2_size['Target'].unique()

        dd = pd.DataFrame(df.groupby(['Target'])['ethos_label'].value_counts(normalize=False))
        dd.columns = ['value']
        dd = dd.reset_index()
        dd = dd[dd.Target.isin(adj_target)]
        dd = dd[dd.ethos_label != 'neutral']
        dd_hero = dd[dd.ethos_label == 'support']
        dd_antihero = dd[dd.ethos_label == 'attack']

        dd2 = pd.DataFrame({'Target': dd.Target.unique()})
        dd2_hist = dd2.copy()
        dd2anti_scores = []
        dd2hero_scores = []

        if contents_radio_unit == 'score':
            dd2['score'] = np.nan
            dd2['number'] = np.nan
            dd2['appeals'] = np.nan
            for t in dd.Target.unique():
                try:
                    h = dd_hero[dd_hero.Target == t]['value'].iloc[0]
                except:
                    h = 0
                try:
                    ah = dd_antihero[dd_antihero.Target == t]['value'].iloc[0]
                except:
                    ah = 0
                dd2hero_scores.append(h)
                dd2anti_scores.append(ah)
                i = dd2[dd2.Target == t].index
                dd2.loc[i, 'score'] = ah / (ah + h)
                if h > ah:
                    dd2.loc[i, 'number'] = h
                    dd2.loc[i, 'appeals'] = (ah + h)
                    dd2.loc[i, 'ethos_label'] = 'heroes'
                elif h < ah:
                    dd2.loc[i, 'number'] = ah
                    dd2.loc[i, 'ethos_label'] = 'villains'
                    dd2.loc[i, 'appeals'] = (ah + h)
                else:
                    dd2.loc[i, 'number'] = 0
                    dd2.loc[i, 'ethos_label'] = 'nn'
                    dd2.loc[i, 'appeals'] = (ah + h)

            dd2 = dd2[dd2.score != 0]
            dd2 = dd2[dd2.ethos_label != 'nn']
            #dd2['ethos_label'] = np.where(dd2.score < 0, 'villains', 'neutral')
            #dd2['ethos_label'] = np.where(dd2.score > 0, 'heroes', dd2['ethos_label'])
            dd2 = dd2.sort_values(by = ['ethos_label', 'Target'])
            #dd2['score'] = dd2['score'] * 100
            #dd2['score'] = dd2['score'].round()
            dd2['corpus'] = ds
            #st.write(dd2)
            up_data_dict_hist[n] = dd2
            #st.write(dd2)
            #st.stop()
            #dd2['score'] = dd2[dd2['ethos_label'] != 'neutral' ]
            dd2_dist = pd.DataFrame(dd2['ethos_label'].value_counts(normalize=True).round(3)*100).reset_index()
            dd2_dist.columns = ['heroes', 'percentage']
            dd2_dist['corpus'] = ds
            up_data_dict[n] = dd2_dist
            up_data_dicth[n] = dd2[dd2['ethos_label'] == 'heroes']['Target'].unique()
            up_data_dictah[n] = dd2[dd2['ethos_label'] == 'villains']['Target'].unique()
            n += 1

        else:
            dd2['score'] = np.nan
            dd2['number'] = np.nan
            dd2['appeals'] = np.nan
            for t in dd.Target.unique():
                try:
                    h = dd_hero[dd_hero.Target == t]['value'].iloc[0]
                except:
                    h = 0
                try:
                    ah = dd_antihero[dd_antihero.Target == t]['value'].iloc[0]
                except:
                    ah = 0
                dd2hero_scores.append(h)
                dd2anti_scores.append(ah)
                i = dd2[dd2.Target == t].index
                dd2.loc[i, 'score'] = ah / (ah + h)
                if h > ah:
                    dd2.loc[i, 'number'] = h
                    dd2.loc[i, 'appeals'] = (ah + h)
                    dd2.loc[i, 'ethos_label'] = 'heroes'
                elif h < ah:
                    dd2.loc[i, 'number'] = ah
                    dd2.loc[i, 'ethos_label'] = 'villains'
                    dd2.loc[i, 'appeals'] = (ah + h)
                else:
                    dd2.loc[i, 'number'] = 0
                    dd2.loc[i, 'ethos_label'] = 'nn'
                    dd2.loc[i, 'appeals'] = (ah + h)

            dd2 = dd2[dd2.score != 0]
            dd2 = dd2[dd2.ethos_label != 'nn']
            #dd2['ethos_label'] = np.where(dd2.score < 0, 'villains', 'neutral')
            #dd2['ethos_label'] = np.where(dd2.score > 0, 'heroes', dd2['ethos_label'])
            dd2 = dd2.sort_values(by = ['ethos_label', 'Target'])
            dd2['corpus'] = ds
            #st.write(dd2)
            up_data_dict_hist[n] = dd2
            #st.write(dd2)
            #st.stop()
            #dd2['score'] = dd2[dd2['ethos_label'] != 'neutral' ]
            #dd2_dist = pd.DataFrame(dd2['ethos_label'].value_counts(normalize=False)).reset_index()
            dd2_dist = pd.DataFrame(dd2['ethos_label'].value_counts(normalize=True).round(3)*100).reset_index()
            dd2_dist.columns = ['heroes', 'percentage']
            dd2_dist['corpus'] = ds
            up_data_dict[n] = dd2_dist
            up_data_dicth[n] = dd2[dd2['ethos_label'] == 'heroes']['Target'].unique()
            up_data_dictah[n] = dd2[dd2['ethos_label'] == 'villains']['Target'].unique()
            n += 1

    df_dist_ethos_all = up_data_dict[0].copy()
    for k in range(int(len(up_data_dict.keys()))-1):
        k_sub = k+1
        df_dist_ethos_all = pd.concat([df_dist_ethos_all, up_data_dict[k_sub]], axis=0, ignore_index=True)

    sns.set(font_scale=1.35, style='whitegrid')
    f_dist_ethos = sns.catplot(kind='bar', data = df_dist_ethos_all, height=5, aspect=1.2,
                    x = 'heroes', y = 'percentage', hue = 'heroes', dodge=False, legend = False,
                    palette = {'villains':'#FF4444', 'heroes':'#298A32'},
                    col = 'corpus')

    if contents_radio_unit == 'number':
        f_dist_ethos.set(ylabel = 'number', xlabel = '')
        for ax in f_dist_ethos.axes.ravel():
            for p in ax.patches:
                ax.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')
    else:
        f_dist_ethos.set(ylim=(0, 110), xlabel = '')
        for ax in f_dist_ethos.axes.ravel():
            for p in ax.patches:
                ax.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')
    add_spacelines(1)


    df_dist_hist_all = up_data_dict_hist[0].copy()
    for k in range(int(len(up_data_dict_hist.keys()))-1):
        k_sub = k+1
        df_dist_hist_all = pd.concat([df_dist_hist_all, up_data_dict_hist[k_sub]], axis=0, ignore_index=True)
    sns.set(font_scale=1.35, style='whitegrid')
    #st.write(df_dist_hist_all)
    #st.stop()
    #f_dist_ethoshist = sns.catplot(kind='count', data = df_dist_hist_all, height=5, aspect=1.3,
    #                x = 'score', hue = 'ethos_label', dodge=False,
    #                palette = {'villains':'#FF4444', 'heroes':'#298A32'},
    #                col = 'corpus')
    #for axes in f_dist_ethoshist.axes.flat:
    #    _ = axes.set_xticklabels(axes.get_xticklabels(), rotation=90)

    df_dist_hist_all = df_dist_hist_all.rename(columns = {'ethos_label':'label'})
    sns.set(font_scale=1, style='whitegrid')
    f_dist_ethoshist = sns.catplot(kind='strip', data = df_dist_hist_all, height=4, aspect=1.25,
                    y = str(contents_radio_unit), hue = 'label', dodge=False, s=25, alpha=0.75,
                    palette = {'villains':'#FF4444', 'heroes':'#298A32'},
                    x = 'corpus')
    if contents_radio_unit == 'score':
        f_dist_ethoshist.set(xlabel = '', title = 'Distribution of villain scores')
    else:
        f_dist_ethoshist.set(xlabel = '', title = 'Number of (un)-favourable appeals to villains & heroes')


    heroes_tab1, heroes_tab2, heroes_tab3, heroes_tab_explore = st.tabs(['Bar-chart', 'Tables', 'Heroes & villains Single Target Analysis', 'Cases'])
    with heroes_tab1:
        add_spacelines(1)
        st.pyplot(f_dist_ethos)
        add_spacelines(1)
        st.pyplot(f_dist_ethoshist)


    with heroes_tab2:
        add_spacelines()
        st.write( "##### Table: summary of  Villains & heroes" )
        cops_names = df_dist_hist_all.corpus.unique()
        cols_columns = st.columns(len(cops_names))
        for n, c in enumerate(cols_columns):
            with c:
                df_dist_hist_all_0 = df_dist_hist_all[df_dist_hist_all.corpus == cops_names[n]]


                #st.write(cops_names[n])
                df_dist_hist_all_0 = df_dist_hist_all_0.sort_values(by = 'score', ascending=True)
                df_dist_hist_all_0 = df_dist_hist_all_0.reset_index(drop=True)
                df_dist_hist_all_0 = df_dist_hist_all_0.set_index('Target').reset_index()
                df_dist_hist_all_0.index += 1
                df_dist_hist_all_02 = pd.DataFrame(df_dist_hist_all_0.label.value_counts()).reset_index()
                if df_dist_hist_all_02.shape[0] == 1:
                    if not 'heroes' in df_dist_hist_all_02.label.unique():
                        df_dist_hist_all_02.loc[len(df_dist_hist_all_02)] = ['heroes', 0]
                    else:
                        df_dist_hist_all_02.loc[len(df_dist_hist_all_02)] = ['villains', 0]

                st.write( df_dist_hist_all_02.style.apply( colorred, axis=1 ) )
                #st.write(df_dist_hist_all_0.style.apply( highlight, axis=1 ))
                st.write(df_dist_hist_all_0)

                df_dist_hist_all_0.Target = df_dist_hist_all_0.Target.apply(lambda x: "_".join(x.split()))

                add_spacelines(1)
                #st.write( "##### Cloud: names of heroes and villains" )
                #f_att0, _ = make_word_cloud(" ".join(df_dist_hist_all_0[df_dist_hist_all_0.label == 'villains'].Target.values), 800, 500, '#1E1E1E', 'Reds')
                #f_sup0, _ = make_word_cloud(" ".join(df_dist_hist_all_0[df_dist_hist_all_0.label == 'heroes'].Target.values), 800, 500, '#1E1E1E', 'Greens')

                #st.pyplot(f_sup0)
                #add_spacelines(2)
                #st.pyplot(f_att0)

        sns.set(font_scale=1.5, style='whitegrid')
        #st.write(df_dist_hist_all)
        cutoff_n = st.slider('Select a cut-off number', 1, 25, 2)
        cc = df_dist_hist_all.corpus.unique()
        #st.write(df_dist_hist_all)
        df_dist_hist_all = df_dist_hist_all[ (df_dist_hist_all.label == 'heroes') | (df_dist_hist_all.number > int(cutoff_n)) ]
        #df_dist_hist_all1 =  df_dist_hist_all[df_dist_hist_all.corpus == cc[0]]
        #df_dist_hist_all1 = df_dist_hist_all1.sort_values(by = [ 'corpus', 'label','number', ], ascending=[True, False, False])

        #df_dist_hist_all5 = df_dist_hist_all[df_dist_hist_all.corpus == cc[-1]]
        #df_dist_hist_all5 = df_dist_hist_all5.sort_values(by = [ 'corpus', 'label', 'number', ], ascending=[True, False, False])

        #df_dist_hist_all = pd.concat([df_dist_hist_all5.iloc[:12], df_dist_hist_all1.iloc[:12]], axis=0, ignore_index=True)
        #df_dist_hist_all.corpus = df_dist_hist_all.corpus.map( {'Covid':"PolarIs1", 'ElectionsSM':'PolarIs5'} )


        #df_dist_hist_all.loc[df_dist_hist_all.score < 0, 'score'] = df_dist_hist_all.loc[df_dist_hist_all.score < 0, 'score'] * -1
        height_n = 11
        df_dist_hist_all = df_dist_hist_all.sort_values(by = [ 'corpus', 'label', 'number', ], ascending=[True, False, False])

        df_dist_hist_all = df_dist_hist_all.melt(['Target', 'label', 'corpus'])

        #df_dist_hist_all = df_dist_hist_all.sort_values(by = [ 'corpus', 'variable', 'label', 'value' ], ascending=[True,True, False, False])


        if df_dist_hist_all.Target.nunique() > 6:
                height_n = int(df_dist_hist_all.Target.nunique() / 3.2 )
                sns.set(font_scale=2, style='whitegrid')
        f_dist_ethoshist_barh = sns.catplot(kind='bar', data = df_dist_hist_all[df_dist_hist_all.variable == contents_radio_unit], height=height_n, aspect=1,
                        x = 'value', y = 'Target', hue = 'label', dodge=False,
                        palette = {'villains':'#FF4444', 'heroes':'#298A32'},
                        col = 'corpus', sharey=False, sharex=True, row = 'variable')
        if contents_radio_unit == 'score':
            f_dist_ethoshist_barh.set(ylabel = '', )
        else:
            f_dist_ethoshist_barh.set(ylabel = '', )

        plt.tight_layout(pad=2)
        sns.move_legend(f_dist_ethoshist_barh, bbox_to_anchor = (0.5, 1.1), ncols=3, loc='upper center', )
        st.pyplot(f_dist_ethoshist_barh)



    with heroes_tab3:
        add_spacelines(2)
        if singl_an:
            st.write("### Single Target Analysis")
            add_spacelines(1)

            target_shared_list = target_shared[0]
            for n in range(int(len(data_list))-1):
                target_shared_list = set(target_shared_list).intersection(target_shared[n+1])
            selected_target = st.selectbox("Choose a target entity you would like to analyse", set(target_shared_list))

            cols_columns = st.columns(len(data_list), gap='large')
            for n, c in enumerate(cols_columns):
                with c:
                    df = data_list[n].copy()
                    ds = df['corpus'].iloc[0]
                    #st.dataframe(df)
                    if not 'neutral' in df['ethos_label'].unique():
                        df['ethos_label'] = df['ethos_label'].map(ethos_mapping)
                    if not 'negative' in df['pathos_label'].unique():
                        df['pathos_label'] = df['pathos_label'].map(valence_mapping)

                    # all df targets
                    df_target_all = pd.DataFrame(df[df.ethos_label != 'neutral']['ethos_label'].value_counts(normalize = True).round(2)*100)
                    df_target_all.columns = ['percentage']
                    df_target_all.reset_index(inplace=True)
                    df_target_all.columns = ['label', 'percentage']
                    df_target_all = df_target_all.sort_values(by = 'label')
                    df_target_all_att = df_target_all[df_target_all.label == 'attack']['percentage'].iloc[0]
                    df_target_all_sup = df_target_all[df_target_all.label == 'support']['percentage'].iloc[0]

                    # chosen target df

                    df_target = pd.DataFrame(df[df.Target == str(selected_target)]['ethos_label'].value_counts(normalize = True).round(2)*100)
                    df_target.columns = ['percentage']
                    df_target.reset_index(inplace=True)
                    df_target.columns = ['label', 'percentage']

                    if len(df_target) == 1:
                      if not ("attack" in df_target.label.unique()):
                          df_target.loc[len(df_target)] = ["attack", 0]
                      elif not ("support" in df_target.label.unique()):
                          df_target.loc[len(df_target)] = ["support", 0]
                    df_target = df_target.sort_values(by = 'label')
                    df_target_att = df_target[df_target.label == 'attack']['percentage'].iloc[0]
                    df_target_sup = df_target[df_target.label == 'support']['percentage'].iloc[0]

                    add_spacelines(1)
                    df_target.columns = ['ethos', 'percentage']
                    df_dist_ethos = df_target.sort_values(by = 'ethos')
                    df_dist_ethos['corpus'] = ds

                    sns.set(font_scale=1.35, style='whitegrid')
                    f_dist_ethos = sns.catplot(kind='bar', data = df_dist_ethos, height=4, aspect=1.4, legend = False,
                                    x = 'ethos', y = 'percentage', hue = 'ethos', dodge=False, col = 'corpus',
                                    palette = {'attack':'#BB0000', 'neutral':'#949494', 'support':'#026F00'})
                    vals_senti = df_dist_ethos['percentage'].values.round(1)
                    plt.title(f"Ethos towards **{str(selected_target)}** in {df.corpus.iloc[0]} \n")
                    plt.xlabel('')
                    plt.ylim(0, 105)
                    plt.yticks(np.arange(0, 105, 20))
                    for index_senti, v in enumerate(vals_senti):
                        plt.text(x=index_senti , y = v+1 , s=f"{v}%" , fontdict=dict(ha='center'))
                    st.pyplot(f_dist_ethos)


                    st.write('**********************************************************************************')
                    #add_spacelines(1)
                    cols = ['sentence', 'ethos_label', 'source', 'Target', 'pathos_label'] #, 'date', 'conversation_id'
                    if len(df[df.Target == str(selected_target)]) == 1:
                        st.write(f"{len(df[df.Target == str(selected_target)])} case of ethotic statements towards **{selected_target}**  in {df['corpus'].iloc[0]} corpus")
                    else:
                        st.write(f"{len(df[df.Target == str(selected_target)])} cases of ethotic statements towards **{selected_target}**  in {df['corpus'].iloc[0]} corpus")
                    if not "neutral" in df['pathos_label'].unique():
                        df['pathos_label'] = df['pathos_label'].map(valence_mapping)
                    st.dataframe(df[df.Target == str(selected_target)][cols].set_index('source').rename(columns={'ethos_label':'ethos'}), width = None)
                    add_spacelines(1)


    with heroes_tab_explore:
        st.write('### Cases')

        if len(data_list) > 1:
            df = pd.concat( data_list, axis=0, ignore_index=True )
        else:
            df = data_list[0]


        df.Target = df.Target.astype('str')
        if "&" in df.corpus.iloc[0]:
            ds = "Covid & ElectionsSM"
            df['corpus'] = ds
        dff_columns = ['map_ID', 'sentence', 'source', 'ethos_name', 'Target', 'pathos_name']# , 'conversation_id','date'

        if not 'neutral' in df['ethos_label'].unique():
            df['ethos_label'] = df['ethos_label'].map(ethos_mapping)


        df = df.dropna(how='all', axis=1)
        df[df.columns] = df[df.columns].astype('str')
        dff = df.copy()
        select_columns = st.multiselect("Choose columns for specifying conditions", dff_columns, dff_columns[-2])
        cols_columns = st.columns(len(select_columns))
        dict_cond = {}
        for n, c in enumerate(cols_columns):
            with c:
                cond_col = st.multiselect(f"Choose condition for *{select_columns[n]}*",
                                       (dff[select_columns[n]].unique()), (dff[select_columns[n]].unique()[-1]))
                dict_cond[select_columns[n]] = cond_col
        dff_selected = dff.copy()
        dff_selected = dff_selected.drop_duplicates(subset = ['sentence'] )
        for i, k in enumerate(dict_cond.keys()):
            dff_selected = dff_selected[ dff_selected[str(k)].isin(dict_cond[k]) ]
        add_spacelines(2)
        st.dataframe(dff_selected[dff_columns].sort_values(by = select_columns).dropna(axis=1, how='all').reset_index(drop=True), width = None)
        st.write(f"No. of cases: {len(dff_selected.dropna(axis=1, how='all'))}.")



def highlight(s):
    if s.score >= 0.5:
        return ['background-color: red'] * len(s)
    elif s.score < 0.5:
        return ['background-color: green'] * len(s)
    else:
        return ['background-color: white'] * len(s)

def colorred(s):
    if s.label == 'villains':
        return ['background-color: red'] * len(s)
    elif s.label == 'heroes':
        return ['background-color: green'] * len(s)
    else:
        return ['background-color: white'] * len(s)




def TargetHeroScores_compare_freq(data_list, singl_an = True):
    #st.write("### (Anti)-hero Frequency")
    add_spacelines(1)
    #contents_radio_heroes = st.radio("Category of the target of ethotic statements", ("direct ethos", "3rd party ethos"))#"both",
    st.write("Choose category of the target of ethotic statements")
    box_direct = st.checkbox("direct ethos", value = False)
    box_3rd = st.checkbox("3rd party ethos", value = True)

    contents_radio_unit = 'number' #st.radio("Unit of analysis", ("score", "number"))

    up_data_dict = {}
    up_data_dicth = {}
    up_data_dictah = {}
    target_shared = {}
    up_data_dict_hist = {}

    n = 0
    for data in data_list:
        df = data.copy()
        ds = df['corpus'].iloc[0]
        if "&" in df.corpus.iloc[0]:
            ds = "Covid & ElectionsSM"
            df['corpus'] = ds
        if not 'attack' in df['ethos_label'].unique():
            df['ethos_label'] = df['ethos_label'].map(ethos_mapping)
        df["Target"] = df["Target"].astype('str')
        df = df[ (df.Target != 'nan') & (df.Target != '') & (df.ethos_label != 'neutral') ]
        df = df.drop_duplicates()
        df["Target"] = df["Target"].str.replace('Government', 'government')
        target_shared[n] = set(df["Target"].unique())

        if box_direct and not box_3rd:
            targets_limit = df['Target'].dropna().unique()
            targets_limit = [t for t in targets_limit if "@" in t]
            df = df[df.Target.isin(targets_limit)]
            if len(targets_limit) < 2:
                st.error(f'No cases of **{contents_radio_heroes}** found in the chosen corpora.')
                st.stop()
        elif not box_direct and box_3rd:
            targets_limit = df['Target'].dropna().unique()
            targets_limit = [t for t in targets_limit if not "@" in t]
            df = df[df.Target.isin(targets_limit)]
            if len(targets_limit) < 2:
                st.error(f'No cases of **{contents_radio_heroes}** found in the chosen corpora.')
                st.stop()

        dd2_size = df.groupby(['Target'], as_index=False).size()
        dd2_size = dd2_size[dd2_size['size'] > 1]
        adj_target = dd2_size['Target'].unique()

        dd = pd.DataFrame(df.groupby(['Target'])['ethos_label'].value_counts(normalize=False))
        dd.columns = ['value']
        dd = dd.reset_index()
        dd = dd[dd.Target.isin(adj_target)]
        dd = dd[dd.ethos_label != 'neutral']
        dd_hero = dd[dd.ethos_label == 'support']
        dd_antihero = dd[dd.ethos_label == 'attack']

        dd2 = pd.DataFrame({'Target': dd.Target.unique()})
        dd2_hist = dd2.copy()
        dd2anti_scores = []
        dd2hero_scores = []

        if contents_radio_unit == 'score':
            dd2['score'] = np.nan
            dd2['number'] = np.nan
            dd2['appeals'] = np.nan
            for t in dd.Target.unique():
                try:
                    h = dd_hero[dd_hero.Target == t]['value'].iloc[0]
                except:
                    h = 0
                try:
                    ah = dd_antihero[dd_antihero.Target == t]['value'].iloc[0]
                except:
                    ah = 0
                dd2hero_scores.append(h)
                dd2anti_scores.append(ah)
                i = dd2[dd2.Target == t].index
                dd2.loc[i, 'score'] = ah / (ah + h)
                if h > ah:
                    dd2.loc[i, 'number'] = h
                    dd2.loc[i, 'appeals'] = (ah + h)
                    dd2.loc[i, 'ethos_label'] = 'heroes'
                elif h < ah:
                    dd2.loc[i, 'number'] = ah
                    dd2.loc[i, 'ethos_label'] = 'villains'
                    dd2.loc[i, 'appeals'] = (ah + h)
                else:
                    dd2.loc[i, 'number'] = 0
                    dd2.loc[i, 'ethos_label'] = 'nn'
                    dd2.loc[i, 'appeals'] = (ah + h)

            dd2 = dd2[dd2.score != 0]
            dd2 = dd2[dd2.ethos_label != 'nn']
            #dd2['ethos_label'] = np.where(dd2.score < 0, 'villains', 'neutral')
            #dd2['ethos_label'] = np.where(dd2.score > 0, 'heroes', dd2['ethos_label'])
            dd2 = dd2.sort_values(by = ['ethos_label', 'Target'])
            #dd2['score'] = dd2['score'] * 100
            #dd2['score'] = dd2['score'].round()
            dd2['corpus'] = ds
            #st.write(dd2)
            up_data_dict_hist[n] = dd2
            #st.write(dd2)
            #st.stop()
            #dd2['score'] = dd2[dd2['ethos_label'] != 'neutral' ]
            dd2_dist = pd.DataFrame(dd2['ethos_label'].value_counts(normalize=True).round(3)*100).reset_index()
            dd2_dist.columns = ['heroes', 'percentage']
            if len(dd2_dist) == 1:
                if not 'heroes' in dd2_dist.heroes.unique():
                    dd2_dist.loc[len(dd2_dist)] = ['heroes', 0]
                else:
                    dd2_dist.loc[len(dd2_dist)] = ['villains', 0]

            dd2_dist['corpus'] = ds
            up_data_dict[n] = dd2_dist
            up_data_dicth[n] = dd2[dd2['ethos_label'] == 'heroes']['Target'].unique()
            up_data_dictah[n] = dd2[dd2['ethos_label'] == 'villains']['Target'].unique()
            n += 1

        else:
            dd2['score'] = np.nan
            dd2['number'] = np.nan
            dd2['appeals'] = np.nan
            for t in dd.Target.unique():
                try:
                    h = dd_hero[dd_hero.Target == t]['value'].iloc[0]
                except:
                    h = 0
                try:
                    ah = dd_antihero[dd_antihero.Target == t]['value'].iloc[0]
                except:
                    ah = 0
                dd2hero_scores.append(h)
                dd2anti_scores.append(ah)
                i = dd2[dd2.Target == t].index
                dd2.loc[i, 'score'] = ah / (ah + h)
                if h > ah:
                    dd2.loc[i, 'number'] = h
                    dd2.loc[i, 'appeals'] = (ah + h)
                    dd2.loc[i, 'ethos_label'] = 'heroes'
                elif h < ah:
                    dd2.loc[i, 'number'] = ah
                    dd2.loc[i, 'ethos_label'] = 'villains'
                    dd2.loc[i, 'appeals'] = (ah + h)
                else:
                    dd2.loc[i, 'number'] = 0
                    dd2.loc[i, 'ethos_label'] = 'nn'
                    dd2.loc[i, 'appeals'] = (ah + h)

            dd2 = dd2[dd2.score != 0]
            dd2 = dd2[dd2.ethos_label != 'nn']
            #dd2['ethos_label'] = np.where(dd2.score < 0, 'villains', 'neutral')
            #dd2['ethos_label'] = np.where(dd2.score > 0, 'heroes', dd2['ethos_label'])
            dd2 = dd2.sort_values(by = ['ethos_label', 'Target'])
            dd2['corpus'] = ds
            #st.write(dd2)
            up_data_dict_hist[n] = dd2
            #st.write(dd2)
            #st.stop()
            #dd2['score'] = dd2[dd2['ethos_label'] != 'neutral' ]
            #dd2_dist = pd.DataFrame(dd2['ethos_label'].value_counts(normalize=False)).reset_index()
            dd2_dist = pd.DataFrame(dd2['ethos_label'].value_counts(normalize=True).round(3)*100).reset_index()
            dd2_dist.columns = ['heroes', 'percentage']
            if len(dd2_dist) == 1:
                if not 'heroes' in dd2_dist.heroes.unique():
                    dd2_dist.loc[len(dd2_dist)] = ['heroes', 0]
                else:
                    dd2_dist.loc[len(dd2_dist)] = ['villains', 0]
            dd2_dist['corpus'] = ds
            up_data_dict[n] = dd2_dist
            up_data_dicth[n] = dd2[dd2['ethos_label'] == 'heroes']['Target'].unique()
            up_data_dictah[n] = dd2[dd2['ethos_label'] == 'villains']['Target'].unique()
            n += 1

    df_dist_ethos_all = up_data_dict[0].copy()
    for k in range(int(len(up_data_dict.keys()))-1):
        k_sub = k+1
        df_dist_ethos_all = pd.concat([df_dist_ethos_all, up_data_dict[k_sub]], axis=0, ignore_index=True)

    sns.set(font_scale=1.5, style='whitegrid')
    if "&" in df_dist_ethos_all.corpus.iloc[0]:
            ds = "Covid & ElectionsSM"
            df_dist_ethos_all['corpus'] = ds

    df = pd.concat( data_list, axis=0, ignore_index=True )
    df.Target = df.Target.astype('str')
    if not 'attack' in df['ethos_label'].unique():
        df['ethos_label'] = df['ethos_label'].map(ethos_mapping)
    df_target_all = pd.DataFrame(df[df.ethos_label != 'neutral']['ethos_label'].value_counts(normalize=True).round(2)*100).reset_index()
    df_target_all.columns = ['ethos', 'percentage']

    f_dist_ethos = sns.catplot(kind='bar', data = df_dist_ethos_all, height=5, aspect=1.2,
                    x = 'heroes', y = 'percentage', hue = 'heroes', dodge=False, legend = False,
                    palette = {'villains':'#FF5656', 'heroes':'#078120'},
                    col = 'corpus')

    f_dist_ethos.set(ylim=(0, 110), xlabel = '')
    f_dist_ethos.map(plt.axhline, y=df_target_all[df_target_all.ethos == 'attack']['percentage'].iloc[0], ls='--', c='red', alpha=0.75, linewidth=1.85, label = 'baseline villains')
    f_dist_ethos.map(plt.axhline, y=df_target_all[df_target_all.ethos == 'support']['percentage'].iloc[0], ls='--', c='green', alpha=0.75, linewidth=1.85, label = 'baseline heroes')
    plt.legend(loc='upper right', fontsize=13, bbox_to_anchor = (1.03, 0.9) )

    for ax in f_dist_ethos.axes.ravel():
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha = 'center', va = 'center', xytext = (0, 7), textcoords = 'offset points')
    add_spacelines(1)


    df_dist_hist_all = up_data_dict_hist[0].copy()
    for k in range(int(len(up_data_dict_hist.keys()))-1):
        k_sub = k+1
        df_dist_hist_all = pd.concat([df_dist_hist_all, up_data_dict_hist[k_sub]], axis=0, ignore_index=True)

    sns.set(font_scale=1.35, style='whitegrid')

    if "&" in df_dist_hist_all.corpus.iloc[0]:
        ds = "Covid & ElectionsSM"
        df_dist_hist_all['corpus'] = ds

    df_dist_hist_all = df_dist_hist_all.rename(columns = {'ethos_label':'label'})
    sns.set(font_scale=1, style='whitegrid')
    f_dist_ethoshist = sns.catplot(kind='strip', data = df_dist_hist_all, height=4, aspect=1.25,
                    y = str(contents_radio_unit), hue = 'label', dodge=False, s=25, alpha=0.75,
                    palette = {'villains':'#FF5656', 'heroes':'#078120'},
                    x = 'corpus')
    if contents_radio_unit == 'score':
        f_dist_ethoshist.set(xlabel = '', title = 'Distribution of villain scores')
    else:
        f_dist_ethoshist.set(xlabel = '', title = 'Number of (un)-favourable appeals to villains & heroes')


    heroes_tab1, heroes_tab2, heroes_tab_explore = st.tabs(['Bar-chart', 'Tables',  'Cases'])

    with heroes_tab2:
        add_spacelines()
        cops_names = df_dist_hist_all.corpus.unique()
        cols_columns = st.columns(len(cops_names))
        for n, c in enumerate(cols_columns):
            with c:
                df_dist_hist_all_0 = df_dist_hist_all[df_dist_hist_all.corpus == cops_names[n]]

                #st.write(cops_names[n])
                df_dist_hist_all_0 = df_dist_hist_all_0.sort_values(by = 'score', ascending=True)
                df_dist_hist_all_0 = df_dist_hist_all_0.reset_index(drop=True)
                df_dist_hist_all_0 = df_dist_hist_all_0.set_index('Target').reset_index()
                df_dist_hist_all_0.index += 1
                df_dist_hist_all_02 = pd.DataFrame(df_dist_hist_all_0.label.value_counts()).reset_index()
                if df_dist_hist_all_02.shape[0] == 1:

                    if not 'heroes' in df_dist_hist_all_02.label.unique():
                        df_dist_hist_all_02.loc[len(df_dist_hist_all_02)] = ['heroes', 0]
                    else:
                        df_dist_hist_all_02.loc[len(df_dist_hist_all_02)] = ['villains', 0]

                st.write( "##### Summary ", df_dist_hist_all_0.corpus.iloc[0] )
                st.write( df_dist_hist_all_02.style.apply( colorred, axis=1 ) )

                add_spacelines(2)
                st.write( "##### Detailed", df_dist_hist_all_0.corpus.iloc[0]  )
                st.write(df_dist_hist_all_0.style.apply( highlight, axis=1 ))

                df_dist_hist_all_0.Target = df_dist_hist_all_0.Target.apply(lambda x: "_".join(x.split()))

                #st.write( "##### Cloud: names of heroes and villains" )
                #f_att0, _ = make_word_cloud(" ".join(df_dist_hist_all_0[df_dist_hist_all_0.label == 'villains'].Target.values), 800, 500, '#1E1E1E', 'Reds')
                #f_sup0, _ = make_word_cloud(" ".join(df_dist_hist_all_0[df_dist_hist_all_0.label == 'heroes'].Target.values), 800, 500, '#1E1E1E', 'Greens')

                #st.pyplot(f_sup0)
                #add_spacelines(2)
                #st.pyplot(f_att0)


    with heroes_tab1:
        add_spacelines(1)
        st.write( "##### Summary " )
        st.pyplot(f_dist_ethos)

        add_spacelines(2)
        st.write( "##### Detailed " )

        #st.write(df_dist_hist_all)
        #st.write(df_dist_hist_all)

        #cutoff_n = st.slider('Select a cut-off number', 1, 25, 2)
        cutoff_neg = st.slider('Select a cut-off number for villains', 1, 25, 4)
        cutoff_pos = st.slider('Select a cut-off number for heroes', 1, 25, 2)
        cc = df_dist_hist_all.corpus.unique()
        #st.write(df_dist_hist_all)

        df_dist_hist_all = df_dist_hist_all[ ( (df_dist_hist_all.label == 'heroes') & (df_dist_hist_all.number > int(cutoff_pos)) ) |\
                                            ( (df_dist_hist_all.label == 'villains') & (df_dist_hist_all.number > int(cutoff_neg)) ) ]


        height_n = 7
        sns.set(font_scale=1.5, style='whitegrid')
        if df_dist_hist_all.Target.nunique() > 20:
                height_n = int(df_dist_hist_all.Target.nunique() / 3.2 )
                sns.set(font_scale=1.7, style='whitegrid')
        #st.write(df_dist_hist_all)
        df_dist_hist_all = df_dist_hist_all.sort_values(by = [ 'corpus', 'label', 'number', ], ascending=[True, False, False])
        df_dist_ethos_all2_base = df_dist_hist_all[contents_radio_unit].mean().round(2)

        df_dist_hist_all = df_dist_hist_all.melt(['Target', 'label', 'corpus'])

        #df_dist_hist_all = df_dist_hist_all.sort_values(by = [ 'corpus', 'variable', 'label', 'value' ], ascending=[True,True, False, False])

        #df_dist_hist_all['corpus'] = ds
        f_dist_ethoshist_barh = sns.catplot(kind='bar', data = df_dist_hist_all[df_dist_hist_all.variable == contents_radio_unit],
                        height=height_n, aspect=1.3,
                        x = 'value', y = 'Target', hue = 'label', dodge=False,
                        palette = {'villains':'#FF5656', 'heroes':'#078120'},
                        col = 'corpus', sharey=False, sharex=True, row = 'variable')
        if contents_radio_unit == 'score':
            f_dist_ethoshist_barh.set(ylabel = '', )
        else:
            f_dist_ethoshist_barh.set(ylabel = '', )

        #f_dist_ethoshist_barh.map(plt.axvline, x=df_dist_ethos_all2_base, ls='--', c='black', alpha=0.75, linewidth=2, label = 'baseline' )
        #plt.tight_layout(pad=0.5)
        #sns.move_legend(f_dist_ethoshist_barh, bbox_to_anchor = (0.54, 1.1), ncols=3, loc='upper center',)
        f_dist_ethoshist_barh._legend.remove()
        if "&" in df_dist_hist_all.corpus.iloc[0]:
            xv = 0.4
        else:
            xv = -0.10
        plt.legend(bbox_to_anchor = (xv, 1.3), ncols=3, loc='upper center',)

        st.pyplot(f_dist_ethoshist_barh)


    with heroes_tab_explore:
        st.write('### Cases')

        if len(data_list) > 1:
            df = pd.concat( data_list, axis=0, ignore_index=True )
        else:
            df = data_list[0]

        df["Target"] = df["Target"].astype('str')
        df = df[ (df.Target != 'nan') & (df.Target != '') & (df.ethos_label != 'neutral') ]
        df = df.drop_duplicates()
        df["Target"] = df["Target"].str.replace('Government', 'government')
        if "&" in df.corpus.iloc[0]:
            ds = "Covid & ElectionsSM"
            df['corpus'] = ds
        dff_columns = [ 'sentence', 'source', 'ethos_name', 'Target', 'pathos_name']# , 'conversation_id','date'
        df['ethos_name'] = df['ethos_label']
        df['pathos_name'] = df['pathos_label']
        if not 'neutral' in df['ethos_label'].unique():
            df['ethos_name'] = df['ethos_label'].map(ethos_mapping)
        if not 'negative' in df['pathos_label'].unique():
            df['pathos_name'] = df['pathos_label'].map(valence_mapping)

        df = df.dropna(how='all', axis=1)
        df[df.columns] = df[df.columns].astype('str')
        dff = df.copy()
        select_columns = st.multiselect("Choose columns for specifying conditions", dff_columns, dff_columns[-2])
        cols_columns = st.columns(len(select_columns))
        dict_cond = {}
        for n, c in enumerate(cols_columns):
            with c:
                cond_col = st.multiselect(f"Choose condition for *{select_columns[n]}*",
                                       (dff[select_columns[n]].unique()), (dff[select_columns[n]].unique()[-1]))
                dict_cond[select_columns[n]] = cond_col
        dff_selected = dff.copy()
        dff_selected = dff_selected.drop_duplicates(subset = ['sentence'] )
        for i, k in enumerate(dict_cond.keys()):
            dff_selected = dff_selected[ dff_selected[str(k)].isin(dict_cond[k]) ]
        add_spacelines(2)
        st.dataframe(dff_selected[dff_columns].sort_values(by = select_columns).dropna(axis=1, how='all').reset_index(drop=True), width = None)
        st.write(f"No. of cases: {len(dff_selected.dropna(axis=1, how='all'))}.")



def TargetHeroScores_compare_scor(data_list, singl_an = True):
    #st.write("### (Anti)-hero Score")
    add_spacelines(1)
    #contents_radio_heroes = st.radio("Category of the target of ethotic statements", ( "direct ethos", "3rd party ethos"))
    st.write("Choose category of the target of ethotic statements")
    box_direct = st.checkbox("direct ethos", value = False)
    box_3rd = st.checkbox("3rd party ethos", value = True)
    contents_radio_unit = 'score' #st.radio("Unit of analysis", ("score", "number"))

    up_data_dict = {}
    up_data_dicth = {}
    up_data_dictah = {}
    target_shared = {}
    up_data_dict_hist = {}

    n = 0
    for data in data_list:
        df = data.copy()
        ds = df['corpus'].iloc[0]
        if not 'attack' in df['ethos_label'].unique():
            df['ethos_label'] = df['ethos_label'].map(ethos_mapping)
        df["Target"] = df["Target"].astype('str')
        df = df[ (df.Target != 'nan') & (df.Target != '') & (df.ethos_label != 'neutral') ]
        df["Target"] = df["Target"].str.replace('Government', 'government')
        df = df.drop_duplicates()
        target_shared[n] = set(df["Target"].unique())

        if box_direct and not box_3rd:
            targets_limit = df['Target'].dropna().unique()
            targets_limit = [t for t in targets_limit if "@" in t]
            df = df[df.Target.isin(targets_limit)]
            if len(targets_limit) < 2:
                st.error(f'No cases of **{contents_radio_heroes}** found in the chosen corpora.')
                st.stop()
        elif not box_direct and box_3rd:
            targets_limit = df['Target'].dropna().unique()
            targets_limit = [t for t in targets_limit if not "@" in t]
            df = df[df.Target.isin(targets_limit)]
            if len(targets_limit) < 2:
                st.error(f'No cases of **{contents_radio_heroes}** found in the chosen corpora.')
                st.stop()

        dd2_size = df.groupby(['Target'], as_index=False).size()
        dd2_size = dd2_size[dd2_size['size'] > 1]
        adj_target = dd2_size['Target'].unique()

        dd = pd.DataFrame(df.groupby(['Target'])['ethos_label'].value_counts(normalize=False))
        dd.columns = ['value']
        dd = dd.reset_index()
        dd = dd[dd.Target.isin(adj_target)]
        dd = dd[dd.ethos_label != 'neutral']
        dd_hero = dd[dd.ethos_label == 'support']
        dd_antihero = dd[dd.ethos_label == 'attack']

        dd2 = pd.DataFrame({'Target': dd.Target.unique()})
        dd2_hist = dd2.copy()
        dd2anti_scores = []
        dd2hero_scores = []

        if contents_radio_unit == 'score':
            dd2['score'] = np.nan
            dd2['number'] = np.nan
            dd2['appeals'] = np.nan
            for t in dd.Target.unique():
                try:
                    h = dd_hero[dd_hero.Target == t]['value'].iloc[0]
                except:
                    h = 0
                try:
                    ah = dd_antihero[dd_antihero.Target == t]['value'].iloc[0]
                except:
                    ah = 0
                dd2hero_scores.append(h)
                dd2anti_scores.append(ah)
                i = dd2[dd2.Target == t].index
                dd2.loc[i, 'score'] = ah / (ah + h)
                if h > ah:
                    dd2.loc[i, 'number'] = h
                    dd2.loc[i, 'appeals'] = (ah + h)
                    dd2.loc[i, 'ethos_label'] = 'heroes'
                elif h < ah:
                    dd2.loc[i, 'number'] = ah
                    dd2.loc[i, 'ethos_label'] = 'villains'
                    dd2.loc[i, 'appeals'] = (ah + h)
                else:
                    dd2.loc[i, 'number'] = 0
                    dd2.loc[i, 'ethos_label'] = 'nn'
                    dd2.loc[i, 'appeals'] = (ah + h)

            dd2 = dd2[dd2.score != 0]
            dd2 = dd2[dd2.ethos_label != 'nn']
            #dd2['ethos_label'] = np.where(dd2.score < 0, 'villains', 'neutral')
            #dd2['ethos_label'] = np.where(dd2.score > 0, 'heroes', dd2['ethos_label'])
            dd2 = dd2.sort_values(by = ['ethos_label', 'Target'])
            #dd2['score'] = dd2['score'] * 100
            #dd2['score'] = dd2['score'].round()
            dd2['corpus'] = ds
            #st.write(dd2)
            up_data_dict_hist[n] = dd2
            #st.write(dd2)
            #st.stop()
            #dd2['score'] = dd2[dd2['ethos_label'] != 'neutral' ]
            dd2_dist = pd.DataFrame(dd2['ethos_label'].value_counts(normalize=True).round(3)*100).reset_index()
            dd2_dist.columns = ['heroes', 'percentage']
            if len(dd2_dist) == 1:
                if not 'heroes' in dd2_dist.heroes.unique():
                    dd2_dist.loc[len(dd2_dist)] = ['heroes', 0]
                else:
                    dd2_dist.loc[len(dd2_dist)] = ['villains', 0]
            dd2_dist['corpus'] = ds
            up_data_dict[n] = dd2_dist
            up_data_dicth[n] = dd2[dd2['ethos_label'] == 'heroes']['Target'].unique()
            up_data_dictah[n] = dd2[dd2['ethos_label'] == 'villains']['Target'].unique()
            n += 1

        else:
            dd2['score'] = np.nan
            dd2['number'] = np.nan
            dd2['appeals'] = np.nan
            for t in dd.Target.unique():
                try:
                    h = dd_hero[dd_hero.Target == t]['value'].iloc[0]
                except:
                    h = 0
                try:
                    ah = dd_antihero[dd_antihero.Target == t]['value'].iloc[0]
                except:
                    ah = 0
                dd2hero_scores.append(h)
                dd2anti_scores.append(ah)
                i = dd2[dd2.Target == t].index
                dd2.loc[i, 'score'] = ah / (ah + h)
                if h > ah:
                    dd2.loc[i, 'number'] = h
                    dd2.loc[i, 'appeals'] = (ah + h)
                    dd2.loc[i, 'ethos_label'] = 'heroes'
                elif h < ah:
                    dd2.loc[i, 'number'] = ah
                    dd2.loc[i, 'ethos_label'] = 'villains'
                    dd2.loc[i, 'appeals'] = (ah + h)
                else:
                    dd2.loc[i, 'number'] = 0
                    dd2.loc[i, 'ethos_label'] = 'nn'
                    dd2.loc[i, 'appeals'] = (ah + h)

            dd2 = dd2[dd2.score != 0]
            dd2 = dd2[dd2.ethos_label != 'nn']
            #dd2['ethos_label'] = np.where(dd2.score < 0, 'villains', 'neutral')
            #dd2['ethos_label'] = np.where(dd2.score > 0, 'heroes', dd2['ethos_label'])
            dd2 = dd2.sort_values(by = ['ethos_label', 'Target'])
            dd2['corpus'] = ds
            #st.write(dd2)
            up_data_dict_hist[n] = dd2
            #st.write(dd2)
            #st.stop()
            #dd2['score'] = dd2[dd2['ethos_label'] != 'neutral' ]
            #dd2_dist = pd.DataFrame(dd2['ethos_label'].value_counts(normalize=False)).reset_index()
            dd2_dist = pd.DataFrame(dd2['ethos_label'].value_counts(normalize=True).round(3)*100).reset_index()
            dd2_dist.columns = ['heroes', 'percentage']
            if len(dd2_dist) == 1:
                if not 'heroes' in dd2_dist.heroes.unique():
                    dd2_dist.loc[len(dd2_dist)] = ['heroes', 0]
                else:
                    dd2_dist.loc[len(dd2_dist)] = ['villains', 0]
            dd2_dist['corpus'] = ds
            up_data_dict[n] = dd2_dist
            up_data_dicth[n] = dd2[dd2['ethos_label'] == 'heroes']['Target'].unique()
            up_data_dictah[n] = dd2[dd2['ethos_label'] == 'villains']['Target'].unique()
            n += 1

    df_dist_ethos_all = up_data_dict[0].copy()
    for k in range(int(len(up_data_dict.keys()))-1):
        k_sub = k+1
        df_dist_ethos_all = pd.concat([df_dist_ethos_all, up_data_dict[k_sub]], axis=0, ignore_index=True)

    if "&" in df_dist_ethos_all.corpus.iloc[0]:
            ds = "Covid & ElectionsSM"
            df_dist_ethos_all['corpus'] = ds

    sns.set(font_scale=1.5, style='whitegrid')
    df = pd.concat( data_list, axis=0, ignore_index=True )
    df.Target = df.Target.astype('str')
    if not 'attack' in df['ethos_label'].unique():
        df['ethos_label'] = df['ethos_label'].map(ethos_mapping)
    df_target_all = pd.DataFrame(df[df.ethos_label != 'neutral']['ethos_label'].value_counts(normalize=True).round(2)*100).reset_index()
    df_target_all.columns = ['ethos', 'percentage']

    f_dist_ethos = sns.catplot(kind='bar', data = df_dist_ethos_all, height=5, aspect=1.2,
                    x = 'heroes', y = 'percentage', hue = 'heroes', dodge=False, legend = False,
                    palette = {'villains':'#FF5656', 'heroes':'#078120'},
                    col = 'corpus')

    f_dist_ethos.set(ylim=(0, 110), xlabel = '')
    f_dist_ethos.map(plt.axhline, y=df_target_all[df_target_all.ethos == 'attack']['percentage'].iloc[0], ls='--', c='red', alpha=0.75, linewidth=1.85, label = 'baseline villains')
    f_dist_ethos.map(plt.axhline, y=df_target_all[df_target_all.ethos == 'support']['percentage'].iloc[0], ls='--', c='green', alpha=0.75, linewidth=1.85, label = 'baseline heroes')
    plt.legend(loc='upper right', fontsize=13, bbox_to_anchor = (1.03, 0.9))

    for ax in f_dist_ethos.axes.ravel():
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha = 'center', va = 'center', xytext = (0, 7), textcoords = 'offset points')
    add_spacelines(1)


    df_dist_hist_all = up_data_dict_hist[0].copy()
    for k in range(int(len(up_data_dict_hist.keys()))-1):
        k_sub = k+1
        df_dist_hist_all = pd.concat([df_dist_hist_all, up_data_dict_hist[k_sub]], axis=0, ignore_index=True)

    sns.set(font_scale=1.35, style='whitegrid')

    df_dist_hist_all = df_dist_hist_all.rename(columns = {'ethos_label':'label'})
    if "&" in df_dist_hist_all.corpus.iloc[0]:
            ds = "Covid & ElectionsSM"
            df_dist_hist_all['corpus'] = ds
    sns.set(font_scale=1, style='whitegrid')
    df_dist_hist_all_base = df_dist_hist_all.groupby('label', as_index=False).score.mean().round(2)

    f_dist_ethoshist = sns.catplot(kind='strip', data = df_dist_hist_all, height=4, aspect=1.25,
                    y = str(contents_radio_unit), hue = 'label', dodge=False, s=55, alpha=0.85, edgecolor = 'black', linewidth = 0.4,
                    palette = {'villains':'#FF5656', 'heroes':'#078120'},
                    x = 'corpus', )
    sns.move_legend(f_dist_ethoshist, frameon = True, loc = 'upper right', bbox_to_anchor = (0.98, 0.72))
    #f_dist_ethoshist.map(plt.axhline, y=df_dist_hist_all_base[df_dist_hist_all_base.label=='heroes'].score.iloc[0], ls='--', c='green', alpha=0.75, linewidth=1.85, label = 'baseline heroes')
    #f_dist_ethoshist.map(plt.axhline, y=df_dist_hist_all_base[df_dist_hist_all_base.label=='villains'].score.iloc[0], ls='--', c='red', alpha=0.75, linewidth=1.85, label = 'baseline villains')

    if contents_radio_unit == 'score':
        f_dist_ethoshist.set(xlabel = '', title = 'Distribution of villain scores')
    else:
        f_dist_ethoshist.set(xlabel = '', title = 'Number of (un)-favourable appeals to villains & heroes')


    heroes_tab1, heroes_tab2, heroes_tab_explore = st.tabs(['Bar-chart', 'Tables',  'Cases'])

    with heroes_tab2:
        add_spacelines()

        cops_names = df_dist_hist_all.corpus.unique()
        cols_columns = st.columns(len(cops_names))
        for n, c in enumerate(cols_columns):
            with c:
                df_dist_hist_all_0 = df_dist_hist_all[df_dist_hist_all.corpus == cops_names[n]]

                #st.write(cops_names[n])
                df_dist_hist_all_0 = df_dist_hist_all_0.sort_values(by = 'score', ascending=True)
                df_dist_hist_all_0 = df_dist_hist_all_0.reset_index(drop=True)
                df_dist_hist_all_0 = df_dist_hist_all_0.set_index('Target').reset_index()
                df_dist_hist_all_0.index += 1
                df_dist_hist_all_02 = pd.DataFrame(df_dist_hist_all_0.label.value_counts()).reset_index()
                if df_dist_hist_all_02.shape[0] == 1:
                    if not 'heroes' in df_dist_hist_all_02.label.unique():
                        df_dist_hist_all_02.loc[len(df_dist_hist_all_02)] = ['heroes', 0]
                    else:
                        df_dist_hist_all_02.loc[len(df_dist_hist_all_02)] = ['villains', 0]


                st.write( "##### Summary ", df_dist_hist_all_0.corpus.iloc[0] )
                st.write( df_dist_hist_all_02.style.apply( colorred, axis=1 ) )

                add_spacelines(2)
                st.write( "##### Detailed", df_dist_hist_all_0.corpus.iloc[0]  )
                st.write(df_dist_hist_all_0.style.apply( highlight, axis=1 ))

                df_dist_hist_all_0.Target = df_dist_hist_all_0.Target.apply(lambda x: "_".join(x.split()))

                #st.write( "##### Cloud: names of heroes and villains" )
                #f_att0, _ = make_word_cloud(" ".join(df_dist_hist_all_0[df_dist_hist_all_0.label == 'villains'].Target.values), 800, 500, '#1E1E1E', 'Reds')
                #f_sup0, _ = make_word_cloud(" ".join(df_dist_hist_all_0[df_dist_hist_all_0.label == 'heroes'].Target.values), 800, 500, '#1E1E1E', 'Greens')

                #st.pyplot(f_sup0)
                #add_spacelines(2)
                #st.pyplot(f_att0)


    with heroes_tab1:
        add_spacelines(1)
        st.write( "##### Summary" )
        st.pyplot(f_dist_ethoshist)

        add_spacelines(2)
        st.write( "##### Detailed" )
        add_spacelines(1)
        #st.write(df_dist_hist_all)
        cutoff_neg = st.slider('Select a cut-off number for villains', 1, 25, 4)
        cutoff_pos = st.slider('Select a cut-off number for heroes', 1, 25, 2)
        cc = df_dist_hist_all.corpus.unique()
        #st.write(df_dist_hist_all)
        df_dist_hist_all = df_dist_hist_all[ ( (df_dist_hist_all.label == 'heroes') & (df_dist_hist_all.number > int(cutoff_pos)) ) |\
                                            ( (df_dist_hist_all.label == 'villains') & (df_dist_hist_all.number > int(cutoff_neg)) ) ]

        #st.write(df_dist_hist_all)

        df_dist_hist_all = df_dist_hist_all.sort_values(by = [ 'corpus', 'label', 'number', ], ascending=[True, False, False])
        df_dist_ethos_all2_base = df_dist_hist_all[contents_radio_unit].mean().round(2)

        df_dist_hist_all = df_dist_hist_all.melt(['Target', 'label', 'corpus'])
        if "&" in df_dist_hist_all.corpus.iloc[0]:
            ds = "Covid & ElectionsSM"
            df_dist_hist_all['corpus'] = ds

        #df_dist_hist_all = df_dist_hist_all.sort_values(by = [ 'corpus', 'variable', 'label', 'value' ], ascending=[True,True, False, False])

        sns.set(font_scale=1.35, style='whitegrid')
        height_n = int(df_dist_hist_all.Target.nunique() / 3.2 )
        if df_dist_hist_all.Target.nunique() < 16:
                height_n = 9
                sns.set(font_scale=1.2, style='whitegrid')

        f_dist_ethoshist_barh = sns.catplot(kind='bar',
                        data = df_dist_hist_all[df_dist_hist_all.variable == contents_radio_unit], height=height_n, aspect=1.3,
                        x = 'value', y = 'Target', hue = 'label', dodge=False,# legend=False,
                        palette = {'villains':'#FF5656', 'heroes':'#078120'},
                        col = 'corpus', sharey=False, sharex=True, row = 'variable')

        f_dist_ethoshist_barh.set(ylabel = '', )
        #f_dist_ethoshist_barh.map(plt.axvline, x=df_dist_ethos_all2_base, ls='--', c='black', alpha=0.75, linewidth=2.5, label = 'baseline' )
        hatches = ["//",  " "]
        # Loop over the bars
        #for hatch, patch in zip(hatches, f_dist_ethoshist_barh.artists):
        #    patch.set_hatch(hatch)

        #plt.tight_layout(pad=0.5)
        #sns.move_legend(f_dist_ethoshist_barh, bbox_to_anchor = (0.54, 1.1), ncols=3, loc='upper center', )
        f_dist_ethoshist_barh._legend.remove()
        if "&" in df_dist_hist_all.corpus.iloc[0]:
            xv = 0.4
        else:
            xv = -0.10
        plt.legend(bbox_to_anchor = (xv, 1.3), ncols=3, loc='upper center',)
        st.pyplot(f_dist_ethoshist_barh)


    with heroes_tab_explore:
        st.write('### Cases')

        if len(data_list) > 1:
            df = pd.concat( data_list, axis=0, ignore_index=True )
        else:
            df = data_list[0]

        df.Target = df.Target.astype('str')
        df = df.drop_duplicates()
        df = df[ (df.Target != 'nan') & (df.Target != '') & (df.ethos_label != 'neutral') ]
        df = df.drop_duplicates()
        df["Target"] = df["Target"].str.replace('Government', 'government')
        if "&" in df.corpus.iloc[0]:
            ds = "Covid & ElectionsSM"
            df['corpus'] = ds
        dff_columns = [ 'sentence', 'source', 'ethos_name', 'Target', 'pathos_name']# , 'conversation_id','date'

        df['ethos_name'] = df['ethos_label']
        df['pathos_name'] = df['pathos_label']

        if not 'neutral' in df['ethos_label'].unique():
            df['ethos_name'] = df['ethos_label'].map(ethos_mapping)
        if not 'negative' in df['pathos_label'].unique():
            df['pathos_name'] = df['pathos_label'].map(valence_mapping)


        dff = df.copy()
        select_columns = st.multiselect("Choose columns for specifying conditions", dff_columns, dff_columns[-2])
        cols_columns = st.columns(len(select_columns))
        dict_cond = {}
        for n, c in enumerate(cols_columns):
            with c:
                cond_col = st.multiselect(f"Choose condition for *{select_columns[n]}*",
                                       (dff[select_columns[n]].unique()), (dff[select_columns[n]].unique()[-1]))
                dict_cond[select_columns[n]] = cond_col
        dff_selected = dff.copy()
        dff_selected = dff_selected.drop_duplicates(subset = ['sentence'] )
        for i, k in enumerate(dict_cond.keys()):
            dff_selected = dff_selected[ dff_selected[str(k)].isin(dict_cond[k]) ]
        add_spacelines(2)
        st.dataframe(dff_selected[dff_columns].sort_values(by = select_columns).dropna(axis=1, how='all').reset_index(drop=True), width = None)
        st.write(f"No. of cases: {len(dff_selected.dropna(axis=1, how='all'))}.")






def TargetHeroScores_compare_scor2(data_list, singl_an = True):
    #st.write("### (Anti)-hero Score")
    add_spacelines(1)
    #contents_radio_heroes = st.radio("Category of the target of ethotic statements", ( "direct ethos", "3rd party ethos"))
    st.write("Choose category of the target of ethotic statements")
    box_direct = st.checkbox("direct ethos", value = False)
    box_3rd = st.checkbox("3rd party ethos", value = True)
    contents_radio_unit = 'score' #st.radio("Unit of analysis", ("score", "number"))

    up_data_dict = {}
    up_data_dicth = {}
    up_data_dictah = {}
    target_shared = {}
    up_data_dict_hist = {}

    n = 0
    for data in data_list:
        df = data.copy()
        ds = df['corpus'].iloc[0]
        df = df.drop_duplicates()
        if not 'attack' in df['ethos_label'].unique():
            df['ethos_label'] = df['ethos_label'].map(ethos_mapping)
        df["Target"] = df["Target"].astype('str')
        df = df[ (df.Target != 'nan') & (df.Target != '') & (df.ethos_label != 'neutral') ]
        df["Target"] = df["Target"].str.replace('Government', 'government')
        target_shared[n] = set(df["Target"].unique())

        if box_direct and not box_3rd:
            targets_limit = df['Target'].dropna().unique()
            targets_limit = [t for t in targets_limit if "@" in t]
            df = df[df.Target.isin(targets_limit)]
            if len(targets_limit) < 2:
                st.error(f'No cases of **{contents_radio_heroes}** found in the chosen corpora.')
                st.stop()
        elif not box_direct and box_3rd:
            targets_limit = df['Target'].dropna().unique()
            targets_limit = [t for t in targets_limit if not "@" in t]
            df = df[df.Target.isin(targets_limit)]
            if len(targets_limit) < 2:
                st.error(f'No cases of **{contents_radio_heroes}** found in the chosen corpora.')
                st.stop()

        dd2_size = df.groupby(['Target'], as_index=False).size()
        dd2_size = dd2_size[dd2_size['size'] > 1]
        adj_target = dd2_size['Target'].unique()

        dd = pd.DataFrame(df.groupby(['Target'])['ethos_label'].value_counts(normalize=False))
        dd.columns = ['value']
        dd = dd.reset_index()
        dd = dd[dd.Target.isin(adj_target)]
        dd = dd[dd.ethos_label != 'neutral']
        dd_hero = dd[dd.ethos_label == 'support']
        dd_antihero = dd[dd.ethos_label == 'attack']

        dd2 = pd.DataFrame({'Target': dd.Target.unique()})
        dd2_hist = dd2.copy()
        dd2anti_scores = []
        dd2hero_scores = []

        dd2['score'] = np.nan
        dd2['number'] = np.nan
        dd2['appeals'] = np.nan
        for t in dd.Target.unique():
            try:
                h = dd_hero[dd_hero.Target == t]['value'].iloc[0]
            except:
                h = 0
            try:
                ah = dd_antihero[dd_antihero.Target == t]['value'].iloc[0]
            except:
                ah = 0
            dd2hero_scores.append(h)
            dd2anti_scores.append(ah)
            i = dd2[dd2.Target == t].index
            dd2.loc[i, 'score'] = (ah - h) / (ah + h)
            if h > ah:
                dd2.loc[i, 'number'] = h
                dd2.loc[i, 'appeals'] = (ah + h)
                dd2.loc[i, 'ethos_label'] = 'heroes'
            elif h < ah:
                dd2.loc[i, 'number'] = ah
                dd2.loc[i, 'ethos_label'] = 'villains'
                dd2.loc[i, 'appeals'] = (ah + h)
            else:
                dd2.loc[i, 'number'] = 0
                dd2.loc[i, 'ethos_label'] = 'nn'
                dd2.loc[i, 'appeals'] = (ah + h)

        dd2 = dd2[dd2.score != 0]
        dd2.score = dd2.score * -1
        dd2 = dd2[dd2.ethos_label != 'nn']
        #dd2['ethos_label'] = np.where(dd2.score < 0, 'villains', 'neutral')
        #dd2['ethos_label'] = np.where(dd2.score > 0, 'heroes', dd2['ethos_label'])
        dd2 = dd2.sort_values(by = ['ethos_label', 'Target'])
        #dd2['score'] = dd2['score'] * 100
        #dd2['score'] = dd2['score'].round()
        dd2['corpus'] = ds
        #st.write(dd2)
        up_data_dict_hist[n] = dd2
        #st.write(dd2)
        #st.stop()
        #dd2['score'] = dd2[dd2['ethos_label'] != 'neutral' ]
        dd2_dist = pd.DataFrame(dd2['ethos_label'].value_counts(normalize=True).round(3)*100).reset_index()
        dd2_dist.columns = ['heroes', 'percentage']
        if len(dd2_dist) == 1:
            if not 'heroes' in dd2_dist.heroes.unique():
                dd2_dist.loc[len(dd2_dist)] = ['heroes', 0]
            else:
                dd2_dist.loc[len(dd2_dist)] = ['villains', 0]
        dd2_dist['corpus'] = ds
        up_data_dict[n] = dd2_dist
        up_data_dicth[n] = dd2[dd2['ethos_label'] == 'heroes']['Target'].unique()
        up_data_dictah[n] = dd2[dd2['ethos_label'] == 'villains']['Target'].unique()
        n += 1

    df_dist_ethos_all = up_data_dict[0].copy()
    for k in range(int(len(up_data_dict.keys()))-1):
        k_sub = k+1
        df_dist_ethos_all = pd.concat([df_dist_ethos_all, up_data_dict[k_sub]], axis=0, ignore_index=True)

    if "&" in df_dist_ethos_all.corpus.iloc[0]:
            ds = "Covid & ElectionsSM"
            df_dist_ethos_all['corpus'] = ds

    sns.set(font_scale=1.5, style='whitegrid')
    df = pd.concat( data_list, axis=0, ignore_index=True )
    df.Target = df.Target.astype('str')
    if not 'attack' in df['ethos_label'].unique():
        df['ethos_label'] = df['ethos_label'].map(ethos_mapping)
    df_target_all = pd.DataFrame(df[df.ethos_label != 'neutral']['ethos_label'].value_counts(normalize=True).round(2)*100).reset_index()
    df_target_all.columns = ['ethos', 'percentage']

    f_dist_ethos = sns.catplot(kind='bar', data = df_dist_ethos_all, height=5, aspect=1.2,
                    x = 'heroes', y = 'percentage', hue = 'heroes', dodge=False, legend = False,
                    palette = {'villains':'#FF5656', 'heroes':'#078120'},
                    col = 'corpus')

    f_dist_ethos.set(ylim=(0, 110), xlabel = '')
    f_dist_ethos.map(plt.axhline, y=df_target_all[df_target_all.ethos == 'attack']['percentage'].iloc[0], ls='--', c='red', alpha=0.75, linewidth=1.85, label = 'baseline villains')
    f_dist_ethos.map(plt.axhline, y=df_target_all[df_target_all.ethos == 'support']['percentage'].iloc[0], ls='--', c='green', alpha=0.75, linewidth=1.85, label = 'baseline heroes')
    plt.legend(loc='upper right', fontsize=13, bbox_to_anchor = (1.03, 0.9))

    for ax in f_dist_ethos.axes.ravel():
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha = 'center', va = 'center', xytext = (0, 7), textcoords = 'offset points')
    add_spacelines(1)




    df_dist_hist_all = up_data_dict_hist[0].copy()
    for k in range(int(len(up_data_dict_hist.keys()))-1):
        k_sub = k+1
        df_dist_hist_all = pd.concat([df_dist_hist_all, up_data_dict_hist[k_sub]], axis=0, ignore_index=True)

    sns.set(font_scale=1.35, style='whitegrid')

    df_dist_hist_all = df_dist_hist_all.rename(columns = {'ethos_label':'label'})
    if "&" in df_dist_hist_all.corpus.iloc[0]:
            ds = "Covid & ElectionsSM"
            df_dist_hist_all['corpus'] = ds
    sns.set(font_scale=1, style='whitegrid')
    df_dist_hist_all_base = df_dist_hist_all.groupby('label', as_index=False).score.mean().round(2)

    f_dist_ethoshist = sns.catplot(kind='strip', data = df_dist_hist_all, height=4, aspect=1.25,
                    y = str(contents_radio_unit), hue = 'label', dodge=False, s=55, alpha=0.85, edgecolor = 'black', linewidth = 0.4,
                    palette = {'villains':'#FF5656', 'heroes':'#078120'},
                    x = 'corpus', )
    #f_dist_ethoshist.map(plt.axhline, y=df_dist_hist_all_base[df_dist_hist_all_base.label=='heroes'].score.iloc[0], ls='--', c='green', alpha=0.75, linewidth=1.85, label = 'baseline heroes')
    #f_dist_ethoshist.map(plt.axhline, y=df_dist_hist_all_base[df_dist_hist_all_base.label=='villains'].score.iloc[0], ls='--', c='red', alpha=0.75, linewidth=1.85, label = 'baseline villains')
    sns.move_legend(f_dist_ethoshist, frameon = True, loc = 'upper right', bbox_to_anchor = (0.98, 0.72))

    if contents_radio_unit == 'score':
        f_dist_ethoshist.set(xlabel = '', title = 'Distribution of villain scores')
    else:
        f_dist_ethoshist.set(xlabel = '', title = 'Number of (un)-favourable appeals to villains & heroes')


    heroes_tab1, heroes_tab2, heroes_tab_explore = st.tabs(['Bar-chart', 'Tables',  'Cases'])

    with heroes_tab2:
        add_spacelines()

        cops_names = df_dist_hist_all.corpus.unique()
        cols_columns = st.columns(len(cops_names))
        for n, c in enumerate(cols_columns):
            with c:
                df_dist_hist_all_0 = df_dist_hist_all[df_dist_hist_all.corpus == cops_names[n]]

                #st.write(cops_names[n])
                df_dist_hist_all_0 = df_dist_hist_all_0.sort_values(by = 'score', ascending=True)
                df_dist_hist_all_0 = df_dist_hist_all_0.reset_index(drop=True)
                df_dist_hist_all_0 = df_dist_hist_all_0.set_index('Target').reset_index()
                df_dist_hist_all_0.index += 1
                df_dist_hist_all_02 = pd.DataFrame(df_dist_hist_all_0.label.value_counts()).reset_index()
                if df_dist_hist_all_02.shape[0] == 1:
                    if not 'heroes' in df_dist_hist_all_02.label.unique():
                        df_dist_hist_all_02.loc[len(df_dist_hist_all_02)] = ['heroes', 0]
                    else:
                        df_dist_hist_all_02.loc[len(df_dist_hist_all_02)] = ['villains', 0]


                st.write( "##### Summary ", df_dist_hist_all_0.corpus.iloc[0] )
                st.write( df_dist_hist_all_02.style.apply( colorred, axis=1 ) )

                def highlight2(s):
                    if s.score < 0:
                        return ['background-color: red'] * len(s)
                    elif s.score > 0:
                        return ['background-color: green'] * len(s)
                    else:
                        return ['background-color: white'] * len(s)


                add_spacelines(2)
                st.write( "##### Detailed", df_dist_hist_all_0.corpus.iloc[0]  )
                st.write(df_dist_hist_all_0.style.apply( highlight2, axis=1 ))

                df_dist_hist_all_0.Target = df_dist_hist_all_0.Target.apply(lambda x: "_".join(x.split()))

                #st.write( "##### Cloud: names of heroes and villains" )
                #f_att0, _ = make_word_cloud(" ".join(df_dist_hist_all_0[df_dist_hist_all_0.label == 'villains'].Target.values), 800, 500, '#1E1E1E', 'Reds')
                #f_sup0, _ = make_word_cloud(" ".join(df_dist_hist_all_0[df_dist_hist_all_0.label == 'heroes'].Target.values), 800, 500, '#1E1E1E', 'Greens')

                #st.pyplot(f_sup0)
                #add_spacelines(2)
                #st.pyplot(f_att0)


    with heroes_tab1:
        add_spacelines(1)
        st.write( "##### Summary" )
        st.pyplot(f_dist_ethoshist)

        add_spacelines(2)
        st.write( "##### Detailed" )
        add_spacelines(1)
        #st.write(df_dist_hist_all)
        cutoff_neg = st.slider('Select a cut-off number for villains', 1, 25, 4)
        cutoff_pos = st.slider('Select a cut-off number for heroes', 1, 25, 2)
        cc = df_dist_hist_all.corpus.unique()
        #st.write(df_dist_hist_all)
        df_dist_hist_all = df_dist_hist_all[ ( (df_dist_hist_all.label == 'heroes') & (df_dist_hist_all.number > int(cutoff_pos)) ) |\
                                            ( (df_dist_hist_all.label == 'villains') & (df_dist_hist_all.number > int(cutoff_neg)) ) ]

        df_dist_hist_all = df_dist_hist_all.sort_values(by = [ 'corpus', 'label', 'number', ], ascending=[True, False, False])
        df_dist_ethos_all2_base = df_dist_hist_all[contents_radio_unit].mean().round(2)

        df_dist_hist_all = df_dist_hist_all.melt(['Target', 'label', 'corpus'])
        if "&" in df_dist_hist_all.corpus.iloc[0]:
            ds = "Covid & ElectionsSM"
            df_dist_hist_all['corpus'] = ds

        #df_dist_hist_all = df_dist_hist_all.sort_values(by = [ 'corpus', 'variable', 'label', 'value' ], ascending=[True,True, False, False])

        sns.set(font_scale=1.4, style='whitegrid')
        height_n = int(df_dist_hist_all.Target.nunique() / 3.2 )
        if df_dist_hist_all.Target.nunique() < 15:
                height_n = 9
                sns.set(font_scale=1.1, style='whitegrid')

        f_dist_ethoshist_barh = sns.catplot(kind='bar',
                        data = df_dist_hist_all[df_dist_hist_all.variable == contents_radio_unit], height=height_n, aspect=1.1,
                        x = 'value', y = 'Target', hue = 'label', dodge=False,# legend=False,
                        palette = {'villains':'#FF5656', 'heroes':'#078120'},
                        col = 'corpus', sharey=False, sharex=True, row = 'variable')

        f_dist_ethoshist_barh.set(ylabel = '', xlim = (-1, 1))
        #f_dist_ethoshist_barh.map(plt.axvline, x=df_dist_ethos_all2_base, ls='--', c='black', alpha=0.75, linewidth=2, label = 'baseline')

        #plt.tight_layout(pad=0.5)
        #sns.move_legend(f_dist_ethoshist_barh, bbox_to_anchor = (0.54, 1.1), ncols=3, loc='upper center', )
        f_dist_ethoshist_barh._legend.remove()
        if "&" in df_dist_hist_all.corpus.iloc[0]:
            xv = 0.4
        else:
            xv = -0.10
        plt.legend(bbox_to_anchor = (xv, 1.2), ncols=3, loc='upper center',)
        st.pyplot(f_dist_ethoshist_barh)


    with heroes_tab_explore:
        st.write('### Cases')

        if len(data_list) > 1:
            df = pd.concat( data_list, axis=0, ignore_index=True )
        else:
            df = data_list[0]

        df.Target = df.Target.astype('str')
        if "&" in df.corpus.iloc[0]:
            ds = "Covid & ElectionsSM"
            df['corpus'] = ds
        dff_columns = [ 'sentence', 'source', 'ethos_name', 'Target', 'pathos_name']# , 'conversation_id','date'
        df = df.drop_duplicates()

        df['ethos_name'] = df['ethos_label']
        df['pathos_name'] = df['pathos_label']

        if not 'neutral' in df['ethos_label'].unique():
            df['ethos_name'] = df['ethos_label'].map(ethos_mapping)
        if not 'negative' in df['pathos_label'].unique():
            df['pathos_name'] = df['pathos_label'].map(valence_mapping)


        dff = df.copy()
        select_columns = st.multiselect("Choose columns for specifying conditions", dff_columns, dff_columns[-2])
        cols_columns = st.columns(len(select_columns))
        dict_cond = {}
        for n, c in enumerate(cols_columns):
            with c:
                cond_col = st.multiselect(f"Choose condition for *{select_columns[n]}*",
                                       (dff[select_columns[n]].unique()), (dff[select_columns[n]].unique()[-1]))
                dict_cond[select_columns[n]] = cond_col
        dff_selected = dff.copy()
        dff_selected = dff_selected.drop_duplicates(subset = ['sentence'] )
        for i, k in enumerate(dict_cond.keys()):
            dff_selected = dff_selected[ dff_selected[str(k)].isin(dict_cond[k]) ]
        add_spacelines(2)
        st.dataframe(dff_selected[dff_columns].sort_values(by = select_columns).dropna(axis=1, how='all').reset_index(drop=True), width = None)
        st.write(f"No. of cases: {len(dff_selected.dropna(axis=1, how='all'))}.")






def TargetHeroScores_compare_prof(data_list, singl_an = True):
    #st.write("### (Anti)-hero Profile")
    add_spacelines(1)
    #contents_radio_heroes = st.radio("Category of the target of ethotic statements", ( "direct ethos", "3rd party ethos"))#"both",
    st.write("Choose category of the target of ethotic statements")
    box_direct = st.checkbox("direct ethos", value = False)
    box_3rd = st.checkbox("3rd party ethos", value = True)
    contents_radio_unit = 'number' #st.radio("Unit of analysis", ("score", "number"))

    up_data_dict = {}
    up_data_dicth = {}
    up_data_dictah = {}
    target_shared = {}
    up_data_dict_hist = {}

    n = 0
    for data in data_list:
        df = data[data['kind'] == 'ethos'].copy()
        ds = df['corpus'].iloc[0]
        if not 'attack' in df['ethos_label'].unique():
            df['ethos_label'] = df['ethos_label'].map(ethos_mapping)
        df["Target"] = df["Target"].astype('str')
        df = df[ (df.Target != 'nan') & (df.Target != '') & (df.ethos_label != 'neutral') ]
        df["Target"] = df["Target"].str.replace('Government', 'government')
        target_shared[n] = set(df["Target"].unique())
        df = df.drop_duplicates()

        if box_direct and not box_3rd:
            targets_limit = df['Target'].dropna().unique()
            targets_limit = [t for t in targets_limit if "@" in t]
            df = df[df.Target.isin(targets_limit)]
            df = df.drop_duplicates()
            target_shared[n] = set(df["Target"].unique())
            if len(targets_limit) < 2:
                st.error(f'No cases of **{contents_radio_heroes}** found in the chosen corpora.')
                st.stop()
        elif not box_direct and box_3rd:
            targets_limit = df['Target'].dropna().unique()
            targets_limit = [t for t in targets_limit if not "@" in t]
            df = df[df.Target.isin(targets_limit)]
            df = df.drop_duplicates()
            target_shared[n] = set(df["Target"].unique())
            if len(targets_limit) < 2:
                st.error(f'No cases of **{contents_radio_heroes}** found in the chosen corpora.')
                st.stop()

        dd2_size = df.groupby(['Target'], as_index=False).size()
        dd2_size = dd2_size[dd2_size['size'] > 1]
        adj_target = dd2_size['Target'].unique()

        dd = pd.DataFrame(df.groupby(['Target'])['ethos_label'].value_counts(normalize=False))
        dd.columns = ['value']
        dd = dd.reset_index()
        dd = dd[dd.Target.isin(adj_target)]
        dd = dd[dd.ethos_label != 'neutral']
        dd_hero = dd[dd.ethos_label == 'support']
        dd_antihero = dd[dd.ethos_label == 'attack']

        dd2 = pd.DataFrame({'Target': dd.Target.unique()})
        dd2_hist = dd2.copy()
        dd2anti_scores = []
        dd2hero_scores = []

        if contents_radio_unit == 'score':
            dd2['score'] = np.nan
            dd2['number'] = np.nan
            dd2['appeals'] = np.nan
            for t in dd.Target.unique():
                try:
                    h = dd_hero[dd_hero.Target == t]['value'].iloc[0]
                except:
                    h = 0
                try:
                    ah = dd_antihero[dd_antihero.Target == t]['value'].iloc[0]
                except:
                    ah = 0
                dd2hero_scores.append(h)
                dd2anti_scores.append(ah)
                i = dd2[dd2.Target == t].index
                dd2.loc[i, 'score'] = ah / (ah + h)
                if h > ah:
                    dd2.loc[i, 'number'] = h
                    dd2.loc[i, 'appeals'] = (ah + h)
                    dd2.loc[i, 'ethos_label'] = 'heroes'
                elif h < ah:
                    dd2.loc[i, 'number'] = ah
                    dd2.loc[i, 'ethos_label'] = 'villains'
                    dd2.loc[i, 'appeals'] = (ah + h)
                else:
                    dd2.loc[i, 'number'] = 0
                    dd2.loc[i, 'ethos_label'] = 'nn'
                    dd2.loc[i, 'appeals'] = (ah + h)

            dd2 = dd2[dd2.score != 0]
            dd2 = dd2[dd2.ethos_label != 'nn']
            #dd2['ethos_label'] = np.where(dd2.score < 0, 'villains', 'neutral')
            #dd2['ethos_label'] = np.where(dd2.score > 0, 'heroes', dd2['ethos_label'])
            dd2 = dd2.sort_values(by = ['ethos_label', 'Target'])
            #dd2['score'] = dd2['score'] * 100
            #dd2['score'] = dd2['score'].round()
            dd2['corpus'] = ds
            #st.write(dd2)
            up_data_dict_hist[n] = dd2
            #st.write(dd2)
            #st.stop()
            #dd2['score'] = dd2[dd2['ethos_label'] != 'neutral' ]
            dd2_dist = pd.DataFrame(dd2['ethos_label'].value_counts(normalize=True).round(3)*100).reset_index()
            dd2_dist.columns = ['heroes', 'percentage']
            dd2_dist['corpus'] = ds
            up_data_dict[n] = dd2_dist
            up_data_dicth[n] = dd2[dd2['ethos_label'] == 'heroes']['Target'].unique()
            up_data_dictah[n] = dd2[dd2['ethos_label'] == 'villains']['Target'].unique()
            n += 1

        else:
            dd2['score'] = np.nan
            dd2['number'] = np.nan
            dd2['appeals'] = np.nan
            for t in dd.Target.unique():
                try:
                    h = dd_hero[dd_hero.Target == t]['value'].iloc[0]
                except:
                    h = 0
                try:
                    ah = dd_antihero[dd_antihero.Target == t]['value'].iloc[0]
                except:
                    ah = 0
                dd2hero_scores.append(h)
                dd2anti_scores.append(ah)
                i = dd2[dd2.Target == t].index
                dd2.loc[i, 'score'] = ah / (ah + h)
                if h > ah:
                    dd2.loc[i, 'number'] = h
                    dd2.loc[i, 'appeals'] = (ah + h)
                    dd2.loc[i, 'ethos_label'] = 'heroes'
                elif h < ah:
                    dd2.loc[i, 'number'] = ah
                    dd2.loc[i, 'ethos_label'] = 'villains'
                    dd2.loc[i, 'appeals'] = (ah + h)
                else:
                    dd2.loc[i, 'number'] = 0
                    dd2.loc[i, 'ethos_label'] = 'nn'
                    dd2.loc[i, 'appeals'] = (ah + h)

            dd2 = dd2[dd2.score != 0]
            dd2 = dd2[dd2.ethos_label != 'nn']
            #dd2['ethos_label'] = np.where(dd2.score < 0, 'villains', 'neutral')
            #dd2['ethos_label'] = np.where(dd2.score > 0, 'heroes', dd2['ethos_label'])
            dd2 = dd2.sort_values(by = ['ethos_label', 'Target'])
            dd2['corpus'] = ds
            #st.write(dd2)
            up_data_dict_hist[n] = dd2
            #st.write(dd2)
            #st.stop()
            #dd2['score'] = dd2[dd2['ethos_label'] != 'neutral' ]
            #dd2_dist = pd.DataFrame(dd2['ethos_label'].value_counts(normalize=False)).reset_index()
            dd2_dist = pd.DataFrame(dd2['ethos_label'].value_counts(normalize=True).round(3)*100).reset_index()
            dd2_dist.columns = ['heroes', 'percentage']
            dd2_dist['corpus'] = ds
            up_data_dict[n] = dd2_dist
            up_data_dicth[n] = dd2[dd2['ethos_label'] == 'heroes']['Target'].unique()
            up_data_dictah[n] = dd2[dd2['ethos_label'] == 'villains']['Target'].unique()
            n += 1

    df_dist_ethos_all = up_data_dict[0].copy()
    for k in range(int(len(up_data_dict.keys()))-1):
        k_sub = k+1
        df_dist_ethos_all = pd.concat([df_dist_ethos_all, up_data_dict[k_sub]], axis=0, ignore_index=True)

    sns.set(font_scale=1.35, style='whitegrid')


    df_dist_hist_all = up_data_dict_hist[0].copy()
    for k in range(int(len(up_data_dict_hist.keys()))-1):
        k_sub = k+1
        df_dist_hist_all = pd.concat([df_dist_hist_all, up_data_dict_hist[k_sub]], axis=0, ignore_index=True)

    sns.set(font_scale=1.35, style='whitegrid')

    df_dist_hist_all = df_dist_hist_all.rename(columns = {'ethos_label':'label'})
    sns.set(font_scale=1, style='whitegrid')


    singl_an = True
    if singl_an:

        if len( list(target_shared.keys()) )  > 1:
            target_shared_list = []
            kk = list(target_shared.keys())
            target_shared_list = list( set(target_shared[kk[0]]).intersection(set(target_shared[kk[1]])) )

        else:
            target_shared_list = target_shared[0]

        selected_target = st.selectbox("Choose a target entity you would like to analyse", set(target_shared_list))


    heroes_tab1, heroes_tab2, explore_tab = st.tabs(['Bar-chart', 'Tables', 'Cases'])

    with heroes_tab1:

            cols_columns = st.columns(len(data_list), gap='large')
            for n, c in enumerate(cols_columns):
                with c:
                    df = data_list[n].copy()
                    df = df.drop_duplicates()
                    if "&" in df['corpus'].iloc[0]:
                        ds = "Covid & ElectionsSM"
                        df['corpus'] = ds
                    ds = df['corpus'].iloc[0]
                    st.write("##### Corpus "+ ds )
                    #st.dataframe(df)
                    if not 'neutral' in df['ethos_label'].unique():
                        df['ethos_label'] = df['ethos_label'].map(ethos_mapping)
                    if not 'negative' in df['pathos_label'].unique():
                        df['pathos_label'] = df['pathos_label'].map(valence_mapping)

                    # all df targets
                    df_target_all = pd.DataFrame(df[df.ethos_label.isin( ['attack', 'support'] )]['ethos_label'].value_counts(normalize = True).round(2)*100)
                    df_target_all.columns = ['percentage']
                    df_target_all.reset_index(inplace=True)
                    df_target_all.columns = ['label', 'percentage']
                    df_target_all = df_target_all.sort_values(by = 'label')
                    df_target_all_att = df_target_all[df_target_all.label == 'attack']['percentage'].iloc[0]
                    df_target_all_sup = df_target_all[df_target_all.label == 'support']['percentage'].iloc[0]

                    # chosen target df

                    df_target = pd.DataFrame(df[ (df.Target == str(selected_target)) & \
                    (df.ethos_label.isin( ['attack', 'support'] )) ]['ethos_label'].value_counts(normalize = True).round(2)*100)
                    df_target.columns = ['percentage']
                    df_target.reset_index(inplace=True)
                    df_target.columns = ['label', 'percentage']

                    if len(df_target) == 1:
                      if not ("attack" in df_target.label.unique()):
                          df_target.loc[len(df_target)] = ["attack", 0]
                      elif not ("support" in df_target.label.unique()):
                          df_target.loc[len(df_target)] = ["support", 0]
                    df_target = df_target.sort_values(by = 'label')
                    df_target_att = df_target[df_target.label == 'attack']['percentage'].iloc[0]
                    df_target_sup = df_target[df_target.label == 'support']['percentage'].iloc[0]

                    add_spacelines(1)
                    df_target.columns = ['ethos', 'percentage']
                    df_dist_ethos = df_target.sort_values(by = 'ethos')
                    df_dist_ethos['corpus'] = ds
                    if "&" in ds:
                        ds = "Covid & ElectionsSM"
                        df_dist_ethos['corpus'] = ds

                    sns.set(font_scale=1.35, style='whitegrid')
                    f_dist_ethos = sns.catplot(kind='bar', data = df_dist_ethos, height=4, aspect=1.4, legend = False,
                                    x = 'ethos', y = 'percentage', hue = 'ethos', dodge=False,
                                    palette = {'attack':'#BB0000', 'neutral':'#949494', 'support':'#026F00'})
                    vals_senti = df_dist_ethos['percentage'].values.round(1)
                    plt.title(f"Ethos towards **{str(selected_target)}** in {df_dist_ethos.corpus.iloc[0]} \n")
                    plt.xlabel('')
                    plt.ylim(0, 105)
                    plt.yticks(np.arange(0, 105, 20))
                    for index_senti, v in enumerate(vals_senti):
                        plt.text(x=index_senti , y = v+1 , s=f"{v}%" , fontdict=dict(ha='center'))

                    add_spacelines(1)
                    col1, col2 = st.columns([3, 2])
                    with col2:
                        st.write("**Hero score** 👑")
                        col2.metric(str(selected_target), str(df_target_sup)+ str('%'), str(round((df_target_sup - df_target_all_sup),  1))+ str(' p.p.'),
                        help = f"Percentage of social media posts that support *{selected_target}* and the difference from the average.") # round(((df_target_sup / df_target_all_sup) * 100) - 100, 1)

                    with col1:
                        st.write("**Villain score** 👎")
                        col1.metric(str(selected_target), str(df_target_att)+ str('%'), str(round((df_target_att - df_target_all_att),  1))+ str(' p.p.'), delta_color="inverse",
                        help = f"Percentage of social media posts that attack *{selected_target}* and the difference from the average.")
                    add_spacelines(1)

                    st.pyplot(f_dist_ethos)


    with heroes_tab2:
        add_spacelines()
        sns.set(font_scale=2, style='whitegrid')
        if len(data_list) > 1:
            df = pd.concat( data_list, axis=0, ignore_index=True )
        else:
            df = data_list[0]
        if "&" in df.corpus.iloc[0]:
            ds = "Covid & ElectionsSM"
            df['corpus'] = ds

        if not 'negative' in df['pathos_label'].unique():
            df['pathos_label'] = df['pathos_label'].map(valence_mapping).str.replace('neutral p', 'neutral')

        if not 'neutral' in df['ethos_label'].unique():
            df['ethos_label'] = df['ethos_label'].map(ethos_mapping)

        df['pathos_name'] = df['pathos_label']
        df.loc[df['pathos_name'].isna(), 'pathos_name'] = df.loc[df['pathos_name'].isna(), 'pathos_label']
        df.loc[~(df['source'].str.startswith("@")), 'source'] = df.loc[~(df['source'].str.startswith("@")), 'source'].apply(lambda x: "@" + x)
        df = df.drop_duplicates()

        cols = ['sentence', 'ethos_label', 'source', 'Target', 'pathos_name', 'corpus'] #, 'date', 'conversation_id'
        if len(df[df.Target == str(selected_target)]) == 1:
            st.write(f"{len(df[df.Target == str(selected_target)])} case of ethotic statements towards **{selected_target}** ")
        else:
            st.write(f"{len(df[df.Target == str(selected_target)])} cases of ethotic statements towards **{selected_target}** ")
        if not "neutral" in df['pathos_label'].unique():
            df['pathos_label'] = df['pathos_label'].map(valence_mapping)
        st.dataframe(df[df.Target == str(selected_target)][cols].set_index('source').rename(columns={'ethos_label':'ethos', 'pathos_label':'pathos'}), width = None)
        add_spacelines(1)

    with explore_tab:
        add_spacelines()
        select_columns = st.multiselect("Choose columns for specifying conditions", cols, cols[1])
        dff = df.copy()
        cols_columns = st.columns(len(select_columns))
        dict_cond = {}
        for n, c in enumerate(cols_columns):
            with c:
                cond_col = st.multiselect(f"Choose condition for *{select_columns[n]}*",
                                       (dff[select_columns[n]].unique()), (dff[select_columns[n]].unique()[-1]))
                dict_cond[select_columns[n]] = cond_col
        dff_selected = dff.copy()
        for i, k in enumerate(dict_cond.keys()):
            dff_selected = dff_selected[ dff_selected[str(k)].isin(dict_cond[k]) ]
        add_spacelines(2)
        st.dataframe(dff_selected[cols].sort_values(by = select_columns).drop_duplicates().reset_index(drop=True).dropna(how='all', axis=1), width = None)
        st.write(f"No. of cases: {len(dff_selected)}.")


def TargetHeroScores_compare_word(data_list, chbox_3rd, chbox_direct, singl_an = True):
    #st.write("### (Anti)-hero WordCloud")

    contents_radio_unit = 'number' #st.radio("Unit of analysis", ("score", "number"))


    up_data_dict = {}
    up_data_dicth = {}
    up_data_dictah = {}
    target_shared = {}
    up_data_dict_hist = {}

    n = 0
    for data in data_list:
        df = data.copy()
        ds = df['corpus'].iloc[0]
        if not 'attack' in df['ethos_label'].unique():
            df['ethos_label'] = df['ethos_label'].map(ethos_mapping)
        df["Target"] = df["Target"].astype('str')
        df = df[ (df.Target != 'nan') & (df.Target != '') & (df.ethos_label != 'neutral') ]
        df["Target"] = df["Target"].str.replace('Government', 'government')
        df = df.drop_duplicates()
        target_shared[n] = set(df["Target"].unique())

        if chbox_direct and not chbox_3rd:
            targets_limit = df['Target'].dropna().unique()
            targets_limit = [t for t in targets_limit if "@" in t]
            df = df[df.Target.isin(targets_limit)]
            target_shared[n] = set(df["Target"].unique())
            if len(targets_limit) < 2:
                st.error(f'No cases of **{contents_radio_heroes}** found in the chosen corpora.')
                st.stop()

        elif not chbox_direct and chbox_3rd:
            targets_limit = df['Target'].dropna().unique()
            targets_limit = [t for t in targets_limit if not "@" in t]
            df = df[df.Target.isin(targets_limit)]
            target_shared[n] = set(df["Target"].unique())
            if len(targets_limit) < 2:
                st.error(f'No cases of **{contents_radio_heroes}** found in the chosen corpora.')
                st.stop()

        dd2_size = df.groupby(['Target'], as_index=False).size()
        dd2_size = dd2_size[dd2_size['size'] > 1]
        adj_target = dd2_size['Target'].unique()

        dd = pd.DataFrame(df.groupby(['Target'])['ethos_label'].value_counts(normalize=False))
        dd.columns = ['value']
        dd = dd.reset_index()
        dd = dd[dd.Target.isin(adj_target)]
        dd = dd[dd.ethos_label != 'neutral']
        dd_hero = dd[dd.ethos_label == 'support']
        dd_antihero = dd[dd.ethos_label == 'attack']

        dd2 = pd.DataFrame({'Target': dd.Target.unique()})
        dd2_hist = dd2.copy()
        dd2anti_scores = []
        dd2hero_scores = []

        if contents_radio_unit == 'score':
            dd2['score'] = np.nan
            dd2['number'] = np.nan
            dd2['appeals'] = np.nan
            for t in dd.Target.unique():
                try:
                    h = dd_hero[dd_hero.Target == t]['value'].iloc[0]
                except:
                    h = 0
                try:
                    ah = dd_antihero[dd_antihero.Target == t]['value'].iloc[0]
                except:
                    ah = 0
                dd2hero_scores.append(h)
                dd2anti_scores.append(ah)
                i = dd2[dd2.Target == t].index
                dd2.loc[i, 'score'] = ah / (ah + h)
                if h > ah:
                    dd2.loc[i, 'number'] = h
                    dd2.loc[i, 'appeals'] = (ah + h)
                    dd2.loc[i, 'ethos_label'] = 'heroes'
                elif h < ah:
                    dd2.loc[i, 'number'] = ah
                    dd2.loc[i, 'ethos_label'] = 'villains'
                    dd2.loc[i, 'appeals'] = (ah + h)
                else:
                    dd2.loc[i, 'number'] = 0
                    dd2.loc[i, 'ethos_label'] = 'nn'
                    dd2.loc[i, 'appeals'] = (ah + h)

            dd2 = dd2[dd2.score != 0]
            dd2 = dd2[dd2.ethos_label != 'nn']
            #dd2['ethos_label'] = np.where(dd2.score < 0, 'villains', 'neutral')
            #dd2['ethos_label'] = np.where(dd2.score > 0, 'heroes', dd2['ethos_label'])
            dd2 = dd2.sort_values(by = ['ethos_label', 'Target'])
            #dd2['score'] = dd2['score'] * 100
            #dd2['score'] = dd2['score'].round()
            dd2['corpus'] = ds
            #st.write(dd2)
            up_data_dict_hist[n] = dd2
            #st.write(dd2)
            #st.stop()
            #dd2['score'] = dd2[dd2['ethos_label'] != 'neutral' ]
            dd2_dist = pd.DataFrame(dd2['ethos_label'].value_counts(normalize=True).round(3)*100).reset_index()
            dd2_dist.columns = ['heroes', 'percentage']
            dd2_dist['corpus'] = ds
            up_data_dict[n] = dd2_dist
            up_data_dicth[n] = dd2[dd2['ethos_label'] == 'heroes']['Target'].unique()
            up_data_dictah[n] = dd2[dd2['ethos_label'] == 'villains']['Target'].unique()
            n += 1

        else:
            dd2['score'] = np.nan
            dd2['number'] = np.nan
            dd2['appeals'] = np.nan
            for t in dd.Target.unique():
                try:
                    h = dd_hero[dd_hero.Target == t]['value'].iloc[0]
                except:
                    h = 0
                try:
                    ah = dd_antihero[dd_antihero.Target == t]['value'].iloc[0]
                except:
                    ah = 0
                dd2hero_scores.append(h)
                dd2anti_scores.append(ah)
                i = dd2[dd2.Target == t].index
                dd2.loc[i, 'score'] = ah / (ah + h)
                if h > ah:
                    dd2.loc[i, 'number'] = h
                    dd2.loc[i, 'appeals'] = (ah + h)
                    dd2.loc[i, 'ethos_label'] = 'heroes'
                elif h < ah:
                    dd2.loc[i, 'number'] = ah
                    dd2.loc[i, 'ethos_label'] = 'villains'
                    dd2.loc[i, 'appeals'] = (ah + h)
                else:
                    dd2.loc[i, 'number'] = 0
                    dd2.loc[i, 'ethos_label'] = 'nn'
                    dd2.loc[i, 'appeals'] = (ah + h)

            dd2 = dd2[dd2.score != 0]
            dd2 = dd2[dd2.ethos_label != 'nn']
            #dd2['ethos_label'] = np.where(dd2.score < 0, 'villains', 'neutral')
            #dd2['ethos_label'] = np.where(dd2.score > 0, 'heroes', dd2['ethos_label'])
            dd2 = dd2.sort_values(by = ['ethos_label', 'Target'])
            dd2['corpus'] = ds
            #st.write(dd2)
            up_data_dict_hist[n] = dd2
            #st.write(dd2)
            #st.stop()
            #dd2['score'] = dd2[dd2['ethos_label'] != 'neutral' ]
            #dd2_dist = pd.DataFrame(dd2['ethos_label'].value_counts(normalize=False)).reset_index()
            dd2_dist = pd.DataFrame(dd2['ethos_label'].value_counts(normalize=True).round(3)*100).reset_index()
            dd2_dist.columns = ['heroes', 'percentage']
            dd2_dist['corpus'] = ds
            up_data_dict[n] = dd2_dist
            up_data_dicth[n] = dd2[dd2['ethos_label'] == 'heroes']['Target'].unique()
            up_data_dictah[n] = dd2[dd2['ethos_label'] == 'villains']['Target'].unique()
            n += 1

    df_dist_ethos_all = up_data_dict[0].copy()
    for k in range(int(len(up_data_dict.keys()))-1):
        k_sub = k+1
        df_dist_ethos_all = pd.concat([df_dist_ethos_all, up_data_dict[k_sub]], axis=0, ignore_index=True)

    sns.set(font_scale=1.35, style='whitegrid')


    df_dist_hist_all = up_data_dict_hist[0].copy()
    for k in range(int(len(up_data_dict_hist.keys()))-1):
        k_sub = k+1
        df_dist_hist_all = pd.concat([df_dist_hist_all, up_data_dict_hist[k_sub]], axis=0, ignore_index=True)

    sns.set(font_scale=1.35, style='whitegrid')
    df_dist_hist_all = df_dist_hist_all.rename(columns = {'ethos_label':'label'})


    if len(data_list) > 1:
            df = pd.concat( data_list, axis=0, ignore_index=True )
    else:
            df = data_list[0]

    singl_an = True
    df = df.drop_duplicates()        
    if singl_an:
        if len( list(target_shared.keys()) )  > 1:
            target_shared_list = []
            kk = list(target_shared.keys())
            target_shared_list = list( set(target_shared[kk[0]]).intersection(set(target_shared[kk[1]])) )

        else:
            target_shared_list = target_shared[0]

    add_spacelines(1)
    selected_target = st.selectbox("Choose a target entity you would like to analyse", set(target_shared_list))

    # chosen target df
    df = df[ (df.Target == str(selected_target)) &  (df.ethos_label.isin( ['attack', 'support'] )) ]
    df0 = df.copy()

    if "&" in df.corpus.iloc[0]:
            ds = "Covid & ElectionsSM"
            df['corpus'] = ds

    sns.set(font_scale=1.35, style='whitegrid')
    add_spacelines(1)
    corps = df.corpus.unique()

    heroes_tab1, heroes_tab2, heroes_tab_explore = st.tabs(['Plot', 'Tables', 'Cases'])

    for corp in corps:

        with heroes_tab1:
            add_spacelines(1)
            st.write(corp)
            #df = df0[ df0['corpus'] == corp ]
            df = clean_text(df, 'sentence_lemmatized', text_column_name = "sentence_lemmatized")
            df = df.drop_duplicates()
            data_neutral = df[df.ethos_label != label_cloud]
            neu_text = " ".join(data_neutral['sentence_lemmatized'].apply(lambda x: " ".join(t for t in set(x.split()))).to_numpy())
            count_dict_df_neu_text = Counter(neu_text.split(" "))

            df_neu_text = pd.DataFrame( {"word": list(count_dict_df_neu_text.keys()),
                                        'other #': list(count_dict_df_neu_text.values())} )

            data_attack = df[df.ethos_label == label_cloud]

            att_text = " ".join(data_attack['sentence_lemmatized'].apply(lambda x: " ".join(t for t in set(x.split()))).to_numpy())
            count_dict_df_att_text = Counter(att_text.split(" "))
            df_att_text = pd.DataFrame( {"word": list(count_dict_df_att_text.keys()),
                                        label_cloud+' #': list(count_dict_df_att_text.values())} )


            df_for_wordcloud = pd.merge(df_att_text, df_neu_text, on = 'word', how = 'outer')
            df_for_wordcloud.fillna(0, inplace=True)
            #st.write(df_for_wordcloud)

            df_for_wordcloud['general #'] = df_for_wordcloud['other #'].astype('int') + df_for_wordcloud[label_cloud+' #'].astype('int')
            df_for_wordcloud['word'] = df_for_wordcloud['word'].str.replace("'", "_").replace("”", "_").replace("’", "_")
            df_for_wordcloud.sort_values(by = label_cloud + ' #', inplace=True, ascending=False)
            df_for_wordcloud.reset_index(inplace=True, drop=True)
            #st.write(df_for_wordcloud)

            analysis_for = label_cloud
            df_for_wordcloud['precis'] = (round(df_for_wordcloud[label_cloud+' #'] / df_for_wordcloud['general #'], 3) * 100).apply(float) # att

            #fig_cloud1, df_cloud_words1, figure_cloud_words1 = wordcloud_lexeme(df_for_wordcloud, lexeme_threshold = threshold_cloud, analysis_for = str(label_cloud))
            if label_cloud == 'attack':
              #print(f'Analysis for: {analysis_for} ')
              cmap_wordcloud = 'Reds' #gist_heat
            elif label_cloud == 'both':
              #print(f'Analysis for: {analysis_for} ')
              cmap_wordcloud = 'autumn' #viridis
            else:
              #print(f'Analysis for: {analysis_for} ')
              cmap_wordcloud = 'Greens'

            dfcloud = df_for_wordcloud[(df_for_wordcloud['precis'] >= int(threshold_cloud)) & (df_for_wordcloud['general #'] > 1) & (df_for_wordcloud.word.map(len)>3)]
            #print(f'There are {len(dfcloud)} words for the analysis of language {analysis_for} with precis threshold equal to {lexeme_threshold}.')
            try:
                    n_words = dfcloud['word'].nunique()
            except:
                    st.error('No words with a specified threshold.')
                    st.stop()
            text = []
            for i in dfcloud.index:
              w = dfcloud.loc[i, 'word']
              w = str(w).strip()
              if analysis_for == 'both':
                n = int(dfcloud.loc[i, 'support #'] + dfcloud.loc[i, 'attack #'])
              else:
                n = int(dfcloud.loc[i, str(analysis_for)+' #']) #  + dfcloud.loc[i, 'attack #']   dfcloud.loc[i, 'support #']+  general
              l = np.repeat(w, n)
              text.extend(l)

            import random
            random.shuffle(text)
            #st.write(f"There are {n_words} words.")
            if n_words < 1:
                st.error('No words with a specified threshold. \n Try lower value of threshold.')
                st.stop()
            figure_cloud, figure_cloud_words = make_word_cloud(" ".join(text), 1000, 620, '#1E1E1E', str(cmap_wordcloud), stops = box_stopwords)
            #st.write(f"There are {len(figure_cloud_words)} words.")
            #st.pyplot(figure_cloud)

            df_cloud_words1 = dfcloud.copy()
            df_cloud_words1 = df_cloud_words1.rename(columns = {'general #':'overall #', 'precis':'precision'})
            df_cloud_words1 = df_cloud_words1.sort_values(by = 'precision', ascending = False)
            df_cloud_words1 = df_cloud_words1.reset_index(drop = True)
            df_cloud_words1.index += 1
            #st.write(df_cloud_words1)

            st.write(f"There are {len(figure_cloud_words)} words.")
            st.pyplot(figure_cloud)


        with heroes_tab2:
            add_spacelines(1)
            st.write(corp)
            st.write(df_cloud_words1)


    with heroes_tab_explore:
        add_spacelines()
        cols_odds1 = ['source', 'sentence', 'ethos_label',  'Target',  'freq_words_'+label_cloud, 'corpus']

        if selected_rhet_dim == 'logos':
            df = df.rename(columns = {'connection':'logos'})
            #cols_odds1 = ['locution_conclusion', 'locution_premise', 'logos', 'argument_linked', 'freq_words_'+label_cloud]
            cols_odds1 = ['locution_conclusion', 'locution_premise','premise', 'conclusion', 'sentence_lemmatized', 'logos', 'argument_linked', 'freq_words_'+label_cloud]
            df['sentence_lemmatized'] = df['sentence_lemmatized'].astype('str')
            df['logos'] = df['logos'].map({'Default Inference':'support', 'Default Conflict':'attack'})

        pos_list_freq = df_cloud_words1.word.tolist()
        freq_word_pos = st.multiselect('Choose word(s) you would like to see data cases for', pos_list_freq, pos_list_freq[:2])
        df_odds_pos_words = set(freq_word_pos)
        df['freq_words_'+label_cloud] = df.sentence_lemmatized.apply(lambda x: " ".join( set(x.split()).intersection(df_odds_pos_words) ))
        df = df.drop_duplicates().dropna(axis=1, how='all')
        #st.write(df)
        if "&" in df.corpus.iloc[0]:
            ds = "Covid & ElectionsSM"
            df['corpus'] = ds
        add_spacelines(1)
        st.write(f'Cases with **{freq_word_pos}** words:')
        st.dataframe(df[ (df['freq_words_'+label_cloud].str.split().map(len) >= 1) & (df[selected_rhet_dim] == label_cloud) ][cols_odds1])# .set_index('source')





def TargetHeroScores_compare_old(data_list, singl_an = True):
    st.write("### Villains & heroes")
    add_spacelines(1)
    contents_radio_heroes = st.radio("Category of the target of ethotic statements", ("both", "direct ethos", "3rd party ethos"))

    up_data_dict = {}
    up_data_dicth = {}
    up_data_dictah = {}
    target_shared = {}
    up_data_dict_hist = {}

    n = 0
    for data in data_list:
        df = data.copy()
        ds = df['corpus'].iloc[0]
        if not 'attack' in df['ethos_label'].unique():
            df['ethos_label'] = df['ethos_label'].map(ethos_mapping)
        df["Target"] = df["Target"].astype('str')
        df = df[ (df.Target != 'nan') & (df.Target != '') & (df.ethos_label != 'neutral') ]
        df["Target"] = df["Target"].str.replace('Government', 'government')
        target_shared[n] = set(df["Target"].unique())

        if contents_radio_heroes == "direct ethos":
            targets_limit = df['Target'].dropna().unique()
            targets_limit = [t for t in targets_limit if "@" in t]
            df = df[df.Target.isin(targets_limit)]
            if len(targets_limit) < 2:
                st.error(f'No cases of **{contents_radio_heroes}** found in the chosen corpora.')
                st.stop()
        elif contents_radio_heroes == "3rd party ethos":
            targets_limit = df['Target'].dropna().unique()
            targets_limit = [t for t in targets_limit if not "@" in t]
            df = df[df.Target.isin(targets_limit)]
            if len(targets_limit) < 2:
                st.error(f'No cases of **{contents_radio_heroes}** found in the chosen corpora.')
                st.stop()

        dd2_size = df.groupby(['Target'], as_index=False).size()
        dd2_size = dd2_size[dd2_size['size'] > 1]
        adj_target = dd2_size['Target'].unique()

        dd = pd.DataFrame(df.groupby(['Target'])['ethos_label'].value_counts(normalize=True))
        dd.columns = ['normalized_value']
        dd = dd.reset_index()
        dd = dd[dd.Target.isin(adj_target)]
        dd = dd[dd.ethos_label != 'neutral']
        dd_hero = dd[dd.ethos_label == 'support']
        dd_antihero = dd[dd.ethos_label == 'attack']

        dd2 = pd.DataFrame({'Target': dd.Target.unique()})
        dd2_hist = dd2.copy()
        dd2anti_scores = []
        dd2hero_scores = []
        dd2['score'] = np.nan
        for t in dd.Target.unique():
            try:
                h = dd_hero[dd_hero.Target == t]['normalized_value'].iloc[0]
            except:
                h = 0
            try:
                ah = dd_antihero[dd_antihero.Target == t]['normalized_value'].iloc[0]
            except:
                ah = 0
            dd2hero_scores.append(h)
            dd2anti_scores.append(ah)
            i = dd2[dd2.Target == t].index
            dd2.loc[i, 'score'] = h - ah

        dd2 = dd2[dd2.score != 0]
        dd2['ethos_label'] = np.where(dd2.score < 0, 'villains', 'neutral')
        dd2['ethos_label'] = np.where(dd2.score > 0, 'heroes', dd2['ethos_label'])
        dd2 = dd2.sort_values(by = ['ethos_label', 'Target'])
        dd2['score'] = dd2['score'] * 100
        dd2['score'] = dd2['score'].round()
        dd2['corpus'] = ds
        up_data_dict_hist[n] = dd2
        #st.write(dd2)
        #st.stop()
        #dd2['score'] = dd2[dd2['ethos_label'] != 'neutral' ]
        dd2_dist = pd.DataFrame(dd2['ethos_label'].value_counts(normalize=True).round(3)*100).reset_index()
        dd2_dist.columns = ['heroes', 'percentage']
        dd2_dist['corpus'] = ds
        up_data_dict[n] = dd2_dist
        up_data_dicth[n] = dd2[dd2['ethos_label'] == 'heroes']['Target'].unique()
        up_data_dictah[n] = dd2[dd2['ethos_label'] == 'villains']['Target'].unique()
        n += 1

    df_dist_ethos_all = up_data_dict[0].copy()
    for k in range(int(len(up_data_dict.keys()))-1):
        k_sub = k+1
        df_dist_ethos_all = pd.concat([df_dist_ethos_all, up_data_dict[k_sub]], axis=0, ignore_index=True)

    sns.set(font_scale=1.35, style='whitegrid')
    f_dist_ethos = sns.catplot(kind='bar', data = df_dist_ethos_all, height=5, aspect=1.2,
                    x = 'heroes', y = 'percentage', hue = 'heroes', dodge=False,
                    palette = {'villains':'#FF4444', 'heroes':'#298A32'},
                    col = 'corpus')
    f_dist_ethos.set(ylim=(0, 110), xlabel = '')
    for ax in f_dist_ethos.axes.ravel():
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')
    add_spacelines(1)


    df_dist_hist_all = up_data_dict_hist[0].copy()
    for k in range(int(len(up_data_dict_hist.keys()))-1):
        k_sub = k+1
        df_dist_hist_all = pd.concat([df_dist_hist_all, up_data_dict_hist[k_sub]], axis=0, ignore_index=True)
    sns.set(font_scale=1.35, style='whitegrid')
    #st.write(df_dist_hist_all)
    #st.stop()
    #f_dist_ethoshist = sns.catplot(kind='count', data = df_dist_hist_all, height=5, aspect=1.3,
    #                x = 'score', hue = 'ethos_label', dodge=False,
    #                palette = {'villains':'#FF4444', 'heroes':'#298A32'},
    #                col = 'corpus')
    #for axes in f_dist_ethoshist.axes.flat:
    #    _ = axes.set_xticklabels(axes.get_xticklabels(), rotation=90)

    df_dist_hist_all = df_dist_hist_all.rename(columns = {'ethos_label':'label'})
    sns.set(font_scale=1, style='whitegrid')
    f_dist_ethoshist = sns.catplot(kind='strip', data = df_dist_hist_all, height=4, aspect=1.25,
                    y = 'score', hue = 'label', dodge=False, s=35, alpha=0.8,
                    palette = {'villains':'#FF4444', 'heroes':'#298A32'},
                    x = 'corpus')
    f_dist_ethoshist.set(xlabel = '', title = 'Distribution of villain scores')
    sns.move_legend(f_dist_ethoshist, frameon = True, loc = 'upper right', bbox_to_anchor = (0.98, 0.72))

    heroes_tab1, heroes_tab2 = st.tabs(['Heroes & villains Plots', 'Heroes & villains Tables'])
    with heroes_tab1:
        add_spacelines(1)
        st.pyplot(f_dist_ethos)
        add_spacelines(2)
        st.pyplot(f_dist_ethoshist)

        add_spacelines(2)
        if singl_an:
            st.write("### Single Target Analysis")
            add_spacelines(1)

            target_shared_list = target_shared[0]
            for n in range(int(len(data_list))-1):
                target_shared_list = set(target_shared_list).intersection(target_shared[n+1])
            selected_target = st.selectbox("Choose a target entity you would like to analyse", set(target_shared_list))

            cols_columns = st.columns(len(data_list), gap='large')
            for n, c in enumerate(cols_columns):
                with c:
                    df = data_list[n].copy()
                    ds = df['corpus'].iloc[0]
                    #st.dataframe(df)
                    if not 'neutral' in df['ethos_label'].unique():
                        df['ethos_label'] = df['ethos_label'].map(ethos_mapping)
                    if not 'negative' in df['pathos_label'].unique():
                        df['pathos_label'] = df['pathos_label'].map(valence_mapping)

                    # all df targets
                    df_target_all = pd.DataFrame(df[df.ethos_label != 'neutral']['ethos_label'].value_counts(normalize = True).round(2)*100)
                    df_target_all.columns = ['percentage']
                    df_target_all.reset_index(inplace=True)
                    df_target_all.columns = ['label', 'percentage']
                    df_target_all = df_target_all.sort_values(by = 'label')
                    df_target_all_att = df_target_all[df_target_all.label == 'attack']['percentage'].iloc[0]
                    df_target_all_sup = df_target_all[df_target_all.label == 'support']['percentage'].iloc[0]

                    # chosen target df
                    df_target = pd.DataFrame(df[df.Target == str(selected_target)]['ethos_label'].value_counts(normalize = True).round(2)*100)
                    df_target.columns = ['percentage']
                    df_target.reset_index(inplace=True)
                    df_target.columns = ['label', 'percentage']

                    if len(df_target) == 1:
                      if not ("attack" in df_target.label.unique()):
                          df_target.loc[len(df_target)] = ["attack", 0]
                      elif not ("support" in df_target.label.unique()):
                          df_target.loc[len(df_target)] = ["support", 0]
                    df_target = df_target.sort_values(by = 'label')
                    df_target_att = df_target[df_target.label == 'attack']['percentage'].iloc[0]
                    df_target_sup = df_target[df_target.label == 'support']['percentage'].iloc[0]

                    add_spacelines(1)
                    df_target.columns = ['ethos', 'percentage']
                    df_dist_ethos = df_target.sort_values(by = 'ethos')
                    df_dist_ethos['corpus'] = ds

                    sns.set(font_scale=1.35, style='whitegrid')
                    f_dist_ethos = sns.catplot(kind='bar', data = df_dist_ethos, height=4, aspect=1.4,
                                    x = 'ethos', y = 'percentage', hue = 'ethos', dodge=False, col = 'corpus',
                                    palette = {'attack':'#BB0000', 'neutral':'#022D96', 'support':'#026F00'})
                    vals_senti = df_dist_ethos['percentage'].values.round(1)
                    plt.title(f"Ethos towards **{str(selected_target)}** in {df.corpus.iloc[0]} \n")
                    plt.xlabel('')
                    plt.ylim(0, 105)
                    plt.yticks(np.arange(0, 105, 20))
                    for index_senti, v in enumerate(vals_senti):
                        plt.text(x=index_senti , y = v+1 , s=f"{v}%" , fontdict=dict(ha='center'))
                    st.pyplot(f_dist_ethos)



                    st.write('**********************************************************************************')
                    #add_spacelines(1)
                    cols = ['sentence', 'ethos_label', 'source', 'Target', 'pathos_label'] #, 'date', 'conversation_id'
                    if len(df[df.Target == str(selected_target)]) == 1:
                        st.write(f"{len(df[df.Target == str(selected_target)])} case of ethotic statements towards **{selected_target}**  in {df['corpus'].iloc[0]} corpus")
                    else:
                        st.write(f"{len(df[df.Target == str(selected_target)])} cases of ethotic statements towards **{selected_target}**  in {df['corpus'].iloc[0]} corpus")
                    if not "neutral" in df['pathos_label'].unique():
                        df['pathos_label'] = df['pathos_label'].map(valence_mapping)
                    st.dataframe(df[df.Target == str(selected_target)][cols].set_index('source').rename(columns={'ethos_label':'ethos'}), width = None)
                    add_spacelines(1)


    with heroes_tab2:
        cops_names = df_dist_hist_all.corpus.unique()
        cols_columns = st.columns(len(cops_names))
        for n, c in enumerate(cols_columns):
            with c:
                df_dist_hist_all_0 = df_dist_hist_all[df_dist_hist_all.corpus == cops_names[n]]
                add_spacelines(1)
                st.write(cops_names[n])
                df_dist_hist_all_0 = df_dist_hist_all_0.sort_values(by = 'score')
                df_dist_hist_all_0 = df_dist_hist_all_0.reset_index(drop=True)
                st.write(df_dist_hist_all_0.set_index('Target'))
                df_dist_hist_all_0.Target = df_dist_hist_all_0.Target.apply(lambda x: "_".join(x.split()))

                f_att0, _ = make_word_cloud(" ".join(df_dist_hist_all_0[df_dist_hist_all_0.label == 'villains'].Target.values), 800, 500, '#1E1E1E', 'Reds')
                f_sup0, _ = make_word_cloud(" ".join(df_dist_hist_all_0[df_dist_hist_all_0.label == 'heroes'].Target.values), 800, 500, '#1E1E1E', 'Greens')

                add_spacelines(1)
                st.pyplot(f_att0)
                add_spacelines(2)
                st.pyplot(f_sup0)





def TargetHeroScores(data_list):
  st.write("### Villains & heroes")
  tabheroes1, tabheroes2 = st.tabs(['Heroes & villains Plots', 'Heroes & villains Tables'])

  add_spacelines(1)
  with tabheroes1:
    contents_radio_heroes = st.radio("Category of the target of ethotic statements", ("both", "direct ethos", "3rd party ethos"))

    up_data_dict = {}
    up_data_dicth = {}
    up_data_dictah = {}
    dd_target_table = pd.DataFrame(columns = ['Target', 'score', 'ethos_label'])
    n = 0
    for data in data_list:
        df = data.copy()
        ds = df['corpus'].iloc[0]
        if not 'attack' in df['ethos_label'].unique():
            df['ethos_label'] = df['ethos_label'].map(ethos_mapping)
        df["Target"] = df["Target"].astype('str')
        df = df[ (df.Target != 'nan') & (df.Target != '') & (df.ethos_label != 'neutral') ]
        df["Target"] = df["Target"].str.replace('Government', 'government')

        if contents_radio_heroes == "direct ethos":
            targets_limit = df['Target'].dropna().unique()
            targets_limit = [t for t in targets_limit if "@" in t]
            df = df[df.Target.isin(targets_limit)]
            if len(targets_limit) < 2:
                st.error(f'No cases of **{contents_radio_heroes}** found in the chosen corpora.')
                st.stop()
        elif contents_radio_heroes == "3rd party ethos":
            targets_limit = df['Target'].dropna().unique()
            targets_limit = [t for t in targets_limit if not "@" in t]
            df = df[df.Target.isin(targets_limit)]
            if len(targets_limit) < 2:
                st.error(f'No cases of **{contents_radio_heroes}** found in the chosen corpora.')
                st.stop()

        dd2_size = df.groupby(['Target'], as_index=False).size()
        dd2_size = dd2_size[dd2_size['size'] > 1]
        adj_target = dd2_size['Target'].unique()

        dd = pd.DataFrame(df.groupby(['Target'])['ethos_label'].value_counts(normalize=True))
        dd.columns = ['normalized_value']
        dd = dd.reset_index()
        dd = dd[dd.Target.isin(adj_target)]
        dd = dd[dd.ethos_label != 'neutral']
        dd_hero = dd[dd.ethos_label == 'support']
        dd_antihero = dd[dd.ethos_label == 'attack']

        dd2 = pd.DataFrame({'Target': dd.Target.unique()})
        dd2_hist = dd2.copy()
        dd2anti_scores = []
        dd2hero_scores = []
        dd2['score'] = np.nan
        for t in dd.Target.unique():
            try:
                h = dd_hero[dd_hero.Target == t]['normalized_value'].iloc[0]
            except:
                h = 0
            try:
                ah = dd_antihero[dd_antihero.Target == t]['normalized_value'].iloc[0]
            except:
                ah = 0
            dd2hero_scores.append(h)
            dd2anti_scores.append(ah)
            i = dd2[dd2.Target == t].index
            dd2.loc[i, 'score'] = h - ah

        dd2 = dd2[dd2.score != 0]
        dd2['ethos_label'] = np.where(dd2.score < 0, 'villains', 'neutral')
        dd2['ethos_label'] = np.where(dd2.score > 0, 'heroes', dd2['ethos_label'])
        dd2 = dd2.sort_values(by = ['ethos_label', 'Target'])
        dd2['score'] = dd2['score'] * 100
        dd_target_table = pd.concat([dd_target_table, dd2], axis = 0, ignore_index = True)
        #dd2['score'] = dd2[dd2['ethos_label'] != 'neutral' ]
        dd2_dist = pd.DataFrame(dd2['ethos_label'].value_counts(normalize=True).round(3)*100).reset_index()
        dd2_dist.columns = ['heroes', 'percentage']
        dd2_dist['corpus'] = ds
        up_data_dict[n] = dd2_dist
        up_data_dicth[n] = dd2[dd2['ethos_label'] == 'heroes']['Target'].unique()
        up_data_dictah[n] = dd2[dd2['ethos_label'] == 'villains']['Target'].unique()
        n += 1

    df_dist_ethos_all = up_data_dict[0].copy()
    for k in range(int(len(up_data_dict.keys()))-1):
        k_sub = k+1
        df_dist_ethos_all = pd.concat([df_dist_ethos_all, up_data_dict[k_sub]], axis=0, ignore_index=True)

    sns.set(font_scale=1.25, style='whitegrid')
    f_dist_ethos = sns.catplot(kind='bar', data = df_dist_ethos_all, height=5, aspect=1.2,
                    x = 'heroes', y = 'percentage', hue = 'heroes', dodge=False,
                    palette = {'villains':'#FF4444', 'heroes':'#298A32'},
                    col = 'corpus')
    f_dist_ethos.set(ylim=(0, 110), xlabel = '')
    for ax in f_dist_ethos.axes.ravel():
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')
    add_spacelines(1)
    with st.container():
        st.pyplot(f_dist_ethos)

    add_spacelines(2)
    st.write("### Single Target Analysis")
    add_spacelines(1)
    selected_target = st.selectbox("Choose a target entity you would like to analyse", set(adj_target))

    # all df targets
    df_target_all = pd.DataFrame(df[df.ethos_label != 'neutral']['ethos_label'].value_counts(normalize = True).round(2)*100)
    df_target_all.columns = ['percentage']
    df_target_all.reset_index(inplace=True)
    df_target_all.columns = ['label', 'percentage']
    df_target_all = df_target_all.sort_values(by = 'label')

    df_target_all_att = df_target_all[df_target_all.label == 'attack']['percentage'].iloc[0]
    df_target_all_sup = df_target_all[df_target_all.label == 'support']['percentage'].iloc[0]

    # chosen target df
    df_target = pd.DataFrame(df[df.Target == str(selected_target)]['ethos_label'].value_counts(normalize = True).round(2)*100)
    df_target.columns = ['percentage']
    df_target.reset_index(inplace=True)
    df_target.columns = ['label', 'percentage']

    if len(df_target) == 1:
      if not ("attack" in df_target.label.unique()):
          df_target.loc[len(df_target)] = ["attack", 0]
      elif not ("support" in df_target.label.unique()):
          df_target.loc[len(df_target)] = ["support", 0]

    df_target = df_target.sort_values(by = 'label')
    df_target_att = df_target[df_target.label == 'attack']['percentage'].iloc[0]
    df_target_sup = df_target[df_target.label == 'support']['percentage'].iloc[0]


    with st.container():
        st.info(f'Selected entity: ** {str(selected_target)} **')
        add_spacelines(1)
        col2, col1 = st.columns([3, 2])
        with col1:
            st.subheader("Positivity score")
            col1.metric(str(selected_target), str(df_target_sup)+ str('%') + f" ({len(df[ (df.Target == str(selected_target)) & (df['ethos_label'] == 'support') ])})" ,
            str(round((df_target_sup - df_target_all_sup),  1))+ str(' p.p.'),
            help = f"Percentage (number in brackets) of texts that support ** {str(selected_target)} **") # round(((df_target_sup / df_target_all_sup) * 100) - 100, 1)

        with col2:
            st.subheader("Negativity score")
            col2.metric(str(selected_target), str(df_target_att)+ str('%') + f" ({len(df[ (df.Target == str(selected_target)) & (df['ethos_label'] == 'attack') ])})",
            str(round((df_target_att - df_target_all_att),  1))+ str(' p.p.'), delta_color="inverse",
            help = f"Percentage (number in brackets) of texts that attack ** {str(selected_target)} **") # ((df_target_att / df_target_all_att) * 100) - 100, 1)

        add_spacelines(2)

        #if not ("neutral" in df_target.label.unique()):
            #df_target.loc[len(df_target)] = ["neutral", 0]
        df_target.columns = ['ethos', 'percentage']
        df_dist_ethos = df_target.sort_values(by = 'ethos')

        sns.set(font_scale=1.25, style='whitegrid')
        f_dist_ethos = sns.catplot(kind='bar', data = df_dist_ethos, height=4, aspect=1.4,
                        x = 'ethos', y = 'percentage', hue = 'ethos', dodge=False,
                        palette = {'attack':'#BB0000', 'neutral':'#022D96', 'support':'#026F00'})
        vals_senti = df_dist_ethos['percentage'].values.round(1)
        plt.title(f"Ethos towards **{str(selected_target)}** in {df.corpus.iloc[0]} \n")
        plt.xlabel('')
        plt.ylim(0, 105)
        plt.yticks(np.arange(0, 105, 20))
        for index_senti, v in enumerate(vals_senti):
            plt.text(x=index_senti , y = v+1 , s=f"{v}%" , fontdict=dict(ha='center'))

        plot1, plot2, plot3 = st.columns([1, 6, 1], gap='small')
        with plot2:
            st.pyplot(f_dist_ethos)

        st.write('**********************************************************************************')
        #add_spacelines(1)
        cols = [
            'sentence', 'ethos_label', 'source', 'Target', 'pathos_label'] #, 'date', 'conversation_id'
        #st.write('#### Cases of ethotic statements towards **', selected_target, ' **')
        if len(df[df.Target == str(selected_target)]) == 1:
            st.write(f"{len(df[df.Target == str(selected_target)])} case of ethotic statements towards ** {selected_target} **  in {df['corpus'].iloc[0]} corpus")
        else:
            st.write(f"{len(df[df.Target == str(selected_target)])} cases of ethotic statements towards ** {selected_target} **  in {df['corpus'].iloc[0]} corpus")

        if not "neutral" in df['pathos_label'].unique():
            df['pathos_label'] = df['pathos_label'].map(valence_mapping)
        st.dataframe(df[df.Target == str(selected_target)][cols].set_index('source').rename(columns={'ethos_label':'ethos'}), width = None)


  with tabheroes2:
      add_spacelines(2)
      st.write("##### Anti-(heroes)")
      dd_target_table.score = dd_target_table.score.round()
      dd_target_table = dd_target_table.sort_values(by = 'score')
      dd_target_table = dd_target_table.reset_index(drop = True)
      dd_target_table.index += 1
      dd_target_table.columns = ['Target', 'Score', 'Label']
      st.write(dd_target_table.set_index('Target'))
  add_spacelines(1)




def distribution_plot(data):
    df = data.copy()

    if not 'neutral' in df['ethos_label'].unique():
        df['ethos_label'] = df['ethos_label'].map(ethos_mapping)
    if not 'negative' in df['pathos_label'].unique():
        df['pathos_label'] = df['pathos_label'].map(valence_mapping)

    st.write("### Ethos distribution")
    add_spacelines(2)
    contents_radio_targs = st.radio("Category of the target of ethotic statements", ("both", "direct ethos", "3rd party ethos"))

    df["Target"] = df["Target"].astype('str')
    df["Target"] = df["Target"].str.replace('Government', 'government')

    if contents_radio_targs == "direct ethos":
        targets_limit = df['Target'].dropna().unique()
        targets_limit = [t for t in targets_limit if "@" in t]
        targets_limit.append('nan')
        df = df[df.Target.isin(targets_limit)]
        if len(targets_limit) < 1:
            st.error(f'No cases of **{contents_radio_targs}** found in the chosen corpora.')
            st.stop()
    elif contents_radio_targs == "3rd party ethos":
        targets_limit = df['Target'].dropna().unique()
        targets_limit = [t for t in targets_limit if not "@" in t]
        targets_limit.append('nan')
        df = df[df.Target.isin(targets_limit)]
        if len(targets_limit) < 1:
            st.error(f'No cases of **{contents_radio_targs}** found in the chosen corpora.')
            st.stop()

    df_dist_ethos = pd.DataFrame(df['ethos_label'].value_counts(normalize = True).round(2)*100)
    df_dist_ethos.columns = ['percentage']
    df_dist_ethos.reset_index(inplace=True)
    df_dist_ethos.columns = ['ethos', 'percentage']
    df_dist_ethos = df_dist_ethos.sort_values(by = 'ethos')

    per = []
    eth = []
    eth.append('no ethos')
    per.append(float(df_dist_ethos[df_dist_ethos.ethos == 'neutral']['percentage'].iloc[0]))
    eth.append('ethos')
    per.append(100 - float(df_dist_ethos[df_dist_ethos.ethos == 'neutral']['percentage'].iloc[0]))
    df_dist_ethos_all0 = pd.DataFrame({'ethos':eth, 'percentage':per})

    sns.set(font_scale=1.1, style='whitegrid')
    f_dist_ethos0 = sns.catplot(kind='bar', data = df_dist_ethos_all0, height=4.5, aspect=1.4,
                    x = 'ethos', y = 'percentage', hue = 'ethos', dodge=False,
                    palette = {'ethos':'#EA9200', 'no ethos':'#022D96'})
    f_dist_ethos0.set(ylim=(0, 110))
    plt.xlabel("")
    plt.title(f"Ethos distribution in **{contents_radio}** \n")
    vals_senti0 = df_dist_ethos_all0['percentage'].values.round(1)
    for index_senti2, v in enumerate(vals_senti0):
        plt.text(x=index_senti2, y = v+1 , s=f"{v}%" , fontdict=dict(fontsize=13, ha='center'))

    f_dist_ethos = sns.catplot(kind='bar', data = df_dist_ethos, height=4.5, aspect=1.4,
                    x = 'ethos', y = 'percentage', hue = 'ethos', dodge=False,
                    palette = {'attack':'#BB0000', 'neutral':'#022D96', 'support':'#026F00'})
    vals_senti = df_dist_ethos['percentage'].values.round(1)
    f_dist_ethos.set(ylim=(0, 110))
    plt.title(f"Ethos distribution in **{contents_radio}** \n")
    for index_senti, v in enumerate(vals_senti):
        plt.text(x=index_senti , y = v+1 , s=f"{v}%" , fontdict=dict(fontsize=13, ha='center'))


    df_dist_ethos2 = pd.DataFrame(df[df['ethos_label'] != 'neutral']['ethos_label'].value_counts(normalize = True).round(2)*100)

    df_dist_ethos2.columns = ['percentage']
    df_dist_ethos2.reset_index(inplace=True)
    df_dist_ethos2.columns = ['ethos', 'percentage']
    df_dist_ethos2 = df_dist_ethos2.sort_values(by = 'ethos')

    f_dist_ethos2 = sns.catplot(kind='bar', data = df_dist_ethos2, height=4.5, aspect=1.4,
                    x = 'ethos', y = 'percentage', hue = 'ethos', dodge=False,
                    palette = {'attack':'#BB0000', 'support':'#026F00'})
    f_dist_ethos2.set(ylim=(0, 110))
    plt.title(f"Ethos distribution in **{contents_radio}** \n")
    vals_senti2 = df_dist_ethos2['percentage'].values.round(1)
    for index_senti2, v in enumerate(vals_senti2):
        plt.text(x=index_senti2, y = v+1 , s=f"{v}%" , fontdict=dict(fontsize=13, ha='center'))

    plot1_dist_ethos, plot2_dist_ethos, plot3_dist_ethos = st.columns([1, 8, 1])
    with plot1_dist_ethos:
        st.write('')
    with plot2_dist_ethos:
        st.pyplot(f_dist_ethos0)
        add_spacelines(1)
        st.pyplot(f_dist_ethos)
        add_spacelines(1)
        st.pyplot(f_dist_ethos2)
    with plot3_dist_ethos:
        st.write('')
    add_spacelines(2)


    with st.expander("Pathos distribution"):
        add_spacelines(1)

    if contents_radio_targs == "direct ethos":
        targets_limit = df['Target'].dropna().unique()
        targets_limit = [t for t in targets_limit if "@" in t]
        targets_limit.append('nan')
        df = df[df.Target.isin(targets_limit)]
        if len(targets_limit) < 1:
            st.error(f'No cases of **{contents_radio_targs}** found in the chosen corpora.')
            st.stop()
    elif contents_radio_targs == "3rd party ethos":
        targets_limit = df['Target'].dropna().unique()
        targets_limit = [t for t in targets_limit if not "@" in t]
        targets_limit.append('nan')
        df = df[df.Target.isin(targets_limit)]

        if not 'neutral' in df['pathos_label'].unique():
            df['pathos_label'] = df['pathos_label'].map(valence_mapping)
        df_dist_ethos = pd.DataFrame(df['pathos_label'].value_counts(normalize = True).round(2)*100)
        df_dist_ethos.columns = ['percentage']
        df_dist_ethos.reset_index(inplace=True)
        df_dist_ethos.columns = ['pathos', 'percentage']
        df_dist_ethos = df_dist_ethos.sort_values(by = 'pathos')


        per = []
        eth = []
        eth.append('no pathos')
        per.append(float(df_dist_ethos[df_dist_ethos.pathos == 'neutral']['percentage'].iloc[0]))
        eth.append('pathos')
        per.append(100 - float(df_dist_ethos[df_dist_ethos.pathos == 'neutral']['percentage'].iloc[0]))
        df_dist_ethos_all0 = pd.DataFrame({'pathos':eth, 'percentage':per})

        f_dist_ethos0 = sns.catplot(kind='bar', data = df_dist_ethos_all0, height=4.5, aspect=1.4,
                        x = 'pathos', y = 'percentage', hue = 'pathos', dodge=False,
                        palette = {'pathos':'#EA9200', 'no pathos':'#022D96'})
        f_dist_ethos0.set(ylim=(0, 110))
        plt.xlabel("")
        plt.title(f"Pathos distribution in **{contents_radio}** \n")
        vals_senti0 = df_dist_ethos_all0['percentage'].values.round(1)
        for index_senti2, v in enumerate(vals_senti0):
            plt.text(x=index_senti2, y = v+1 , s=f"{v}%" , fontdict=dict(fontsize=13, ha='center'))

        f_dist_ethos = sns.catplot(kind='bar', data = df_dist_ethos, height=4.5, aspect=1.4,
                        x = 'pathos', y = 'percentage', hue = 'pathos', dodge=False,
                        palette = {'negative':'#BB0000', 'neutral':'#022D96', 'positive':'#026F00'})
        vals_senti = df_dist_ethos['percentage'].values.round(1)
        f_dist_ethos.set(ylim=(0, 110))
        plt.title(f"Pathos distribution in **{contents_radio}** \n")
        for index_senti, v in enumerate(vals_senti):
            plt.text(x=index_senti , y = v+1 , s=f"{v}%" , fontdict=dict(fontsize=12, ha='center'))


        df_dist_ethos2 = pd.DataFrame(df[df['pathos_label'] != 'neutral']['pathos_label'].value_counts(normalize = True).round(2)*100)
        df_dist_ethos2.columns = ['percentage']
        df_dist_ethos2.reset_index(inplace=True)
        df_dist_ethos2.columns = ['pathos', 'percentage']
        df_dist_ethos2 = df_dist_ethos2.sort_values(by = 'pathos')

        sns.set(font_scale=1.1, style='whitegrid')
        f_dist_ethos2 = sns.catplot(kind='bar', data = df_dist_ethos2, height=4.5, aspect=1.4,
                        x = 'pathos', y = 'percentage', hue = 'pathos', dodge=False,
                        palette = {'negative':'#BB0000', 'positive':'#026F00'})
        vals_senti2 = df_dist_ethos2['percentage'].values.round(1)
        f_dist_ethos2.set(ylim=(0, 110))
        plt.title(f"Pathos distribution in **{contents_radio}** \n")
        for index_senti2, v in enumerate(vals_senti2):
            plt.text(x=index_senti2, y = v+1 , s=f"{v}%" , fontdict=dict(fontsize=12, ha='center'))

        plot1_dist_ethos, plot2_dist_ethos, plot3_dist_ethos = st.columns([1, 8, 1])
        with plot1_dist_ethos:
            st.write('')
        with plot2_dist_ethos:
            st.pyplot(f_dist_ethos0)
            add_spacelines(1)
            st.pyplot(f_dist_ethos)
            add_spacelines(1)
            st.pyplot(f_dist_ethos2)
        with plot3_dist_ethos:
            st.write('')
        add_spacelines(1)



def distribution_plot_compare_logos(data_list, an_type):
    c_contents1, c_contents2, c_contents3 = st.columns( 3 )
    with c_contents1:
        contents_radio_categories_val_units = st.radio("Choose the unit of statistic to display", ("percentage", "number" ) ) # horizontal=True, , label_visibility = 'collapsed'

    with c_contents2:
        contents_radio_categories = st.radio("Choose categories to display", ("3-LEP categories", "6-LEP categories", "4-E categories" )) # horizontal=True,

    if contents_radio_categories == "4-E categories":
        with c_contents3:
            contents_radio_targs = st.radio("Choose category of targets of ethotic statements", ("direct ethos", "3rd party ethos"))
    else:
        with c_contents3:
            st.write("")


    up_data_dict = {}
    up_data_dict2 = {}
    up_data_dict_ethos = {}

    df_dict_exp = {}
    n = 0
    ne = 0
    de = 0
    naming = 'category'

    map_naming = {'attack':'Ethos Attack', 'neutral':'No Ethos', 'support':'Ethos Support',
            'Default Conflict': 'Logos Attack',
            'Neutral' : 'No Logos',
            'Default Inference' : 'Logos Support',
            'neutral p': 'No Pathos', 'negative': 'Negative Pathos', 'positive': 'Positive Pathos' }

    naming_2categories = {'attack':'Ethos Attack', 'support':'Ethos Support',
            'Default Conflict': 'Logos Attack',          'Default Inference' : 'Logos Support',
            'negative': 'Negative Pathos', 'positive': 'Positive Pathos' }

    map_naming_bin =  {'attack':'Ethos', 'neutral':'No Ethos', 'support':'Ethos',
            'Default Conflict': 'Logos',
            'Neutral' : 'No Logos',
            'Default Inference' : 'Logos',
            'neutral p': 'No Pathos', 'negative': 'Pathos', 'positive': 'Pathos' }


    for nn in np.arange(0, int(len(data_list)), 1):
        data = data_list[nn]
        #nn = nn+1
        #st.write(data, data_list[nn])
        df = data.copy()
        ds = df['corpus'].iloc[0]
        if '&' in ds:
            ds = 'Covid & ElectionsSM'
        df['corpus'] = ds

        #st.dataframe(df)
        if df['kind'].iloc[0] == 'ethos':
            naming_cols = 'ethos_label'

            if not 'attack' in df['ethos_label'].unique():
                df['ethos_label'] = df['ethos_label'].map(ethos_mapping)
            if not 'neutral' in df['pathos_label'].unique():
                df['pathos_label'] = df['pathos_label'].map(valence_mapping)

            df["Target"] = df["Target"].astype('str')
            df["Target"] = df["Target"].str.replace('Government', 'government').str.replace('@CNN', 'CNN')

            if contents_radio_categories == "4-E categories":

                if contents_radio_targs == "direct ethos":
                    targets_limit = df['Target'].dropna().unique()
                    targets_limit = [t for t in targets_limit if "@" in t]
                    if 'Hansard' in ds:
                        targets_limit = list( df['Target'].dropna().unique() )
                    targets_limit.append('nan')
                    df = df[df.Target.isin(targets_limit)]
                    if len(targets_limit) < 1:
                        st.error(f'No cases of **{contents_radio_targs}** found in the chosen corpora.')
                        st.stop()
                elif contents_radio_targs == "3rd party ethos":
                    targets_limit = df['Target'].dropna().unique()
                    targets_limit = [t for t in targets_limit if not "@" in t]
                    if 'Hansard' in ds:
                        targets_limit = []
                    targets_limit.append('nan')
                    df = df[df.Target.isin(targets_limit)]
                    if len(targets_limit) < 1:
                        st.error(f'No cases of **{contents_radio_targs}** found in the chosen corpora.')
                        st.stop()

            if contents_radio_categories_val_units == 'percentage':
                df_dist_ethos = pd.DataFrame(df['pathos_label'].map(map_naming_bin).value_counts(normalize = True).round(2)*100)
            else:
                df_dist_ethos = pd.DataFrame(df['pathos_label'].map(map_naming_bin).value_counts(normalize = False))

            df_dist_ethos.columns = ['percentage']
            df_dist_ethos.reset_index(inplace=True)
            #st.dataframe(df_dist_ethos)
            df_dist_ethos.columns = [naming, 'percentage']
            df_dist_ethos = df_dist_ethos.sort_values(by = naming)
            df_dist_ethos['corpus'] = ds
            #df_dist_ethos[naming] = df_dist_ethos[naming].map(map_naming_bin)
            up_data_dict[n] = df_dist_ethos

            if contents_radio_categories_val_units == 'percentage':
                df_dist_ethos = pd.DataFrame( df[ df['pathos_label'].isin(['negative', 'positive']) ].pathos_label.value_counts(normalize = True).round(2)*100)
            else:
                df_dist_ethos = pd.DataFrame( df[ df['pathos_label'].isin(['negative', 'positive']) ].pathos_label.value_counts(normalize = False) )

            df_dist_ethos.columns = ['percentage']
            df_dist_ethos.reset_index(inplace=True)
            #st.dataframe(df_dist_ethos)
            df_dist_ethos.columns = [naming, 'percentage']
            df_dist_ethos = df_dist_ethos.sort_values(by = naming)
            df_dist_ethos['corpus'] = ds
            df_dist_ethos[naming] = df_dist_ethos[naming].map(map_naming)
            up_data_dict2[n] = df_dist_ethos
            n += 1

            if contents_radio_categories_val_units == 'percentage':
                df_dist_ethos2 = pd.DataFrame(df[ (df[naming_cols].isin( naming_2categories.keys() )) ][naming_cols].value_counts(normalize = True).round(2)*100)
            else:
                df_dist_ethos2 = pd.DataFrame(df[ (df[naming_cols].isin( naming_2categories.keys() )) ][naming_cols].value_counts(normalize = False) )

            df_dist_ethos2.columns = ['percentage']
            df_dist_ethos2.reset_index(inplace=True)
            df_dist_ethos2.columns = [naming, 'percentage']
            df_dist_ethos2[naming] = df_dist_ethos2[naming].map(map_naming)
            df_dist_ethos2 = df_dist_ethos2.sort_values(by = naming)
            df_dist_ethos2['corpus'] = ds
            up_data_dict_ethos[ne] = df_dist_ethos2
            ne += 1
            df_dict_exp[de] = df
            de += 1


        elif df['kind'].iloc[0] == 'logos':
            #df = data_list[nn]
            naming_cols = 'connection'
            df[naming_cols] = df[naming_cols].map( { 'Default Conflict': 'Default Conflict', 'Default Inference' :'Default Inference' } ).fillna('Neutral')
            #df_dist_ethos = pd.DataFrame(df[naming_cols].value_counts(normalize = True).round(2)*100)
            connection_cats = ['Default Conflict', 'Default Inference']
            #df = df[ df[naming_cols].isin(connection_cats) ]
            df_dict_exp[de] = df
            de += 1

        if contents_radio_categories_val_units == 'percentage':
            df_dist_ethos = pd.DataFrame(df[naming_cols].map(map_naming_bin).value_counts(normalize = True).round(2)*100)
        else:
            df_dist_ethos = pd.DataFrame(df[naming_cols].map(map_naming_bin).value_counts(normalize = False) )

        df_dist_ethos.columns = ['percentage']
        df_dist_ethos.reset_index(inplace=True)
        #st.dataframe(df_dist_ethos)
        df_dist_ethos.columns = [naming, 'percentage']
        df_dist_ethos = df_dist_ethos.sort_values(by = naming)
        df_dist_ethos['corpus'] = ds
        #df_dist_ethos[naming] = df_dist_ethos[naming]
        up_data_dict[n] = df_dist_ethos

        if contents_radio_categories_val_units == 'percentage':
            df_dist_ethos2 = pd.DataFrame(df[ (df[naming_cols].isin( naming_2categories.keys() )) ][naming_cols].value_counts(normalize = True).round(2)*100)
        else:
            df_dist_ethos2 = pd.DataFrame(df[ (df[naming_cols].isin( naming_2categories.keys() )) ][naming_cols].value_counts(normalize = False) )

        df_dist_ethos2.columns = ['percentage']
        df_dist_ethos2.reset_index(inplace=True)
        df_dist_ethos2.columns = [naming, 'percentage']
        df_dist_ethos2[naming] = df_dist_ethos2[naming].map(map_naming)
        df_dist_ethos2 = df_dist_ethos2.sort_values(by = naming)
        df_dist_ethos2['corpus'] = ds
        up_data_dict2[n] = df_dist_ethos2
        n += 1

    df_dist_ethos_all = up_data_dict[0].copy()
    for k in range(int(len(up_data_dict.keys()))-1):
        k_sub = k+1
        df_dist_ethos_all = pd.concat([df_dist_ethos_all, up_data_dict[k_sub]], axis=0, ignore_index=True)


    naming_cats = set( ['Ethos Attack', 'No Ethos', 'Ethos Support',
                    'Logos Attack', 'No Logos','Logos Support',
                      'No Pathos',  'Negative Pathos', 'Positive Pathos' ] )
    ds = df_dist_ethos_all.corpus.iloc[0]

    df_dist_ethos_all['cat'] = df_dist_ethos_all[naming].map({
    'Ethos Attack':'Ethos', 'Ethos Support':'Ethos', 'Logos Attack':'Logos',
    'Logos Support':'Logos', 'Negative Pathos':'Pathos', 'Positive Pathos':'Pathos',
    'No Ethos':'Ethos', 'No Logos':'Logos', 'No Pathos':'Pathos',
    'Ethos':'Ethos', 'Pathos':'Pathos', 'Logos':'Logos',
    })
    df_dist_ethos_all = df_dist_ethos_all.sort_values(by = ['cat', naming])
    df_dist_ethos_all2_base = df_dist_ethos_all[df_dist_ethos_all.category.isin(['Ethos', 'Pathos', 'Logos'])].percentage.mean().round()
    #st.write(df_dist_ethos_all2_base)
    df_dist_ethos_all['deviation'] = df_dist_ethos_all.loc[:, 'percentage'] - df_dist_ethos_all2_base

    sns.set(font_scale=1.4, style='whitegrid')

    f_dist_ethos = sns.catplot(kind='bar', data = df_dist_ethos_all[df_dist_ethos_all.category.isin( ['Ethos', 'Logos', 'Pathos'] )],
    height=4.5, aspect=1.4,
                    x = naming, y = 'percentage', hue = naming, dodge=False, legend = False,
                    palette = {'Ethos':'#FB598F', 'Logos':'#DD8309', 'Pathos':'#EBEB04'}, # colors_log,
                    col = 'corpus', )
    if contents_radio_categories_val_units == 'percentage':
        f_dist_ethos.set(ylim=(0, 101), xlabel='', ylabel = str(contents_radio_categories_val_units) )
    else:
        f_dist_ethos.set(xlabel='', ylabel = str(contents_radio_categories_val_units),
        ylim=(0, df_dist_ethos_all[df_dist_ethos_all.category.isin( ['Ethos', 'Logos', 'Pathos'] )]['percentage'].max()+401 ), )

    #hatches = ['/', 'x', '.']
    #bars = f_dist_ethos.patches
    #for pat,bar in zip(hatches,bars):
    #       bar.set_hatch(pat)

    f_dist_ethos.map(plt.axhline, y=df_dist_ethos_all2_base, ls='--', c='black', alpha=0.65, linewidth=2, label = 'baseline')
    plt.legend(loc = 'upper left', bbox_to_anchor = (0.67, 0.93))

    for ax in f_dist_ethos.axes.ravel():
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()),
            ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')

    f_dist_ethos_dev = sns.catplot(kind='bar', data = df_dist_ethos_all[df_dist_ethos_all.category.isin( ['Ethos', 'Logos', 'Pathos'] )],
    height=4.5, aspect=1.4,
                    x = naming, y = 'deviation', hue = naming, dodge=False, legend = False,
                    palette = {'Ethos':'#FB598F', 'Logos':'#DD8309', 'Pathos':'#EBEB04'}, # colors_log,
                    col = 'corpus', )
    hatches = ['/', 'x', '.']

    if contents_radio_categories_val_units == 'percentage':
        f_dist_ethos_dev.set( xlabel='', ylabel = 'deviation',
        ylim = ( (df_dist_ethos_all[df_dist_ethos_all.category.isin( ['Ethos', 'Logos', 'Pathos'] )]['deviation'].abs().max() *-1) - 6,
                df_dist_ethos_all[df_dist_ethos_all.category.isin( ['Ethos', 'Logos', 'Pathos'] )]['deviation'].abs().max()+ 16))

    else:
        f_dist_ethos_dev.set( xlabel='', ylabel = 'deviation',
        ylim = ( (df_dist_ethos_all[df_dist_ethos_all.category.isin( ['Ethos', 'Logos', 'Pathos'] )]['deviation'].abs().max() *-1) - 51,
                df_dist_ethos_all[df_dist_ethos_all.category.isin( ['Ethos', 'Logos', 'Pathos'] )]['deviation'].abs().max()+ 151))



    f_dist_ethos_dev.map(plt.axhline, y=0, ls='--', c='black', alpha=0.75, linewidth=1.6, label = 'baseline')
    plt.legend(loc = 'upper left', bbox_to_anchor = (0.67, 0.955))

    for ax in f_dist_ethos_dev.axes.ravel():
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()),
            ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')


    df_dist_ethos_all2 = up_data_dict2[0].copy()
    for k in range(int(len(up_data_dict2.keys()))-1):
        k_sub2 = k+1
        df_dist_ethos_all2 = pd.concat([df_dist_ethos_all2, up_data_dict2[k_sub2]], axis=0, ignore_index=True)

    #st.dataframe(df_dist_ethos_all2)
    naming_cats = set( ['Ethos Attack', 'Ethos Support', 'Logos Attack', 'Logos Support', 'Negative Pathos', 'Positive Pathos'] )

    df_dist_ethos_all2['cat'] = df_dist_ethos_all2[naming].map({
    'Ethos Attack':'Ethos', 'Ethos Support':'Ethos', 'Logos Attack':'Logos',
    'Logos Support':'Logos', 'Negative Pathos':'Pathos', 'Positive Pathos':'Pathos',
    'No Ethos':'Ethos', 'No Logos':'Logos', 'No Pathos':'Pathos',
    'Ethos':'Ethos', 'Pathos':'Pathos', 'Logos':'Logos',
    })
    df_dist_ethos_all2['cat base'] = df_dist_ethos_all2[naming].map({
    'Ethos Attack':'Attack', 'Ethos Support':'Support', 'Logos Attack':'Attack',
    'Logos Support':'Support', 'Negative Pathos':'Attack', 'Positive Pathos':'Support',
    }).fillna('Neutral')
    df_dist_ethos_all2 = df_dist_ethos_all2.sort_values(by = ['cat', naming])
    df_dist_ethos_all2_base = df_dist_ethos_all2.groupby('cat base', as_index=False).percentage.mean().round()

    df_dist_ethos_all2['deviation'] = np.nan
    df_dist_ethos_all2.loc[df_dist_ethos_all2['cat base'] == 'Attack', 'deviation'] = df_dist_ethos_all2.loc[df_dist_ethos_all2['cat base'] == 'Attack', 'percentage'].astype('int') - df_dist_ethos_all2_base.loc[df_dist_ethos_all2_base['cat base'] == 'Attack', 'percentage'].iloc[0]
    df_dist_ethos_all2.loc[df_dist_ethos_all2['cat base'] == 'Support', 'deviation'] = df_dist_ethos_all2.loc[df_dist_ethos_all2['cat base'] == 'Support', 'percentage'].astype('int') - df_dist_ethos_all2_base.loc[df_dist_ethos_all2_base['cat base'] == 'Support', 'percentage'].iloc[0]
    #st.write(df_dist_ethos_all2)

    #st.write(df_dist_ethos_all2_base)

    sns.set(font_scale=1.45, style='whitegrid')
    f_dist_ethos2 = sns.catplot(kind='bar', data = df_dist_ethos_all2.sort_values(by = ['cat', naming]),
                        height=5, aspect=1.3,
                    x = naming, y = 'percentage', hue = naming, dodge=False, legend = False,
                    palette = colors_log,
                    col = 'corpus',  )#col_wrap=1,sharex=False
    hatches = ['/', 'x', '.']


    f_dist_ethos2.map(plt.axhline, y=df_dist_ethos_all2_base.loc[df_dist_ethos_all2_base['cat base'] == 'Attack', 'percentage'].iloc[0], ls='--', c='red', alpha=0.75, linewidth=2, label = 'base attack')
    f_dist_ethos2.map(plt.axhline, y=df_dist_ethos_all2_base.loc[df_dist_ethos_all2_base['cat base'] == 'Support', 'percentage'].iloc[0], ls='--', c='green', alpha=0.75, linewidth=2, label = 'base support')

    plt.legend(loc = 'upper left', bbox_to_anchor = (0.85, 0.9), fontsize=13)

    if contents_radio_categories_val_units == 'percentage':
        f_dist_ethos2.set(ylim=(0, 101), xlabel='', ylabel = str(contents_radio_categories_val_units) )
    else:
        f_dist_ethos2.set( xlabel='', ylabel = str(contents_radio_categories_val_units),
        ylim=(0, df_dist_ethos_all2['percentage'].max()+201 ) )


    for ax in f_dist_ethos2.axes.ravel():
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')
    for axes in f_dist_ethos2.axes.flat:
        _ = axes.set_xticklabels(axes.get_xticklabels(), rotation=60)
    #plt.tight_layout(pad=1.5)

    sns.set(font_scale=1.1, style='whitegrid')
    f_dist_ethos2_dev = sns.catplot(kind='bar', data = df_dist_ethos_all2,
                    height=5, aspect=1.2,
                    x = naming, y = 'deviation', hue = naming, dodge=False, legend = False,
                    palette = colors_log, # colors_log,
                    col = 'corpus')
    hatches = ['/', 'x', '.']

    if contents_radio_categories_val_units == 'percentage':
        f_dist_ethos2_dev.set( xlabel='', ylabel = 'deviation',
        ylim = (df_dist_ethos_all2['deviation'].min()-16, df_dist_ethos_all2['deviation'].max()+16))
    else:
        f_dist_ethos2_dev.set( xlabel='', ylabel = str(contents_radio_categories_val_units),
        ylim = (df_dist_ethos_all2['deviation'].min()-116, df_dist_ethos_all2['deviation'].max()+116))

    f_dist_ethos2_dev.map(plt.axhline, y=0, ls='--', c='black', alpha=0.75, linewidth=1.6, label = 'baseline')
    plt.legend(loc = 'upper left', bbox_to_anchor = (0.72, 0.9), fontsize=13)

    for ax in f_dist_ethos2_dev.axes.ravel():
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')
    for axes in f_dist_ethos2_dev.axes.flat:
        _ = axes.set_xticklabels(axes.get_xticklabels(), rotation=60)



    bar_tab, pie_tab, table_tab, explore_tab = st.tabs( ['Bar-chart', 'Pie-chart', 'Tables', 'Cases' ] )

    with bar_tab:
        if contents_radio_categories == "3-LEP categories":
            add_spacelines(2)
            #fig, ax1 = plt.subplots(1, 1, figsize=(9, 5))
            #ax1.pie( np.array([47,30,23]), labels=[ 'logos', 'pathos', 'ethos'], startangle = 0, hatch = ['','\\', '/'],
            #colors = ['#FB598F', '#EBEB04','#DD8309', ], autopct='%.0f', )
            #st.pyplot(fig)
            st.pyplot(f_dist_ethos)
            add_spacelines(2)
            st.pyplot(f_dist_ethos_dev)
            add_spacelines(2)
            with st.expander('Baseline'):
                st.write( "**Baseline** is the average number/percentage of utterances that involve at least one category of logos, ethos or pathos rhetorical strategies " )

        elif contents_radio_categories == "6-LEP categories":
            add_spacelines(2)
            st.pyplot(f_dist_ethos2)
            add_spacelines(2)
            st.pyplot(f_dist_ethos2_dev)
            add_spacelines(2)
            with st.expander('Baseline'):
                st.write( """**Base attack** is the average number/percentage of utterances labelled as (ethotic, logotic) attacks or appeals to negative pathos. **Base support** is the average number/percentage of utterances labelled as (ethotic, logotic) support or appeals to positive pathos.
                """ )



        else:
            df_dist_ethos_all2 = up_data_dict_ethos[0].copy()
            if len(up_data_dict_ethos.keys()) > 1:
                for k in range(int(len(up_data_dict_ethos.keys()))-1):
                    k_sub2 = k+1
                    df_dist_ethos_all2 = pd.concat([df_dist_ethos_all2, up_data_dict_ethos[k_sub2]], axis=0, ignore_index=True)

            #st.dataframe(df_dist_ethos_all2)
            df_target_base = pd.concat(data_list, axis=0, ignore_index=True)
            df_target_base = df_target_base[df_target_base['kind'] == 'ethos']

            if contents_radio_targs == "direct ethos":
                targets_limit = df_target_base['Target'].dropna().unique()
                targets_limit = [t for t in targets_limit if "@" in t]
                targets_limit.append('nan')
                df_target_base = df_target_base[df_target_base.Target.isin(targets_limit)]


            elif contents_radio_targs == "3rd party ethos":
                targets_limit = df_target_base['Target'].dropna().unique()
                targets_limit = [t for t in targets_limit if not "@" in t]
                targets_limit.append('nan')
                df_target_base = df_target_base[df_target_base.Target.isin(targets_limit)]

            if not 'attack' in df_target_base['ethos_label'].unique():
                df_target_base['ethos_label'] = df_target_base['ethos_label'].map(ethos_mapping)

            if contents_radio_categories_val_units == 'percentage':
                df_target_base = pd.DataFrame( df_target_base[ df_target_base['ethos_label'].isin(['attack', 'support']) ].ethos_label.value_counts(normalize = True).round(2)*100)
            else:
                df_target_base = pd.DataFrame( df_target_base[ df_target_base['ethos_label'].isin(['attack', 'support']) ].ethos_label.value_counts(normalize = False) )

            df_target_base.columns = ['percentage']
            df_target_base.reset_index(inplace=True)
            df_target_base.columns = ['cat base', 'percentage']
            df_dist_ethos_all2_base = int( df_target_base['percentage'].mean().round(0) )

            naming_cats = set( ['Ethos Attack', 'Ethos Support', 'Logos Attack', 'Logos Support', 'Negative Pathos', 'Positive Pathos'] )

            df_dist_ethos_all2['cat'] = df_dist_ethos_all2[naming].map({
            'Ethos Attack':'Ethos', 'Ethos Support':'Ethos', 'Logos Attack':'Logos',
            'Logos Support':'Logos', 'Negative Pathos':'Pathos', 'Positive Pathos':'Pathos',
            'No Ethos':'Ethos', 'No Logos':'Logos', 'No Pathos':'Pathos',
            'Ethos':'Ethos', 'Pathos':'Pathos', 'Logos':'Logos',
            })
            df_dist_ethos_all2 = df_dist_ethos_all2.sort_values(by = [naming])

            df_dist_ethos_all2['cat base'] = df_dist_ethos_all2[naming].map({
            'Ethos Attack':'Attack', 'Ethos Support':'Support', 'Logos Attack':'Attack',
            'Logos Support':'Support', 'Negative Pathos':'Attack', 'Positive Pathos':'Support',
            }).fillna('Neutral')

            df_dist_ethos_all2['deviation'] = np.nan
            df_dist_ethos_all2['deviation'] = df_dist_ethos_all2['percentage'].astype('int') - df_dist_ethos_all2_base


            sns.set(font_scale=1.55, style='whitegrid')
            f_dist_ethos2 = sns.catplot(kind='bar', data = df_dist_ethos_all2,
                                height=4.5, aspect=1.2,
                                x = naming, y = 'percentage', hue = naming, dodge=False, legend = False,
                                palette = colors_log, col = 'corpus',
                                )
            #axes = f_dist_ethos2.axes.flatten()
            f_dist_ethos2.map(plt.axhline, y=df_dist_ethos_all2_base, ls='--', c='black', alpha=0.55, linewidth=1.7, label = 'baseline')
            plt.legend(loc = 'upper left', bbox_to_anchor = (0.68, 0.9), fontsize = 10)


            if contents_radio_categories_val_units == 'percentage':
                f_dist_ethos2.set(ylim=(0, 101), xlabel= contents_radio_targs, ylabel = contents_radio_categories_val_units)
            else:
                f_dist_ethos2.set( xlabel= contents_radio_targs, ylabel = contents_radio_categories_val_units,
                ylim=(0, df_dist_ethos_all2['percentage'].max()+101 ) )

            for ax in f_dist_ethos2.axes.ravel():
                for p in ax.patches:
                    ax.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')

            add_spacelines(2)
            plt.tight_layout(pad=1)
            st.pyplot(f_dist_ethos2)

            add_spacelines(2)
            sns.set(font_scale=1.2, style='whitegrid')

            f_dist_ethos2_dev = sns.catplot(kind='bar', data = df_dist_ethos_all2,
                                height=4.5, aspect=1.1,
                                x = naming, y = 'deviation', hue = naming, dodge=False, legend = False,
                                palette = colors_log, col = 'corpus',
                                )

            f_dist_ethos2_dev.set( xlabel= contents_radio_targs,
            ylim = ((df_dist_ethos_all2['deviation'].abs().max()*-1 ) - 26, df_dist_ethos_all2['deviation'].abs().max() + 26))


            f_dist_ethos2_dev.map(plt.axhline, y=0, ls='--', c='black', alpha=0.75, linewidth=1.6, label = 'baseline')
            plt.legend(loc = 'upper left', bbox_to_anchor = (0.68, 0.9), fontsize = 10)

            for ax in f_dist_ethos2_dev.axes.ravel():
                for p in ax.patches:
                    ax.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')
            st.pyplot(f_dist_ethos2_dev)



    with pie_tab:
        sns.set(font_scale=1.4, style='whitegrid')
        if contents_radio_categories == "3-LEP categories":
            add_spacelines(2)
            cor_col = 'corpus' in df_dist_ethos_all.columns

            if cor_col:
                corps = df_dist_ethos_all.corpus.unique()
                for corp in corps:
                    st.write(corp)
                    df_dist_ethos_all_corp = df_dist_ethos_all[df_dist_ethos_all.corpus == corp]
                    add_spacelines(1)
                    fig, ax= plt.subplots(1, 3, figsize=(12, 7))
                    fig.subplots_adjust(wspace=0.25)
                    #x1 = plt.pie( [36, 64], [ 'logos', 'neutral'] )
                    #x2 = plt.pie( [27, 73], [ 'ethos', 'neutral'] )
                    #x3 = plt.pie( [32, 68], [ 'pathos', 'neutral'] )
                    categories_rhet = df_dist_ethos_all_corp.cat.unique()
                    dcolr = {
                    'Ethos': '#FFE64A', 'No Ethos': '#8004ED', 'Logos': '#FFE64A', 'No Logos':  '#8004ED', 'No Pathos':  '#8004ED', 'Pathos': '#FFE64A',
                    }

                    for ii, categor in enumerate(categories_rhet):
                        labs = df_dist_ethos_all_corp[df_dist_ethos_all_corp.cat == categor].sort_values(by = 'category')['category'].values

                        ax[ii].pie( df_dist_ethos_all_corp[df_dist_ethos_all_corp.cat == categor].sort_values(by = 'category').percentage.values,
                        labels = labs,
                        startangle = 0, colors = [dcolr[key] for key in labs], autopct='%.0f', textprops={'color':'black', 'size':15}) # hatch = ['/','']
                        ax[ii].set_title(categor+ ' distribution')

                    st.pyplot(fig)
                    add_spacelines(1)

            else:
                fig, ax= plt.subplots(1, 3, figsize=(12, 7))
                fig.subplots_adjust(wspace=0.25)
                #x1 = plt.pie( [36, 64], [ 'logos', 'neutral'] )
                #x2 = plt.pie( [27, 73], [ 'ethos', 'neutral'] )
                #x3 = plt.pie( [32, 68], [ 'pathos', 'neutral'] )
                categories_rhet = df_dist_ethos_all.cat.unique()
                dcolr = {
                'Ethos': '#FFE64A', 'No Ethos': '#8004ED', 'Logos': '#FFE64A', 'No Logos':  '#8004ED', 'No Pathos':  '#8004ED', 'Pathos': '#FFE64A',
                }

                for ii, categor in enumerate(categories_rhet):
                    labs = df_dist_ethos_all[df_dist_ethos_all.cat == categor].sort_values(by = 'category')['category'].values
                    ax[ii].pie( df_dist_ethos_all[df_dist_ethos_all.cat == categor].sort_values(by = 'category').percentage.values,
                    labels = labs,
                    startangle = 0, colors = [dcolr[key] for key in labs], autopct='%.0f', textprops={'color':'black', 'size':15}) # hatch = ['/','']
                    ax[ii].set_title(categor+ ' distribution')
                st.pyplot(fig)


        elif contents_radio_categories == "6-LEP categories":
            add_spacelines(2)
            cor_col = 'corpus' in df_dist_ethos_all2.columns

            if cor_col:
                corps = df_dist_ethos_all2.corpus.unique()
                for corp in corps:
                    st.write(corp)
                    df_dist_ethos_all_corp = df_dist_ethos_all2[df_dist_ethos_all2.corpus == corp]
                    add_spacelines(1)

                    fig2, ax= plt.subplots(1, 3, figsize=(13, 9))
                    fig2.subplots_adjust(wspace=0.53)

                    categories_rhet = df_dist_ethos_all_corp.cat.unique()

                    for ii, categor in enumerate(categories_rhet):
                        ax[ii].pie( df_dist_ethos_all_corp[df_dist_ethos_all_corp.cat == categor].sort_values(by = 'category').percentage.values,
                        labels = df_dist_ethos_all_corp[df_dist_ethos_all_corp.cat == categor].sort_values(by = 'category')['category'].values,
                        startangle = (ii*11) + 30, colors = ['#ED1818', '#4FFF3A', ], autopct='%.0f',
                        textprops={'color':'black', 'size':15}) # hatch = ['/','']
                        ax[ii].set_title(categor+ ' distribution')
                    st.pyplot(fig2)
                    add_spacelines(1)

            else:
                fig2, ax= plt.subplots(1, 3, figsize=(13, 9))
                fig2.subplots_adjust(wspace=0.53)

                categories_rhet = df_dist_ethos_all2.cat.unique()
                for ii, categor in enumerate(categories_rhet):
                    ax[ii].pie( df_dist_ethos_all2[df_dist_ethos_all2.cat == categor].sort_values(by = 'category').percentage.values,
                    labels = df_dist_ethos_all2[df_dist_ethos_all2.cat == categor].sort_values(by = 'category')['category'].values, hatch = ['/','', ],
                    startangle = (ii*11) + 30, colors = ['#ED1818', '#4FFF3A', ], autopct='%.0f', textprops={'color':'black', 'size':15}) # hatch = ['/','']
                    ax[ii].set_title(categor+ ' distribution')
                st.pyplot(fig2)


        else:

            #st.dataframe(df_dist_ethos_all2)
            naming_cats = set( ['Ethos Attack', 'Ethos Support', 'Logos Attack', 'Logos Support', 'Negative Pathos', 'Positive Pathos'] )
            add_spacelines(2)

            cor_col = 'corpus' in df_dist_ethos_all2.columns
            if cor_col:
                corps = df_dist_ethos_all2.corpus.unique()
                for corp in corps:
                    st.write(corp)
                    df_dist_ethos_all_corp = df_dist_ethos_all2[df_dist_ethos_all2.corpus == corp]
                    add_spacelines(1)
                    fig2, ax= plt.subplots(1, 1, figsize=(10, 6))
                    #x1 = plt.pie( [36, 64], [ 'logos', 'neutral'] )
                    #x2 = plt.pie( [27, 73], [ 'ethos', 'neutral'] )
                    #x3 = plt.pie( [32, 68], [ 'pathos', 'neutral'] )
                    categor = 'Ethos'
                    ax.pie( df_dist_ethos_all_corp.sort_values(by = 'category').percentage.values,
                            labels = df_dist_ethos_all_corp.sort_values(by = 'category')['category'].values, hatch = ['/','', ],
                            startangle = 30, colors = ['#ED1818', '#4FFF3A', ], autopct='%.0f', textprops={'color':'black', 'size':15 }) # hatch = ['/','']
                    ax.set_title(f' {contents_radio_targs.capitalize()} distribution')# categor+' ' +' distribution'
                    st.pyplot(fig2)

            else:
                fig2, ax= plt.subplots(1, 1, figsize=(10, 6))
                categor = 'Ethos'
                ax.pie( df_dist_ethos_all2.sort_values(by = 'category').percentage.values,
                        labels = df_dist_ethos_all2.sort_values(by = 'category')['category'].values, hatch = ['/','', ],
                        startangle = 30, colors = ['#ED1818', '#4FFF3A', ], autopct='%.0f', textprops={'color':'black', 'size':15 }) # hatch = ['/','']
                ax.set_title(f' {contents_radio_targs.capitalize()} distribution')# categor+' ' +' distribution'
                st.pyplot(fig2)



    with table_tab:
        if contents_radio_categories == "3-LEP categories":
            st.write(df_dist_ethos_all.reset_index(drop=True).rename(columns = {'percentage':contents_radio_categories_val_units}).loc[:, ['category', contents_radio_categories_val_units, 'deviation', 'corpus'] ])


        elif contents_radio_categories == "6-LEP categories":
            st.write(df_dist_ethos_all2.reset_index(drop=True).rename(columns = {'percentage':contents_radio_categories_val_units}).loc[:, ['category', contents_radio_categories_val_units, 'deviation', 'corpus'] ])

        else:
            st.write(df_dist_ethos_all2.reset_index(drop=True).rename(columns = {'percentage':contents_radio_categories_val_units}).loc[:, ['category', contents_radio_categories_val_units, 'deviation', 'corpus'] ])




    with explore_tab:
        st.write('### Cases')

        df = pd.concat( data_list, axis=0, ignore_index=True )
        if "&" in df.corpus.iloc[0]:
            ds = "Covid & ElectionsSM"
            df['corpus'] = ds
        #st.write(df)
        #st.write(df.info())
        df.loc[ ~(df.locution_conclusion.isna()) , 'locution'] = df.loc[~(df.locution_conclusion.isna()) ].locution_conclusion.astype('str') + " - " + df.loc[~(df.locution_conclusion.isna()) ].locution_premise.astype('str')
        df.loc[~(df.locution_conclusion.isna()), 'content'] = df.loc[~(df.locution_conclusion.isna()) ].conclusion.astype('str') + " - " + df.loc[ ~(df.locution_conclusion.isna()) ].premise.astype('str')
        df.loc[~(df.sentence.isna()), 'locution'] = df.loc[ ~(df.sentence.isna()) ].sentence.astype('str')
        #df.loc[(df.nodeset_id.isna()), 'nodeset_id'] = df.loc[(df.nodeset_id.isna()) ].map_ID.astype('str')
        df.loc[~(df.locution_conclusion.isna()), 'source'] = df.loc[~(df.locution_conclusion.isna()) ].speaker_premise.astype('str')

        # ['sentence', 'source', '(anti)-hero label', 'ethos_label', 'sentiment', 'emotion', 'Target',]# , 'conversation_id','date'
        #st.write(df.columns)' in
        if 'ethos_label' in df.columns:
            if not 'attack' in df['ethos_label'].unique():
                df['ethos_name'] = df['ethos_label'].map(ethos_mapping)
        if not 'positive' in df['pathos_label'].unique():
                df['pathos_name'] = df['pathos_label'].map(valence_mapping)
        dff_columns2 =  '''
            locution
            connection
            source
            Target
            corpus
            argument_linked
            nodeset_id
            illocution
            pathos_name
            ethos_name'''.split('\n')

        dff_columns2 = [ ccol.strip() for ccol in dff_columns2 ]
        dff_columns =  list( set(df.columns).intersection( set(dff_columns2) ) )


        df = df.dropna(how='all', axis=1)
        df[df.columns] = df[df.columns].astype('str')
        dff = df[dff_columns].copy()
        select_columns = st.multiselect("Choose columns for specifying conditions", dff_columns, dff_columns[0])
        cols_columns = st.columns(len(select_columns))
        dict_cond = {}
        for n, c in enumerate(cols_columns):
            with c:
                cond_col = st.multiselect(f"Choose condition for *{select_columns[n]}*",
                                       (dff[select_columns[n]].unique()), (dff[select_columns[n]].unique()[-1]))
                dict_cond[select_columns[n]] = cond_col
        dff_selected = dff.copy()
        dff_selected = dff_selected.drop_duplicates(subset = ['locution'] )
        for i, k in enumerate(dict_cond.keys()):
            dff_selected = dff_selected[ dff_selected[str(k)].isin(dict_cond[k]) ]
        add_spacelines(2)
        st.dataframe(dff_selected[dff_columns].sort_values(by = select_columns).reset_index(drop=True).dropna(how='all', axis=1), width = None)
        st.write(f"No. of cases: {len(dff_selected)}.")


colors = {'joy' : '#8DF903', 'anger' : '#FD7E00', 'sadness' : '#CA00B9',
          'fear' : '#000000', 'disgust' :'#840079', 'no sentiment' : '#2002B5','surprise' : '#E1CA01',
          'positive':'#097604', 'negative':'#9B0101', 'neutral':'#2002B5',
          'contains_emotion':'#B10156', 'no_emotion':'#2002B5',
          'support':'#097604', 'attack':'#9B0101'}



def distribution_plot_compareX_sub_cross(data_list0, an_unit, dim1, dim2):
    if 'label' in dim1:
        dim10 = dim1.split("_")[0]
    else:
        dim10 = dim1

    if 'label' in dim2:
        dim20 = dim2.split("_")[0]
    else:
        dim20 = dim2

    if len(data_list0) == 1:
        add_spacelines(1)
        up_data_dict = {}
        up_data_dict2 = {}
        n = 0
        for data in data_list0:
            df = data.copy()
            ds = df['corpus'].iloc[0]

            if not 'attack' in df['ethos_label'].unique():
                df['ethos_label'] = df['ethos_label'].map(ethos_mapping)
            if not 'positive' in df['pathos_label'].unique():
                df['pathos_label'] = df['pathos_label'].map(valence_mapping)

            if an_unit == 'number':
                col_unit = 'number'
                df_dist_ethos = pd.DataFrame(df.groupby([dim1])[dim2].value_counts())

            else:
                col_unit = 'percentage'
                df_dist_ethos = pd.DataFrame(df.groupby([dim1])[dim2].value_counts(normalize = True).round(2)*100)


            df_dist_ethos.columns = [col_unit]
            df_dist_ethos.reset_index(inplace=True)
            df_dist_ethos.columns = [dim10, dim20, col_unit]
            df_dist_ethos = df_dist_ethos.sort_values(by = dim10)
            up_data_dict[n] = df_dist_ethos
            df_dist_ethos['corpora'] = ds

            n += 1

        df_dist_ethos_all = up_data_dict[0].copy()
        for k in range(int(len(up_data_dict.keys()))-1):
            k_sub = k+1
            df_dist_ethos_all = pd.concat([df_dist_ethos_all, up_data_dict[k_sub]], axis=0, ignore_index=True)

        sns.set(font_scale=1.2, style='whitegrid')
        maxval = df_dist_ethos_all[col_unit].max()
        fg1=sns.catplot(kind='bar', data=df_dist_ethos_all, y = dim10, x = col_unit,
                        hue=dim20, dodge=True, palette = colors)
        if col_unit == 'percentage':
            plt.xlim(0, 100)
            plt.xticks(np.arange(0, 100, 10))
        else:
            plt.xlim(0, maxval+111)
            plt.xticks(np.arange(0, maxval+111, 100))
        plt.title(f'{dim10.capitalize()} x {dim20.capitalize()}')

        fg2=sns.catplot(kind='bar', data=df_dist_ethos_all, y = dim20, x = col_unit,
                        hue=dim10, dodge=True, palette = colors)
        if col_unit == 'percentage':
            plt.xlim(0, 100)
            plt.xticks(np.arange(0, 100, 10))
        else:
            plt.xlim(0, maxval+111)
            plt.xticks(np.arange(0, maxval+111, 100))
        plt.title(f'{dim20.capitalize()} x {dim10.capitalize()}')

        return fg1

    else:
        st.info("Function not supported for multiple corpora comparison.")




def distribution_plot_compareX_sub_single(data_list0, an_unit, dim1):
    if 'label' in dim1:
        dim0 = dim1.split("_")[0]
    else:
        dim0 = dim1

    up_data_dict = {}
    up_data_dict2 = {}
    n = 0
    for data in data_list0:
        df = data.copy()
        ds = str(df['corpus'].iloc[0])
        #st.dataframe(df)
        if not 'attack' in df['ethos_label'].unique():
            df['ethos_label'] = df['ethos_label'].map(ethos_mapping)
        if not 'positive' in df['pathos_label'].unique():
            df['pathos_label'] = df['pathos_label'].map(valence_mapping)

        if an_unit == 'number':
            col_unit = 'number'
            df_dist_ethos = pd.DataFrame(df[dim1].value_counts())
            df_dist_ethos2 = pd.DataFrame(df[df[dim1] != 'neutral'][dim1].value_counts())
        else:
            col_unit = 'percentage'
            df_dist_ethos = pd.DataFrame(df[dim1].value_counts(normalize = True).round(2)*100)
            df_dist_ethos2 = pd.DataFrame(df[df[dim1] != 'neutral'][dim1].value_counts(normalize = True).round(2)*100)

        df_dist_ethos.columns = [col_unit]
        df_dist_ethos.reset_index(inplace=True)
        df_dist_ethos.columns = [dim0, col_unit]
        df_dist_ethos = df_dist_ethos.sort_values(by = dim0)

        df_dist_ethos['corpora'] = ds
        up_data_dict[n] = df_dist_ethos

        df_dist_ethos2.columns = [col_unit]
        df_dist_ethos2.reset_index(inplace=True)
        df_dist_ethos2.columns = [dim0, col_unit]
        df_dist_ethos2 = df_dist_ethos2.sort_values(by = dim0)
        df_dist_ethos2['corpora'] = ds
        up_data_dict2[n] = df_dist_ethos2

        n += 1

    df_dist_ethos_all = up_data_dict[0].copy()
    for k in range(int(len(up_data_dict.keys()))-1):
        k_sub = k+1
        df_dist_ethos_all = pd.concat([df_dist_ethos_all, up_data_dict[k_sub]], axis=0, ignore_index=True)

    neu = []
    eth = []
    corp = []
    for cor in df_dist_ethos_all.corpora.unique():
        corp.append(cor)
        nn = df_dist_ethos_all[(df_dist_ethos_all.corpora == cor) & (df_dist_ethos_all[dim0] == 'neutral')][col_unit].iloc[0]
        neu.append(nn)
        eth.append(f'no {dim0}')
        nn2 = df_dist_ethos_all[(df_dist_ethos_all.corpora == cor)][col_unit].sum() - nn
        neu.append(nn2)
        eth.append(dim0)
        corp.append(cor)
    #st.dataframe(df_dist_ethos_all)
    #st.stop()
    df_dist_ethos_all0 = pd.DataFrame({dim0 : eth, col_unit:neu, 'corpora':corp})

    sns.set(font_scale=1.6, style='whitegrid')
    f_dist_ethos0 = sns.catplot(kind='bar', data = df_dist_ethos_all0, height=5.4, aspect=1.2,
                    x = dim0, y = col_unit, hue = dim0, dodge=False, legend = False,
                    palette = {dim0:'#EA9200', f'no {dim0}':'#022D96'},
                    col = 'corpora')
    if an_unit != 'number':
        f_dist_ethos0.set(ylim=(0, 110))
    f_dist_ethos0.set(xlabel="", )
    #plt.title(f"Ethos distribution")
    for ax in f_dist_ethos0.axes.ravel():
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')
    sns.set(font_scale=1.4, style='whitegrid')
    f_dist_ethos = sns.catplot(kind='bar', data = df_dist_ethos_all, height=5.4, aspect=1.2,
                    x = dim0, y = col_unit, hue = dim0, dodge=False, legend = False,
                    palette = {'attack':'#BB0000', 'neutral':'#022D96', 'support':'#026F00',
                                'negative':'#BB0000', 'positive':'#026F00'},
                    col = 'corpora')
    #plt.title(f"Ethos distribution")
    if an_unit != 'number':
        f_dist_ethos.set(ylim=(0, 110))
    for ax in f_dist_ethos.axes.ravel():
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()),
            ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')

    df_dist_ethos_all2 = up_data_dict2[0].copy()
    for k in range(int(len(up_data_dict2.keys()))-1):
        k_sub2 = k+1
        df_dist_ethos_all2 = pd.concat([df_dist_ethos_all2, up_data_dict2[k_sub2]], axis=0, ignore_index=True)

    f_dist_ethos2 = sns.catplot(kind='bar', data = df_dist_ethos_all2, height=5.4, aspect=1.2,
                    x = dim0, y = col_unit, hue = dim0, dodge=False, legend = False,
                    palette = {'attack':'#BB0000', 'neutral':'#022D96', 'support':'#026F00',
                                'negative':'#BB0000', 'positive':'#026F00'},
                    col = 'corpora')
    #plt.title(f"Ethos distribution")
    if an_unit != 'number':
        f_dist_ethos2.set(ylim=(0, 110))
    for ax in f_dist_ethos2.axes.ravel():
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')

    return f_dist_ethos0, f_dist_ethos, f_dist_ethos2



def distribution_plot_compareX(data_list):
    st.write("### Distribution")
    add_spacelines(2)
    contents_radio_unit = st.radio("Unit of analysis", ("percentage", "number"))


    add_spacelines(1)
    c1, c2, c3, c4, c5, c6 = st.tabs(['Ethos', 'Pathos', 'Sentiment',
            "Ethos x Pathos", "Ethos x Sentiment", "Pathos x Sentiment"])
    with c1:
        f_dist_0, f_dist_1, f_dist_2 = distribution_plot_compareX_sub_single(data_list0 = data_list,
                                    an_unit = contents_radio_unit, dim1 = 'ethos_label')
        add_spacelines(1)
        st.pyplot(f_dist_0)
        add_spacelines(1)
        st.pyplot(f_dist_1)
        add_spacelines(1)
        st.pyplot(f_dist_2)
        add_spacelines(1)

    with c2:
        f_dist_0, f_dist_1, f_dist_2 = distribution_plot_compareX_sub_single(data_list0 = data_list,
                                    an_unit = contents_radio_unit, dim1 = 'pathos_label')
        add_spacelines(1)
        st.pyplot(f_dist_0)
        add_spacelines(1)
        st.pyplot(f_dist_1)
        add_spacelines(1)
        st.pyplot(f_dist_2)
        add_spacelines(1)

    with c3:
        f_dist_0, f_dist_1, f_dist_2 = distribution_plot_compareX_sub_single(data_list0 = data_list,
                                    an_unit = contents_radio_unit, dim1 = 'sentiment')
        add_spacelines(1)
        st.pyplot(f_dist_0)
        add_spacelines(1)
        st.pyplot(f_dist_1)
        add_spacelines(1)
        st.pyplot(f_dist_2)
        add_spacelines(1)


    with c4:
        if len(data_list) == 1:
            fg1x = distribution_plot_compareX_sub_cross(data_list0 = data_list,
                        an_unit = contents_radio_unit, dim1 = 'ethos_label', dim2 = 'pathos_label')
            fg2x = distribution_plot_compareX_sub_cross(data_list0 = data_list,
                        an_unit = contents_radio_unit, dim1 = 'pathos_label', dim2 = 'ethos_label')
            ff1, ff2, = st.columns(2)
            with ff1:
                st.pyplot(fg1x)
            with ff2:
                st.pyplot(fg2x)

        else:
            add_spacelines(2)
            st.info("Function not supported for multiple corpora comparison.")

    with c5:
        if len(data_list) == 1:
            fg1x = distribution_plot_compareX_sub_cross(data_list0 = data_list,
                        an_unit = contents_radio_unit, dim1 = 'ethos_label', dim2 = 'sentiment')
            fg2x = distribution_plot_compareX_sub_cross(data_list0 = data_list,
                        an_unit = contents_radio_unit, dim1 = 'sentiment', dim2 = 'ethos_label')
            ff1, ff2, = st.columns(2)
            with ff1:
                st.pyplot(fg1x)
            with ff2:
                st.pyplot(fg2x)

        else:
            add_spacelines(2)
            st.info("Function not supported for multiple corpora comparison.")

    with c6:
        if len(data_list) == 1:
            fg1x = distribution_plot_compareX_sub_cross(data_list0 = data_list,
                        an_unit = contents_radio_unit, dim2 = 'sentiment', dim1 = 'pathos_label')
            fg2x = distribution_plot_compareX_sub_cross(data_list0 = data_list,
                        an_unit = contents_radio_unit, dim1 = 'pathos_label', dim2 = 'ethos_label')
            ff1, ff2, = st.columns(2)
            with ff1:
                st.pyplot(fg1x)
            with ff2:
                st.pyplot(fg2x)

        else:
            add_spacelines(2)
            st.info("Function not supported for multiple corpora comparison.")




def distribution_plot_compare(data_list):
    contents_radio_targs = st.radio("Category of the target of ethotic statements", ("both", "direct ethos", "3rd party ethos"))

    up_data_dict = {}
    up_data_dict2 = {}
    n = 0
    for data in data_list:
        df = data.copy()
        ds = df['corpus'].iloc[0]
        #st.dataframe(df)
        if not 'attack' in df['ethos_label'].unique():
            df['ethos_label'] = df['ethos_label'].map(ethos_mapping)
        df["Target"] = df["Target"].astype('str')
        df["Target"] = df["Target"].str.replace('Government', 'government')

        if contents_radio_targs == "direct ethos":
            targets_limit = df['Target'].dropna().unique()
            targets_limit = [t for t in targets_limit if "@" in t]
            targets_limit.append('nan')
            df = df[df.Target.isin(targets_limit)]
            if len(targets_limit) < 1:
                st.error(f'No cases of **{contents_radio_targs}** found in the chosen corpora.')
                st.stop()
        elif contents_radio_targs == "3rd party ethos":
            targets_limit = df['Target'].dropna().unique()
            targets_limit = [t for t in targets_limit if not "@" in t]
            targets_limit.append('nan')
            df = df[df.Target.isin(targets_limit)]
            if len(targets_limit) < 1:
                st.error(f'No cases of **{contents_radio_targs}** found in the chosen corpora.')
                st.stop()

        df_dist_ethos = pd.DataFrame(df['ethos_label'].value_counts(normalize = True).round(2)*100)
        df_dist_ethos.columns = ['percentage']
        df_dist_ethos.reset_index(inplace=True)
        #st.dataframe(df_dist_ethos)
        df_dist_ethos.columns = ['ethos', 'percentage']
        df_dist_ethos = df_dist_ethos.sort_values(by = 'ethos')
        df_dist_ethos['corpus'] = ds
        up_data_dict[n] = df_dist_ethos

        df_dist_ethos2 = pd.DataFrame(df[df['ethos_label'] != 'neutral']['ethos_label'].value_counts(normalize = True).round(2)*100)
        df_dist_ethos2.columns = ['percentage']
        df_dist_ethos2.reset_index(inplace=True)
        df_dist_ethos2.columns = ['ethos', 'percentage']
        df_dist_ethos2 = df_dist_ethos2.sort_values(by = 'ethos')
        df_dist_ethos2['corpus'] = ds
        up_data_dict2[n] = df_dist_ethos2

        n += 1

    df_dist_ethos_all = up_data_dict[0].copy()
    for k in range(int(len(up_data_dict.keys()))-1):
        k_sub = k+1
        df_dist_ethos_all = pd.concat([df_dist_ethos_all, up_data_dict[k_sub]], axis=0, ignore_index=True)

    neu = []
    eth = []
    corp = []
    for cor in df_dist_ethos_all.corpus.unique():
        corp.append(cor)
        nn = df_dist_ethos_all[(df_dist_ethos_all.corpus == cor) & (df_dist_ethos_all['ethos'] == 'neutral')]['percentage'].iloc[0]
        neu.append(nn)
        eth.append('no ethos')
        neu.append(100 - nn)
        eth.append('ethos')
        corp.append(cor)
    df_dist_ethos_all0 = pd.DataFrame({'ethos':eth, 'percentage':neu, 'corpus':corp})

    sns.set(font_scale=1.55, style='whitegrid')
    f_dist_ethos0 = sns.catplot(kind='bar', data = df_dist_ethos_all0, height=4.5, aspect=1.4,
                    x = 'ethos', y = 'percentage', hue = 'ethos', dodge=False,
                    palette = {'ethos':'#EA9200', 'no ethos':'#022D96'},
                    col = 'corpus')
    f_dist_ethos0.set(ylim=(0, 110))
    f_dist_ethos0.set(xlabel="")
    for ax in f_dist_ethos0.axes.ravel():
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')
    sns.set(font_scale=1.25, style='whitegrid')
    f_dist_ethos = sns.catplot(kind='bar', data = df_dist_ethos_all, height=4.5, aspect=1.4,
                    x = 'ethos', y = 'percentage', hue = 'ethos', dodge=False,
                    palette = {'attack':'#BB0000', 'neutral':'#022D96', 'support':'#026F00'},
                    col = 'corpus')
    f_dist_ethos.set(ylim=(0, 110))
    for ax in f_dist_ethos.axes.ravel():
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()),
            ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')

    df_dist_ethos_all2 = up_data_dict2[0].copy()
    for k in range(int(len(up_data_dict2.keys()))-1):
        k_sub2 = k+1
        df_dist_ethos_all2 = pd.concat([df_dist_ethos_all2, up_data_dict2[k_sub2]], axis=0, ignore_index=True)

    #st.dataframe(df_dist_ethos_all2)
    f_dist_ethos2 = sns.catplot(kind='bar', data = df_dist_ethos_all2, height=4.5, aspect=1.15,
                    x = 'ethos', y = 'percentage', hue = 'ethos', dodge=False, legend=False,
                    palette = {'attack':'#BB0000', 'support':'#026F00', 'neutral':'#022D96'},
                    col = 'corpus')

    f_dist_ethos2.set(ylim=(0, 110))
    for ax in f_dist_ethos2.axes.ravel():
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')

    c1, c2, c3 = st.columns([1, 8, 1], gap='small')
    with c2:
        add_spacelines(1)
        st.pyplot(f_dist_ethos0)
        add_spacelines(1)
        st.pyplot(f_dist_ethos)
        add_spacelines(1)
        st.pyplot(f_dist_ethos2)
    add_spacelines(1)


    add_spacelines(1)
    with st.expander("Pathos distribution"):
        add_spacelines(1)
        up_data_dict = {}
        up_data_dict2 = {}
        n = 0
        for data in data_list:
            df = data.copy()
            ds = df['corpus'].iloc[0]

            if not 'neutral' in df['pathos_label'].unique():
                df['pathos_label'] = df['pathos_label'].map(valence_mapping)

            if not 'attack' in df['ethos_label'].unique():
                df['ethos_label'] = df['ethos_label'].fillna(0)
                df['ethos_label'] = df['ethos_label'].map(ethos_mapping)
                df = df.dropna(subset = ['ethos_label'])
            df['ethos_label'] = df['ethos_label'].fillna('neutral')
            df["Target"] = df["Target"].astype('str')
            df["Target"] = df["Target"].str.replace('Government', 'government')

            if contents_radio_targs == "direct ethos":
                targets_limit = df['Target'].dropna().unique()
                targets_limit = [t for t in targets_limit if "@" in t]
                targets_limit.append('nan')
                df = df[df.Target.isin(targets_limit)]
                if len(targets_limit) < 1:
                    st.error(f'No cases of **{contents_radio_targs}** found in the chosen corpora.')
                    st.stop()
            elif contents_radio_targs == "3rd party ethos":
                targets_limit = df['Target'].dropna().unique()
                targets_limit = [t for t in targets_limit if not "@" in t]
                targets_limit.append('nan')
                df = df[df.Target.isin(targets_limit)]

            df = clean_text(df, 'sentence_lemmatized', text_column_name = "sentence_lemmatized")

            df_dist_ethos = pd.DataFrame(df['pathos_label'].value_counts(normalize = True).round(2)*100)
            df_dist_ethos.columns = ['percentage']
            df_dist_ethos.reset_index(inplace=True)
            df_dist_ethos.columns = ['pathos', 'percentage']
            df_dist_ethos = df_dist_ethos.sort_values(by = 'pathos')
            up_data_dict[n] = df_dist_ethos
            df_dist_ethos['corpus'] = ds
            #st.dataframe(df_dist_ethos)

            df_dist_ethos2 = pd.DataFrame(df[df['pathos_label'] != 'neutral']['pathos_label'].value_counts(normalize = True).round(2)*100)
            df_dist_ethos2.columns = ['percentage']
            df_dist_ethos2.reset_index(inplace=True)
            df_dist_ethos2.columns = ['pathos', 'percentage']
            df_dist_ethos2['corpus'] = ds
            df_dist_ethos2 = df_dist_ethos2.sort_values(by = 'pathos')
            up_data_dict2[n] = df_dist_ethos2
            #st.dataframe(df_dist_ethos2)

            n += 1

        df_dist_ethos_all = up_data_dict[0].copy()
        for k in range(int(len(up_data_dict.keys()))-1):
            k_sub = k+1
            df_dist_ethos_all = pd.concat([df_dist_ethos_all, up_data_dict[k_sub]], axis=0, ignore_index=True)

        neu = []
        eth = []
        corp = []
        for cor in df_dist_ethos_all.corpus.unique():
            corp.append(cor)
            nn = df_dist_ethos_all[(df_dist_ethos_all.corpus == cor) & (df_dist_ethos_all['pathos'] == 'neutral')]['percentage'].iloc[0]
            neu.append(nn)
            eth.append('no pathos')
            neu.append(100 - nn)
            eth.append('pathos')
            corp.append(cor)
        df_dist_ethos_all0 = pd.DataFrame({'pathos':eth, 'percentage':neu, 'corpus':corp})

        sns.set(font_scale=1.4, style='whitegrid')
        f_dist_ethos0 = sns.catplot(kind='bar', data = df_dist_ethos_all0, height=4.5, aspect=1.4,
                        x = 'pathos', y = 'percentage', hue = 'pathos', dodge=False,
                        palette = {'pathos':'#EA9200', 'no pathos':'#022D96'},
                        col = 'corpus')
        f_dist_ethos0.set(ylim=(0, 110))
        f_dist_ethos0.set(xlabel="")
        for ax in f_dist_ethos0.axes.ravel():
            for p in ax.patches:
                ax.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')

        f_dist_ethos = sns.catplot(kind='bar', data = df_dist_ethos_all, height=4.5, aspect=1.4,
                        x = 'pathos', y = 'percentage', hue = 'pathos', dodge=False,
                        palette = {'negative':'#BB0000', 'neutral':'#022D96', 'positive':'#026F00'},
                        col = 'corpus')
        f_dist_ethos.set(ylim=(0, 110))
        for ax in f_dist_ethos.axes.ravel():
            for p in ax.patches:
                ax.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')

        df_dist_ethos_all2 = up_data_dict2[0].copy()
        for k in range(int(len(up_data_dict2.keys()))-1):
            k_sub2 = k+1
            df_dist_ethos_all2 = pd.concat([df_dist_ethos_all2, up_data_dict2[k_sub2]], axis=0, ignore_index=True)

        f_dist_ethos2 = sns.catplot(kind='bar', data = df_dist_ethos_all2, height=4.5, aspect=0.83,
                        x = 'pathos', y = 'percentage', hue = 'pathos', dodge=False,legend=False,
                        palette = {'negative':'#BB0000', 'positive':'#026F00', 'neutral':'#022D96'},
                        col = 'corpus')
        f_dist_ethos2.set(ylim=(0, 110))
        for ax in f_dist_ethos2.axes.ravel():
            for p in ax.patches:
                ax.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')

        cc1, cc2, cc3 = st.columns([1, 8, 1], gap='small')
        with cc2:
            add_spacelines(1)
            st.pyplot(f_dist_ethos0)
            add_spacelines(1)
            st.pyplot(f_dist_ethos)
            add_spacelines(1)
            st.pyplot(f_dist_ethos2)
        add_spacelines(1)






import time

##################### page config  #####################
st.set_page_config(page_title="LEPAn", layout="centered") # centered wide

import re


summary_corpora_list = []
summary_corpora_list_raw = {}
summary_corpora_dict_raw = {}
summary_corpora_list_raw_len = []

#  *********************** sidebar  *********************
with st.sidebar:
    st.title("Contents")

    contents_radio_rhetoric_category_logos = True
    contents_radio_rhetoric_category_ethos = False
    contents_radio_type = st.radio("", ('Home Page', 'Single Corpus Analysis', 'Comparative Corpora Analysis'), ) # label_visibility='collapsed'

    if contents_radio_type == 'Home Page':
        add_spacelines(2)


    elif contents_radio_type == 'Comparative Corpora Analysis':
        add_spacelines(2)
        st.write('Choose corpora')
        box_pol1_log = st.checkbox("Covid", value=True)
        box_pol5_log = st.checkbox("ElectionsSM", value=True)

        corpora_list = []
        corpora_list_et = {}
        corpora_list_log = {}
        add_spacelines(1)

        st.write( " ********************************** " )

        if not np.sum([box_pol1_log, box_pol5_log ]) > 1:
            st.error('Choose at least 2 corpora')
            st.stop()


        if box_pol1_log:
                cor11 = load_data(vac_red)
                cor1 = cor11.copy()
                cor1_src = cor1['source'].unique()
                cor1['conversation_id'] = 0
                cor1_src = [str(s).replace('@', '') for s in cor1_src]
                cor1['Target'] = cor1['Target'].astype('str')
                cor1['source'] = cor1['source'].astype('str').apply(lambda x: ["@"+str(x) if not "@" in x else x][0])
                cor1['Target'] = cor1['Target'].apply(lambda x: ["@"+str(x) if (not "@" in x and x in cor1_src) else x][0])
                cor1['corpus'] = "Covid"
                cor1['kind'] = "ethos"
                cor1.loc[cor1["Target"] == "@CNN", 'Target'] = 'CNN'
                cor1.loc[cor1["Target"] == "the unvaccinated", 'Target'] = 'unvaccinated'
                cor1.loc[cor1["Target"] == "the vaccinated", 'Target'] = 'vaccinated'
                corpora_list.append(cor1)
                summary_corpora_list.append(cor1)

                cor11 = load_data(vac_red_log, indx=False)

                cor11_2 = load_data(vac_red_log2, indx=False)
                cor11_2 = cor11_2.drop_duplicates(subset = ['locution'])
                d3t = cor11_2.locution.unique()
                d2tp = cor11.locution_premise.unique()
                d2tc = cor11.locution_conclusion.unique()
                d2t = set(d2tc).union(d2tp)
                d2t_neu = set(d3t).difference(d2t)
                cor11_2 = cor11_2[cor11_2.locution.isin(d2t_neu)]
                cor11_2['connection'] = 'neutral'
                cor1 = cor11.copy()
                cor1['premise'] =  cor1.groupby(['id_connection'])[['premise']].transform(lambda x: " ".join(x) )
                cor1['locution_premise'] =  cor1.groupby(['id_connection'])[['locution_premise']].transform(lambda x: " ".join(x) )
                cor1 = cor1.drop_duplicates('id_connection')
                cor1 = pd.concat([cor1, cor11_2 ], axis = 0, ignore_index = True)
                cor1['kind'] = "logos"
                cor1['corpus'] = "Covid"
                corpora_list.append(cor1)
                summary_corpora_list.append(cor1)
                corpora_list_log[cor1['corpus'].iloc[0]] = cor1


        if box_pol5_log:
                cor55 = load_data(us16)
                cor5 = cor55.copy()
                cor5['Target'] = cor5['Target'].astype('str')
                cor5['corpus'] = "ElectionsSM"
                cor5['source'] = cor5['source'].astype('str')
                cor5['kind'] = "ethos"
                corpora_list.append(cor5)
                summary_corpora_list.append(cor5)

                cor55 = load_data(us16_log, indx=False)
                cor5 = cor55.copy()

                cor11_2d = pd.read_excel(us16_log2, sheet_name = 'dr1')
                cor11_2g = pd.read_excel(us16_log2, sheet_name = 'gr1')
                cor11_2r = pd.read_excel(us16_log2, sheet_name = 'rr1')
                cor11_2 = pd.concat( [cor11_2d, cor11_2g, cor11_2r], axis = 0, ignore_index = True )

                cor11_2 = cor11_2.drop_duplicates(subset = ['locution'])
                d3t = cor11_2.locution.unique()
                d2tp = cor5.locution_premise.unique()
                d2tc = cor5.locution_conclusion.unique()
                d2t = set(d2tc).union(d2tp)
                d2t_neu = set(d3t).difference(d2t)
                cor11_2 = cor11_2[cor11_2.locution.isin(d2t_neu)]
                cor11_2['connection'] = 'neutral'

                cor5['premise'] =  cor5.groupby(['id_connection'])[['premise']].transform(lambda x: " ".join(x) )
                cor5['locution_premise'] =  cor5.groupby(['id_connection'])[['locution_premise']].transform(lambda x: " ".join(x) )
                cor5 = cor5.drop_duplicates('id_connection')
                cor1 = pd.concat([cor5, cor11_2 ], axis = 0, ignore_index = True)
                cor1['corpus'] = "ElectionsSM"
                cor1['kind'] = "logos"
                corpora_list.append(cor1)
                summary_corpora_list.append(cor1)
                corpora_list_log[cor1['corpus'].iloc[0]] = cor1


    else:
        add_spacelines(2)
        st.write('Choose corpora')
        box_pol1_log = st.checkbox("Covid", value=True)
        box_pol5_log = st.checkbox("ElectionsSM", value=True)

        add_spacelines(1)
        st.write( " ********************************** " )

        corpora_list = []
        corpora_list_et = {}
        corpora_list_log = {}

        if box_pol1_log:
            cor11 = load_data(vac_red)
            cor1 = cor11.copy()
            cor1_src = cor1['source'].unique()
            cor1['conversation_id'] = 0
            cor1_src = [str(s).replace('@', '') for s in cor1_src]
            cor1['Target'] = cor1['Target'].astype('str')
            cor1['source'] = cor1['source'].astype('str').apply(lambda x: ["@"+str(x) if not "@" in x else x][0])
            cor1['Target'] = cor1['Target'].apply(lambda x: ["@"+str(x) if (not "@" in x and x in cor1_src) else x][0])
            cor1['corpus'] = "Covid" # Ethos
            cor1['kind'] = "ethos"
            corpora_list_et[cor1['corpus'].iloc[0]] = cor1
            corpora_list.append(cor1)
            summary_corpora_list.append(cor1)

            cor11 = load_data(vac_red_log)
            cor1 = cor11.copy()

            cor11_2 = load_data(vac_red_log2, indx=False)
            cor11_2 = cor11_2.drop_duplicates(subset = ['locution'])
            d3t = cor11_2.locution.unique()
            d2tp = cor11.locution_premise.unique()
            d2tc = cor11.locution_conclusion.unique()
            d2t = set(d2tc).union(d2tp)
            d2t_neu = set(d3t).difference(d2t)
            cor11_2 = cor11_2[cor11_2.locution.isin(d2t_neu)]
            cor11_2['connection'] = 'neutral'
            cor1 = cor11.copy()
            cor1['premise'] =  cor1.groupby(['id_connection'])[['premise']].transform(lambda x: " ".join(x) )
            cor1['locution_premise'] =  cor1.groupby(['id_connection'])[['locution_premise']].transform(lambda x: " ".join(x) )
            cor1 = cor1.drop_duplicates('id_connection')
            cor1 = pd.concat([cor1, cor11_2 ], axis = 0, ignore_index = True)
            cor1['kind'] = "logos"
            cor1['corpus'] = "Covid"
            corpora_list.append(cor1)
            summary_corpora_list.append(cor1)
            corpora_list_log[cor1['corpus'].iloc[0]] = cor1


        if box_pol5_log:
            cor55 = load_data(us16)
            cor5 = cor55.copy()
            cor5['Target'] = cor5['Target'].astype('str')
            cor5['corpus'] = "ElectionsSM"
            cor5['source'] = cor5['source'].astype('str')
            cor5['kind'] = "ethos"
            corpora_list_et[cor5['corpus'].iloc[0]] = cor5
            corpora_list.append(cor5)
            summary_corpora_list.append(cor5)

            cor55 = load_data(us16_log, indx=False)
            cor5 = cor55.copy()
            #summary_corpora_list_raw.append(cor11_2)

            cor11_2d = pd.read_excel(us16_log2, sheet_name = 'dr1')
            cor11_2g = pd.read_excel(us16_log2, sheet_name = 'gr1')
            cor11_2r = pd.read_excel(us16_log2, sheet_name = 'rr1')
            cor11_2 = pd.concat( [cor11_2d, cor11_2g, cor11_2r], axis = 0, ignore_index = True )

            cor11_2 = cor11_2.drop_duplicates(subset = ['locution'])
            d3t = cor11_2.locution.unique()
            d2tp = cor55.locution_premise.unique()
            d2tc = cor55.locution_conclusion.unique()
            d2t = set(d2tc).union(d2tp)
            d2t_neu = set(d3t).difference(d2t)
            cor11_2 = cor11_2[cor11_2.locution.isin(d2t_neu)]
            cor11_2['connection'] = 'neutral'

            cor5['premise'] =  cor5.groupby(['id_connection'])[['premise']].transform(lambda x: " ".join(x) )
            cor5['locution_premise'] =  cor5.groupby(['id_connection'])[['locution_premise']].transform(lambda x: " ".join(x) )
            cor5 = cor5.drop_duplicates('id_connection')
            cor1 = pd.concat([cor5, cor11_2 ], axis = 0, ignore_index = True)
            cor1['corpus'] = "ElectionsSM"
            cor1['kind'] = "logos"
            corpora_list.append(cor1)
            summary_corpora_list.append(cor1)
            corpora_list_log[cor1['corpus'].iloc[0]] = cor1


        if len(corpora_list_log.keys()) > 1:
            df_log = pd.concat(corpora_list_log.values(), axis = 0, ignore_index = True)
            df_let = pd.concat(corpora_list_et.values(), axis = 0, ignore_index = True)
            ds = " &\n ".join( list(corpora_list_et.keys()) )
            #print(ds)
            #print(len(corpora_list_log.keys()))
            #print(corpora_list_log.keys())
            #cor = pd.concat([cor1, cor5], axis=0, ignore_index=True)
            df_let['corpus'] = ds
            df_log['corpus'] = ds
            corpora_list = []
            corpora_list.append(df_let)
            corpora_list.append(df_log)



    if contents_radio_type != 'Home Page':
        st.write("### Analysis Units")
        contents_radio_an_cat = st.radio("Unit picker", ('Text-based', 'Entity-based'))
        if contents_radio_an_cat == 'Entity-based':
            contents_radio_an_cat_unit = st.radio("Next choose", ['Target-Based Analysis'] )
            st.write(" ******************************* ")
            st.write("#### Statistical module")
            contents_radio3 = st.radio("Statistic", [ 'Heroes & villains Frequency', "Heroes & villains Score-1", "Heroes & villains Score-2", "Heroes & villains Profile", "Heroes & villains WordCloud"], label_visibility='collapsed') # '(Anti)-heroes',
            #contents_radio3 = st.radio("Statistic", [ 'Heroes & villainses', "Profiles"]) # '(Anti)-heroes',
        else:
            contents_radio_an_cat_unit = st.radio("Next choose", [ 'Relation' ])
            st.write(" ******************************* ")
            st.write("#### Statistical module")
            contents_radio3 = st.radio("Statistic", ('Distribution', 'WordCloud', ), label_visibility='collapsed') # , 'Odds ratio', 'Cases'




#####################  page content  #####################
st.title(f"LEPAn: Logos - Ethos - Pathos Analytics")
add_spacelines(1)

@st.cache_data
def SumCorpEthosTable(dataframe, group_column):
    n_corps = dataframe[group_column].nunique()
    dataframe_desc = dataframe.groupby(group_column)['nwords'].sum().reset_index()
    dataframe_desc.columns = [group_column, '#-words']#, 'avg-#-words'
    dataframe_desc_c = dataframe.groupby(group_column)['nwords_content'].sum().reset_index()
    dataframe_desc_c.columns = [group_column, '#-content words']#, 'avg-#-content words'
    dataframe_desc = dataframe_desc.merge(dataframe_desc_c, on = group_column)

    dataframe.source = dataframe.source.astype('str')
    dataframe_src = dataframe.groupby(group_column, as_index = False)['source'].nunique()
    dataframe_src.columns = [group_column, '# speakers']
    dataframe.Target = dataframe.Target.astype('str')
    dataframe_trg = dataframe[dataframe.Target != 'nan'].groupby(group_column, as_index = False)['Target'].nunique()
    dataframe_trg.columns = [group_column, '# targets']

    dataframe_s = dataframe.groupby(group_column, as_index = False).size()
    dataframe_desc = dataframe_desc.merge(dataframe_s, on = group_column)

    dataframe_desc = dataframe_desc.merge(dataframe_src, on = group_column)
    dataframe_desc = dataframe_desc.merge(dataframe_trg, on = group_column)

    dataframe_ed = dataframe.groupby(group_column, as_index = False)[['ethos density', 'E+', 'E-']].mean()
    dataframe_ed[['ethos density', 'E+', 'E-']] = dataframe_ed[['ethos density', 'E+', 'E-']] * 100
    dataframe_desc = dataframe_desc.merge(dataframe_ed, on = group_column)

    dataframe_ed = dataframe.groupby(group_column, as_index = False)[['pathos density', 'P+', 'P-']].mean()
    dataframe_ed[['pathos density', 'P+', 'P-']] = dataframe_ed[['pathos density', 'P+', 'P-']] * 100
    dataframe_desc = dataframe_desc.merge(dataframe_ed, on = group_column)

    dataframe_desc.loc[len(dataframe_desc)] = ['AVERAGE', dataframe_desc['#-words'].mean(),
                            dataframe_desc['#-content words'].mean(), dataframe_desc['size'].mean(),
                            dataframe_desc['# speakers'].mean(), dataframe_desc['# targets'].mean(),
                            dataframe['ethos density'].mean()* 100, dataframe['E+'].mean()* 100, dataframe['E-'].mean()* 100,
                            dataframe['pathos density'].mean()* 100, dataframe['P+'].mean()* 100, dataframe['P-'].mean()* 100]
    dataframe_desc = dataframe_desc.round(1)
    dataframe_desc.loc[len(dataframe_desc)] = ['TOTAL', dataframe_desc['#-words'].iloc[:int(n_corps)].sum(),
                            dataframe_desc['#-content words'].iloc[:int(n_corps)].sum(),
                            dataframe_desc['size'].iloc[:int(n_corps)].sum().round(0),
                            dataframe['source'].nunique(), dataframe['Target'].nunique(),
                            ' n/a ', ' n/a ', ' n/a ', ' n/a ', ' n/a ', ' n/a ']
    dataframe_desc[['#-words', '#-content words', '# speakers', '# targets', 'size']] = dataframe_desc[['#-words', '#-content words', '# speakers', '# targets', 'size']].astype('int')
    return dataframe_desc



#####################  page content  #####################


if contents_radio_type == 'Home Page':
    st.write("LEPAn_v01 ")
    add_spacelines(2)
    MainPage()


else:

    if contents_radio3 == "Corpora Summary":
        st.write("### Corpora Summary")
        add_spacelines(2)

        if contents_radio_rhetoric_category_ethos:
            cc = pd.concat( summary_corpora_list, axis = 0, ignore_index = True )
            cc['ethos density'] = np.where(cc.ethos_label != 0, 1, 0)
            cc['E+'] = np.where(cc.ethos_label == 1, 1, 0)
            cc['E-'] = np.where(cc.ethos_label == 2, 1, 0)
            cc['pathos density'] = np.where(cc.pathos_label != 0, 1, 0)
            cc['P+'] = np.where(cc.pathos_label == 1, 1, 0)
            cc['P-'] = np.where(cc.pathos_label == 2, 1, 0)
            cc['topic'] = cc.corpus.apply(lambda x: str(x).split()[0] )
            cc['platform'] = cc.corpus.apply(lambda x: str(x).split()[-1] )
            n_corps = cc['corpus'].nunique()

            len_corpora = {}
            cc['nwords_content'] = cc['content'].astype("str").str.split().map(len)
            len_corpora[cc['corpus'].iloc[0]] = cc.shape[0]
            cc_desc = cc.groupby('corpus')['nwords'].sum().reset_index()
            cc_desc.columns = ['corpus', '#-words']#, 'avg-#-words'
            cc_desc_c = cc.groupby('corpus')['nwords_content'].sum().reset_index()
            cc_desc_c.columns = ['corpus', '#-content words']#, 'avg-#-content words'
            cc_desc = cc_desc.merge(cc_desc_c, on = 'corpus')

            cc.source = cc.source.astype('str')
            cc_src = cc.groupby('corpus', as_index = False)['source'].nunique()
            cc_src.columns = ['corpus', '# speakers']
            cc.Target = cc.Target.astype('str')
            cc_trg = cc[cc.Target != 'nan'].groupby('corpus', as_index = False)['Target'].nunique()
            cc_trg.columns = ['corpus', '# targets']

            cc_s = cc.groupby('corpus', as_index = False).size()
            cc_desc = cc_desc.merge(cc_s, on = 'corpus')

            cc_desc = cc_desc.merge(cc_src, on = 'corpus')
            cc_desc = cc_desc.merge(cc_trg, on = 'corpus')

            cc_ed = cc.groupby('corpus', as_index = False)[['ethos density', 'E+', 'E-']].mean()
            cc_ed[['ethos density', 'E+', 'E-']] = cc_ed[['ethos density', 'E+', 'E-']] * 100
            cc_desc = cc_desc.merge(cc_ed, on = 'corpus')

            cc_ed = cc.groupby('corpus', as_index = False)[['pathos density', 'P+', 'P-']].mean()
            cc_ed[['pathos density', 'P+', 'P-']] = cc_ed[['pathos density', 'P+', 'P-']] * 100
            cc_desc = cc_desc.merge(cc_ed, on = 'corpus')
            #print(cc_desc.columns)
            # 'corpus', 'sum-#-words', 'avg-#-words', 'sum-#-content words', 'avg-#-content words', '# speakers', '# targets'
            #cc_desc.loc[len(cc_desc)] = ['TOTAL', cc_desc['sum-#-words'].sum(), cc.nwords.mean(),
            #                        cc_desc['sum-#-content words'].sum(), cc.nwords_content.mean(),
            #                        cc['source'].nunique(), cc['Target'].nunique(),
            #                        cc['ethos density'].mean()* 100, cc['E+'].mean()* 100, cc['E-'].mean()* 100]
            #cc_desc.loc[len(cc_desc)] = ['AVERAGE', cc_desc['sum-#-words'].sum(), cc.nwords.mean(),
            #                        cc_desc['sum-#-content words'].sum(), cc.nwords_content.mean(),
            #                        cc['source'].nunique(), cc['Target'].nunique(),
            #                        cc['ethos density'].mean()* 100, cc['E+'].mean()* 100, cc['E-'].mean()* 100]

            cc_desc.loc[len(cc_desc)] = ['AVERAGE', cc_desc['#-words'].mean(),
                                    cc_desc['#-content words'].mean(), cc_desc['size'].mean(),
                                    cc_desc['# speakers'].mean(), cc_desc['# targets'].mean(),
                                    cc['ethos density'].mean()* 100, cc['E+'].mean()* 100, cc['E-'].mean()* 100,
                                    cc['pathos density'].mean()* 100, cc['P+'].mean()* 100, cc['P-'].mean()* 100]
            cc_desc = cc_desc.round(1)
            cc_desc.loc[len(cc_desc)] = ['TOTAL', cc_desc['#-words'].iloc[:int(n_corps)].sum(),
                                    cc_desc['#-content words'].iloc[:int(n_corps)].sum(),
                                    cc_desc['size'].iloc[:int(n_corps)].sum().round(0),
                                    cc['source'].nunique(), cc['Target'].nunique(),
                                    ' n/a ', ' n/a ', ' n/a ', ' n/a ', ' n/a ', ' n/a ']
            #st.write(cc_ed)
            cc_desc[['#-words', '#-content words', '# speakers', '# targets', 'size']] = cc_desc[['#-words', '#-content words', '# speakers', '# targets', 'size']].astype('int')
            #st.write(cc_desc)
            #add_spacelines(2)
            cc_desc2 = SumCorpEthosTable(dataframe = cc, group_column = 'corpus')
            st.write(cc_desc2)

            # download button 2 to download dataframe as xlsx
            import io
            buffer = io.BytesIO()
            @st.cache_data
            def convert_to_csv(df):
                # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv(index=False, sep = '\t').encode('utf-8')

            csv_cc_desc2 = convert_to_csv(cc_desc2)
            # download button 1 to download dataframe as csv
            download1 = st.download_button(
                label="Download data as TSV",
                data=csv_cc_desc2,
                file_name='summary_df_corpus.tsv',
                mime='text/csv'
            )

            add_spacelines(2)

            cc_desc_top = SumCorpEthosTable(dataframe = cc, group_column = 'topic')
            st.write(cc_desc_top)

            csv_cc_desc3 = convert_to_csv(cc_desc_top)
            # download button 1 to download dataframe as csv
            download1 = st.download_button(
                label="Download data as TSV",
                data=csv_cc_desc3,
                file_name='summary_df_topic.tsv',
                mime='text/csv'
            )

            add_spacelines(2)

            cc_desc_platform = SumCorpEthosTable(dataframe = cc, group_column = 'platform')
            st.write(cc_desc_platform)
            add_spacelines(2)

            csv_cc_desc4 = convert_to_csv(cc_desc_platform)
            # download button 1 to download dataframe as csv
            download1 = st.download_button(
                label="Download data as TSV",
                data=csv_cc_desc4,
                file_name='summary_df_platform.tsv',
                mime='text/csv'
            )


        else:
            rels = ['Default Conflict', 'Default Rephrase', 'Default Inference',
                    'Logos Neutral', 'Logos Attack', 'Logos Rephrase', 'Logos Support']
            cc = pd.concat( summary_corpora_list[1::2], axis = 0, ignore_index = True )
            cc_raw = pd.concat( summary_corpora_list_raw.values(), axis = 0, ignore_index = True )
            cc_len = pd.concat( summary_corpora_list_raw_len, axis = 0, ignore_index = True )
            cc_raw['locution'] = cc_raw['locution'].apply(lambda x: ":".join( str(x).split(":")[1:] ))
            cce = pd.concat( summary_corpora_list[::2], axis = 0, ignore_index = True )
            cce['ethos density'] = np.where(cce.ethos_label != 0, 1, 0)
            cce['E+'] = np.where(cce.ethos_label == 1, 1, 0)
            cce['E-'] = np.where(cce.ethos_label == 2, 1, 0)
            cc_eth = cce.groupby('corpus', as_index = False)[['ethos density','E+','E-']].mean()
            cc_eth[['ethos density','E+','E-']] = cc_eth[['ethos density','E+','E-']].round(3)*100

            #st.write(cce.groupby('ethos_label', as_index = False).size())


            cc['RepSp'] = 0
            cc['locution_premise'] = cc['locution_premise'].apply(lambda x: ":".join( str(x).split(":")[1:] ))
            cc['locution_conclusion'] = cc['locution_conclusion'].apply(lambda x: ":".join( str(x).split(":")[1:] ))
            cc['locution'] = cc['locution_premise'].astype('str') + " " + cc['locution_conclusion'].astype('str')
            #st.write(cc_len)
            cc_raw['connection'] = 'Logos Neutral'
            cc_raw['RepSp'] = np.where(cc_raw[['RepSp_int', 'RepSp_ext']].any(axis=1), 1, 0)
            cc.connection = cc.connection.map({'Default Conflict': 'Logos Attack',
                                            'Default Rephrase' : 'Logos Rephrase', 'Default Inference' : 'Logos Support'})

            cc = pd.concat( [cc, cc_raw], axis = 0, ignore_index = True )
            cc['nwords'] = cc.locution.astype('str').str.split().map(len)
            #st.write(cc)

            cc = cc[cc.connection.isin(rels)]
            cc_desc = pd.DataFrame( cc.groupby('corpus')['connection'].value_counts(normalize=True).round(3)*100 )
            #cc_desc = cc_desc.rename(columns = {'connection':'%'})
            cc_desc = cc_desc.reset_index()
            #st.write(cc_desc)

            cc_desc = cc_desc.pivot(index = 'corpus', columns = 'connection', values = 'proportion')
            #st.write(cc_desc)
            cc_descrs = cc.groupby('corpus').RepSp.mean().round(3)*100
            cc_descrs = cc_descrs.reset_index()
            cc_desc = cc_desc.merge(cc_descrs, on = 'corpus')

            cc_desc = cc_desc.merge(cc_eth, on = 'corpus')

            cc_descnw = cc_len.groupby('corpus', as_index = False).nwords.sum()
            cc_descnw.columns = ['corpus', '#-words']
            cc_desc = cc_desc.merge(cc_descnw, on = 'corpus')

            cc_sz = pd.DataFrame({'corpus':summary_corpora_dict_raw.keys(), 'size':summary_corpora_dict_raw.values()})
            cc_desc = cc_desc.merge(cc_sz, on = 'corpus')

            av = ['AVERAGE']
            av.extend(cc_desc.mean(axis=0, numeric_only = True).values)
            cc_desc.loc[len(cc_desc)] = av
            cc_desc = cc_desc.round(1)

            cc_desc.loc[len(cc_desc)] = ['TOTAL', ' n/a ', ' n/a ', ' n/a ', ' n/a ', ' n/a ', ' n/a ', ' n/a ', ' n/a ',
                                        cc_desc['#-words'].iloc[:int(len(summary_corpora_list[1::2]))].sum(),
                                        cc_desc['size'].iloc[:int(len(summary_corpora_list[1::2]))].sum() ]

            cc_desc = cc_desc.fillna(' n/a ')
            cc_desc[['size', '#-words']] = cc_desc[['size', '#-words']].astype('int')
            cc_desc = cc_desc.round(1)
            st.write(cc_desc)

            add_spacelines(2)
            # download button 2 to download dataframe as xlsx
            import io
            buffer = io.BytesIO()
            @st.cache_data
            def convert_to_csv(df):
                # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv(index=False, sep = '\t').encode('utf-8')

            csv_cc_desc = convert_to_csv(cc_desc)
            # download button 1 to download dataframe as csv
            download1 = st.download_button(
                label="Download data as TSV",
                data=csv_cc_desc,
                file_name='summary_df.tsv',
                mime='text/csv'
            )



    elif contents_radio_type == 'Comparative Corpora Analysis' and contents_radio3 == 'Distribution':
        distribution_plot_compare_logos(data_list = corpora_list, an_type = contents_radio_an_cat)

    elif contents_radio_type == 'Single Corpus Analysis' and contents_radio3 == 'Distribution':
        corpora_list_ethos = corpora_list[::2]
        corpora_list_ethos_df = pd.concat( corpora_list_ethos, axis=0, ignore_index = True )
        corp_new = "&\n ".join( corpora_list_ethos_df['corpus'].unique() )
        corpora_list_ethos_df['corpus'] = corp_new
        #st.write(corpora_list_ethos_df, len(corpora_list_ethos))

        corpora_list_logos = corpora_list[1::2]
        corpora_list_logos_df = pd.concat( corpora_list_logos, axis=0, ignore_index = True )
        corp_new = "&\n ".join( corpora_list_logos_df['corpus'].unique() )
        corpora_list_logos_df['corpus'] = corp_new
        #st.write(corpora_list_logos_df, len(corpora_list_logos))
        #st.stop()
        corpora_list = []
        corpora_list.append(corpora_list_ethos_df)
        corpora_list.append(corpora_list_logos_df)
        distribution_plot_compare_logos(data_list = corpora_list, an_type = contents_radio_an_cat)


    elif contents_radio_type == 'Comparative Corpora Analysis' and contents_radio3 == 'Profiles':
        st.info("Module not available currently for the Comparative Corpora Analysis type")
        st.stop()
        st.write("### Profiles")
        rhetoric_dims = ['ethos']
        add_spacelines(2)
        selected_rhet_dim = st.selectbox("Choose a rhetoric category of profiles", rhetoric_dims, index=0)
        add_spacelines(1)
        if selected_rhet_dim != 'logos':
            selected_rhet_dim = selected_rhet_dim.replace("ethos", "ethos_label").replace("pathos", "pathos_label")

        if len(corpora_list) == 6:
            cols_columns1, cols_columns2, cols_columns3 = st.tabs([corpora_list[0].corpus.iloc[0], corpora_list[2].corpus.iloc[0], corpora_list[-2].corpus.iloc[0]])
            with cols_columns1:
                #st.write(corpora_list[0])
                ProfilesEntity_compare(data_list = corpora_list[:1], selected_rhet_dim = selected_rhet_dim)
            with cols_columns2:
                ProfilesEntity_compare(data_list = corpora_list[2:3], selected_rhet_dim = selected_rhet_dim)
            with cols_columns3:
                ProfilesEntity_compare(data_list = corpora_list[-2:-1], selected_rhet_dim = selected_rhet_dim)

        elif len(corpora_list) == 4:
            cols_columns1, cols_columns2 = st.tabs([corpora_list[0].corpus.iloc[0], corpora_list[-2].corpus.iloc[0]])
            with cols_columns1:
                #st.write(corpora_list[0])
                ProfilesEntity_compare(data_list = corpora_list[:1], selected_rhet_dim = selected_rhet_dim)
            with cols_columns2:
                ProfilesEntity_compare(data_list = corpora_list[-2:-1], selected_rhet_dim = selected_rhet_dim)

        elif len(corpora_list) == 2:
            ProfilesEntity_compare(data_list = corpora_list[:1], selected_rhet_dim = selected_rhet_dim)


    elif contents_radio_type != 'Comparative Corpora Analysis' and contents_radio3 == 'Heroes & villains':
        corpora_list_ethos = corpora_list[::2]
        corpora_list_ethos_df = pd.concat( corpora_list_ethos, axis=0, ignore_index = True )
        corp_new = "&\n ".join( corpora_list_ethos_df['corpus'].unique() )
        corpora_list_ethos_df['corpus'] = corp_new
        #st.write(corpora_list_ethos_df, len(corpora_list_ethos))

        corpora_list_logos = corpora_list[1::2]
        corpora_list_logos_df = pd.concat( corpora_list_logos, axis=0, ignore_index = True )
        corp_new = "&\n ".join( corpora_list_logos_df['corpus'].unique() )
        corpora_list_logos_df['corpus'] = corp_new
        #st.write(corpora_list_logos_df, len(corpora_list_logos))
        #st.stop()
        corpora_list = []
        corpora_list.append(corpora_list_ethos_df)
        corpora_list.append(corpora_list_logos_df)
        TargetHeroScores_compare(data_list = corpora_list[:1], singl_an = False)

    elif contents_radio_type != 'Comparative Corpora Analysis' and contents_radio3 == 'Heroes & villains Frequency':
        corpora_list_ethos = corpora_list[::2]
        corpora_list_ethos_df = pd.concat( corpora_list_ethos, axis=0, ignore_index = True )
        corp_new = "&\n ".join( corpora_list_ethos_df['corpus'].unique() )
        corpora_list_ethos_df['corpus'] = corp_new
        #st.write(corpora_list_ethos_df, len(corpora_list_ethos))

        corpora_list_logos = corpora_list[1::2]
        corpora_list_logos_df = pd.concat( corpora_list_logos, axis=0, ignore_index = True )
        corp_new = "&\n ".join( corpora_list_logos_df['corpus'].unique() )
        corpora_list_logos_df['corpus'] = corp_new
        #st.write(corpora_list_logos_df, len(corpora_list_logos))
        #st.stop()
        corpora_list = []
        corpora_list.append(corpora_list_ethos_df)
        corpora_list.append(corpora_list_logos_df)
        TargetHeroScores_compare_freq(data_list = corpora_list[:1], singl_an = False)

    elif contents_radio_type != 'Comparative Corpora Analysis' and contents_radio3 == "Heroes & villains Score-1":
        corpora_list_ethos = corpora_list[::2]
        corpora_list_ethos_df = pd.concat( corpora_list_ethos, axis=0, ignore_index = True )
        corp_new = "&\n ".join( corpora_list_ethos_df['corpus'].unique() )
        corpora_list_ethos_df['corpus'] = corp_new
        #st.write(corpora_list_ethos_df, len(corpora_list_ethos))

        corpora_list_logos = corpora_list[1::2]
        corpora_list_logos_df = pd.concat( corpora_list_logos, axis=0, ignore_index = True )
        corp_new = "&\n ".join( corpora_list_logos_df['corpus'].unique() )
        corpora_list_logos_df['corpus'] = corp_new
        #st.write(corpora_list_logos_df, len(corpora_list_logos))
        #st.stop()
        corpora_list = []
        corpora_list.append(corpora_list_ethos_df)
        corpora_list.append(corpora_list_logos_df)
        TargetHeroScores_compare_scor(data_list = corpora_list[:1], singl_an = False)


    elif contents_radio_type != 'Comparative Corpora Analysis' and contents_radio3 == "Heroes & villains Score-2":
        corpora_list_ethos = corpora_list[::2]
        corpora_list_ethos_df = pd.concat( corpora_list_ethos, axis=0, ignore_index = True )
        corp_new = "&\n ".join( corpora_list_ethos_df['corpus'].unique() )
        corpora_list_ethos_df['corpus'] = corp_new
        #st.write(corpora_list_ethos_df, len(corpora_list_ethos))

        corpora_list_logos = corpora_list[1::2]
        corpora_list_logos_df = pd.concat( corpora_list_logos, axis=0, ignore_index = True )
        corp_new = "&\n ".join( corpora_list_logos_df['corpus'].unique() )
        corpora_list_logos_df['corpus'] = corp_new
        #st.write(corpora_list_logos_df, len(corpora_list_logos))
        #st.stop()
        corpora_list = []
        corpora_list.append(corpora_list_ethos_df)
        corpora_list.append(corpora_list_logos_df)
        TargetHeroScores_compare_scor2(data_list = corpora_list[:1], singl_an = False)


    elif contents_radio_type != 'Comparative Corpora Analysis' and contents_radio3 == "Heroes & villains Profile":
        corpora_list_ethos = corpora_list[::2]
        corpora_list_ethos_df = pd.concat( corpora_list_ethos, axis=0, ignore_index = True )
        corp_new = "&\n ".join( corpora_list_ethos_df['corpus'].unique() )
        corpora_list_ethos_df['corpus'] = corp_new
        #st.write(corpora_list_ethos_df, len(corpora_list_ethos))

        corpora_list_logos = corpora_list[1::2]
        corpora_list_logos_df = pd.concat( corpora_list_logos, axis=0, ignore_index = True )
        corp_new = "&\n ".join( corpora_list_logos_df['corpus'].unique() )
        corpora_list_logos_df['corpus'] = corp_new
        #st.write(corpora_list_logos_df, len(corpora_list_logos))
        #st.stop()
        corpora_list = []
        corpora_list.append(corpora_list_ethos_df)
        corpora_list.append(corpora_list_logos_df)
        TargetHeroScores_compare_prof(data_list = corpora_list[:1], singl_an = False)




    elif contents_radio_type != 'Comparative Corpora Analysis' and contents_radio3 == "Heroes & villains WordCloud":
        corpora_list_ethos = corpora_list[::2]
        corpora_list_ethos_df = pd.concat( corpora_list_ethos, axis=0, ignore_index = True )

        corpora_list = []
        corpora_list.append(corpora_list_ethos_df)

        rhetoric_dims = ['ethos',]

        selected_rhet_dim = st.selectbox("Choose a rhetoric category for a WordCloud", rhetoric_dims, index=0)


        ccol1, ccol2 = st.columns(2)
        with ccol1:
            st.write("Choose category of the target of ethotic statements")
            box_direct = st.checkbox("direct ethos", value = False)
            box_3rd = st.checkbox("3rd party ethos", value = True)
        with ccol2:
            label_cloud = st.radio("Choose a label for words in WordCloud", ('attack', 'support'))
        selected_rhet_dim = selected_rhet_dim.replace("ethos", "ethos_label")
        label_cloud = label_cloud.replace("attack / negative", "attack").replace("support / positive", "support")
        add_spacelines(1)


        target_shared = {}

        n = 0
        for data in corpora_list:
            df = data.copy()

            if not 'attack' in df['ethos_label'].unique():
                df['ethos_label'] = df['ethos_label'].map(ethos_mapping)
            df["Target"] = df["Target"].astype('str')
            df = df[ (df.Target != 'nan') & (df.Target != '') & (df.ethos_label != 'neutral') ]
            df["Target"] = df["Target"].str.replace('Government', 'government')
            target_shared[n] = set(df["Target"].unique())

            if box_direct and not box_3rd:
                targets_limit = df['Target'].dropna().unique()
                targets_limit = [t for t in targets_limit if "@" in t]
                df = df[df.Target.isin(targets_limit)]
                target_shared[n] = set(df["Target"].unique())

            elif not box_direct and box_3rd:
                targets_limit = df['Target'].dropna().unique()
                targets_limit = [t for t in targets_limit if not "@" in t]
                df = df[df.Target.isin(targets_limit)]
                target_shared[n] = set(df["Target"].unique())

            n+=1

        target_shared_list = list(target_shared[0] )
        selected_target = st.selectbox("Choose a target entity you would like to analyse", target_shared_list )
        sel_tar = True
        add_spacelines(1)

        threshold_cloud = st.slider('Select a precision value (threshold) for words in WordCloud', 0, 100, 51)
        st.info(f'Selected precision: **{threshold_cloud}**')
        add_spacelines(1)

        box_stopwords = st.checkbox( "Enable stop words", value = False )


        cols_columns = st.columns( int( len(corpora_list) ) )
        dict_cond = {}
        nn = 0
        for n, c in enumerate(cols_columns):
            with c:
                add_spacelines(1)
                AntiHeroWordCloud_compare(corpora_list[nn:nn+1], rhetoric_dims = ['ethos'], box_stopwords = box_stopwords,
                    selected_rhet_dim = selected_rhet_dim, label_cloud=label_cloud, threshold_cloud=threshold_cloud, targeted = sel_tar, selected_target = selected_target)
                nn += 1


    elif contents_radio_type == 'Comparative Corpora Analysis' and contents_radio3 == "Heroes & villains WordCloud":
        #add_spacelines(1)
        #st.write("Choose category of the target of ethotic statements")
        #box_direct = st.checkbox("direct ethos", value = False)
        #box_3rd = st.checkbox("3rd party ethos", value = True)
        #TargetHeroScores_compare_word(data_list = corpora_list[::2], singl_an = False, chbox_3rd = box_3rd, chbox_direct = box_direct)

        rhetoric_dims = ['ethos',]
        corpora_list = corpora_list[::2]

        selected_rhet_dim = st.selectbox("Choose a rhetoric category for a WordCloud", rhetoric_dims, index=0)


        ccol1, ccol2 = st.columns(2)
        with ccol1:
            st.write("Choose category of the target of ethotic statements")
            box_direct = st.checkbox("direct ethos", value = False)
            box_3rd = st.checkbox("3rd party ethos", value = True)
        with ccol2:
            label_cloud = st.radio("Choose a label for words in WordCloud", ('attack', 'support'))
        selected_rhet_dim = selected_rhet_dim.replace("ethos", "ethos_label")
        label_cloud = label_cloud.replace("attack / negative", "attack").replace("support / positive", "support")
        add_spacelines(1)


        target_shared = {}

        n = 0
        for data in corpora_list:
            df = data.copy()

            if not 'attack' in df['ethos_label'].unique():
                df['ethos_label'] = df['ethos_label'].map(ethos_mapping)
            df["Target"] = df["Target"].astype('str')
            df = df[ (df.Target != 'nan') & (df.Target != '') & (df.ethos_label != 'neutral') ]
            df["Target"] = df["Target"].str.replace('Government', 'government')
            target_shared[n] = set(df["Target"].unique())

            if box_direct and not box_3rd:
                targets_limit = df['Target'].dropna().unique()
                targets_limit = [t for t in targets_limit if "@" in t]
                df = df[df.Target.isin(targets_limit)]
                target_shared[n] = set(df["Target"].unique())

            elif not box_direct and box_3rd:
                targets_limit = df['Target'].dropna().unique()
                targets_limit = [t for t in targets_limit if not "@" in t]
                df = df[df.Target.isin(targets_limit)]
                target_shared[n] = set(df["Target"].unique())

            n+=1

        #target_shared_list = list(target_shared[0] )
        if len( list(target_shared.keys()) )  > 1:
            target_shared_list = []
            kk = list(target_shared.keys())
            target_shared_list = list( set(target_shared[kk[0]]).intersection(set(target_shared[kk[1]])) )
        selected_target = st.selectbox("Choose a target entity you would like to analyse", target_shared_list )
        sel_tar = True
        add_spacelines(1)

        threshold_cloud = st.slider('Select a precision value (threshold) for words in WordCloud', 0, 100, 51)
        st.info(f'Selected precision: **{threshold_cloud}**')
        add_spacelines(1)

        box_stopwords = st.checkbox( "Enable stop words", value = False )


        cols_columns = st.columns( int( len(corpora_list) ) )
        dict_cond = {}
        nn = 0
        for n, c in enumerate(cols_columns):
            with c:
                add_spacelines(1)
                AntiHeroWordCloud_compare(corpora_list[nn:nn+1], rhetoric_dims = ['ethos'], box_stopwords = box_stopwords,
                    selected_rhet_dim = selected_rhet_dim, label_cloud=label_cloud, threshold_cloud=threshold_cloud, targeted = sel_tar, selected_target = selected_target)
                nn += 1


    elif contents_radio_type == 'Comparative Corpora Analysis' and contents_radio3 == 'Heroes & villains':
        if len(corpora_list) == 4:
            corpora_list0 = corpora_list[0]
            corpora_list1 = corpora_list[-2]
            corpora_list = [corpora_list0, corpora_list1]
        elif len(corpora_list) == 2:
            corpora_list0 = corpora_list[0]
            corpora_list = corpora_list0
        TargetHeroScores_compare(data_list = corpora_list, singl_an = False)

    elif contents_radio_type == 'Comparative Corpora Analysis' and contents_radio3 == 'Heroes & villains Frequency':
        if len(corpora_list) == 4:
            corpora_list0 = corpora_list[0]
            corpora_list1 = corpora_list[-2]
            corpora_list = [corpora_list0, corpora_list1]
        elif len(corpora_list) == 2:
            corpora_list0 = corpora_list[0]
            corpora_list = corpora_list0
        TargetHeroScores_compare_freq(data_list = corpora_list, singl_an = False)

    elif contents_radio_type == 'Comparative Corpora Analysis' and contents_radio3 == "Heroes & villains Score-1":
        if len(corpora_list) == 4:
            corpora_list0 = corpora_list[0]
            corpora_list1 = corpora_list[-2]
            corpora_list = [corpora_list0, corpora_list1]
        elif len(corpora_list) == 2:
            corpora_list0 = corpora_list[0]
            corpora_list = corpora_list0
        TargetHeroScores_compare_scor(data_list = corpora_list, singl_an = False)

    elif contents_radio_type == 'Comparative Corpora Analysis' and contents_radio3 == "Heroes & villains Score-2":
        if len(corpora_list) == 4:
            corpora_list0 = corpora_list[0]
            corpora_list1 = corpora_list[-2]
            corpora_list = [corpora_list0, corpora_list1]
        elif len(corpora_list) == 2:
            corpora_list0 = corpora_list[0]
            corpora_list = corpora_list0
        TargetHeroScores_compare_scor2(data_list = corpora_list, singl_an = False)


    elif contents_radio_type == 'Comparative Corpora Analysis' and contents_radio3 == "Heroes & villains Profile":
        if len(corpora_list) == 4:
            corpora_list0 = corpora_list[0]
            corpora_list1 = corpora_list[-2]
            corpora_list = [corpora_list0, corpora_list1]
        elif len(corpora_list) == 2:
            corpora_list0 = corpora_list[0]
            corpora_list = corpora_list0
        TargetHeroScores_compare_prof(data_list = corpora_list, singl_an = False)



    elif contents_radio_type != 'Comparative Corpora Analysis' and contents_radio3 == 'Profiles':
        corpora_list_ent = []
        df_user_et = corpora_list[0]
        df_user_log = corpora_list[1]
        #st.write(df_user_log)
        df_user_et_src = set( df_user_et.source.dropna().astype('str').str.replace("@", "").str.strip().unique() )
        df_user_log_src = set( df_user_log.speaker_premise.dropna().astype('str').str.replace(":", "").str.replace("@", "").str.strip().unique() )
        df_user_src = df_user_et_src.intersection(df_user_log_src)

        df_user_et = df_user_et[df_user_et.source.dropna().astype('str').str.replace("@", "").str.strip().isin(df_user_src) ]
        df_user_log = df_user_log[df_user_log.speaker_premise.dropna().astype('str').str.replace(":", "").str.replace("@", "").str.strip().isin(df_user_src) ]
        df_ents = pd.concat([df_user_log, df_user_et], axis = 0, ignore_index = True)

        src_logp = df_ents.groupby('speaker_premise', as_index=False).size()
        #src_logp = src_logp[src_logp['size'] > 2]
        src_logp = df_ents.speaker_premise.dropna().astype('str').str.strip().unique()
        src_et = df_ents.groupby('source', as_index=False).size()
        #src_et = src_et[src_et['size'] > 2]
        src_et = src_et.source.dropna().astype('str').str.strip().unique()
        src_list = list( set(src_et).union(set(src_logp) ) )
        src_list = list(set(str(e).replace("@", "") for e in src_list))
        if 'look' in src_list:
            src_list.remove('look')
        st.write("### Speaker Analysis")
        add_spacelines(2)
        src = st.selectbox("Choose an entity for analysis", src_list, index=0)

        df_user_et.source = df_user_et.source.dropna().astype('str').str.replace("@", "").str.strip()
        df_user_log.speaker_premise = df_user_log.speaker_premise.dropna().astype('str').str.replace(":", "").str.replace("@", "").str.strip()
        df_user_et = df_user_et[df_user_et.source == src]
        df_user_log = df_user_log[(df_user_log.speaker_premise == src) ] #  (df_user_log.speaker_conclusion == src) |

        try:
            ds = df_user_et.corpus.iloc[0]
        except:
            ds = df_user_log.corpus.iloc[0]
        ds = ds + " - **" + str(src) +"**"
        df_user_et.corpus = ds
        df_user_log.corpus = ds
        if len(df_user_et) > 0:
            corpora_list_ent.append(df_user_et)
        if len(df_user_log) > 0:
            corpora_list_ent.append(df_user_log)
        distribution_plot_compare_logos(data_list = corpora_list_ent, an_type = contents_radio_an_cat)



    elif contents_radio3 == 'WordCloud' and contents_radio_type == 'Comparative Corpora Analysis':
        #generateWordCloud_log(corpora_list, rhetoric_dims = ['ethos', 'logos'], an_type = contents_radio_an_cat)
            
        rhetoric_dims = ['ethos', 'logos', 'pathos']
        selected_rhet_dim = st.selectbox("Choose a rhetoric category for a WordCloud", rhetoric_dims, index=0)
        add_spacelines(1)

        if selected_rhet_dim == 'pathos':
            label_cloud = st.radio("Choose a label for words in WordCloud", ('negative', 'positive'))
            selected_rhet_dim = selected_rhet_dim.replace("ethos", "ethos_label").replace("pathos", "pathos_label")
            label_cloud = label_cloud.replace("negative", "attack").replace("positive", "support")
        else:
            label_cloud = st.radio("Choose a label for words in WordCloud", ('attack', 'support'))
            selected_rhet_dim = selected_rhet_dim.replace("ethos", "ethos_label")
            label_cloud = label_cloud.replace("attack / negative", "attack").replace("support / positive", "support")

        add_spacelines(1)
        threshold_cloud = st.slider('Select a precision value (threshold) for words in WordCloud', 0, 100, 80)
        st.info(f'Selected precision: **{threshold_cloud}**')

        dict_cond = {}
        nn = 0
        for n in range( int( len(corpora_list) / 2 ) ):
            fig_cloud1, df_cloud_words1, freq_word_pos, dd, cc = generateWordCloud_compare(corpora_list[nn:nn+2], rhetoric_dims = ['ethos', 'logos', 'pathos'],
                    selected_rhet_dim = selected_rhet_dim, label_cloud=label_cloud, threshold_cloud=threshold_cloud)
            dict_cond[n] = [fig_cloud1, df_cloud_words1, cc, freq_word_pos, dd]
            nn +=2

        tab_plot, tab_tab, tab_case = st.tabs(['Plots', 'Tables', 'Cases'])            
        with tab_plot:
                cols_columns = st.columns( int( len(corpora_list) / 2 ) )            
                for n, c in enumerate(cols_columns):
                    with c:
                        add_spacelines(1)
                        fig_cloud2 = dict_cond[n][0]
                        cc2 =  dict_cond[n][2]
                        st.write("**{cc2}**")
                        st.pyplot(fig_cloud2)

        with tab_tab:
                cols_columns2 = st.columns( int( len(corpora_list) / 2 ) )            
                for n, c in enumerate(cols_columns2):
                    with c:
                        add_spacelines(1)
                        cc2 =  dict_cond[n][2]
                        st.write("**{cc2}**")
                        st.write(f'WordCloud frequency table: ')
                        df_cloud_words2 = dict_cond[n][1]
                        st.write(df_cloud_words2)

        with tab_case:
                cols_columns3 = st.columns( int( len(corpora_list) / 2 ) )            
                for n, c in enumerate(cols_columns3):
                    with c:
                        add_spacelines(1)
                        cc2 =  dict_cond[n][2]
                        st.write("**{cc2}**")
                        st.write(f'WordCloud frequency table: ')
                        freq_word_pos2 = dict_cond[n][-2]                    
                        st.write(f'Cases with **{freq_word_pos2}** words:')
                        dd2 = dict_cond[n][-1]
                        st.dataframe(dd2)

        

    elif contents_radio_type == 'Single Corpus Analysis' and contents_radio3 == 'WordCloud':
        generateWordCloud(corpora_list, rhetoric_dims = ['ethos', 'logos', 'pathos'], an_type = contents_radio_an_cat)
