# old
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


#import spacy
#nlp = spacy.load('en_core_web_sm')

pd.options.mode.chained_assignment = None
import warnings
#warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


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



def MainPage():
    #add_spacelines(2)
    with st.expander("Read abstract"):
        add_spacelines(1)
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
                        "Corpus": ['PolarIs1', 'US2016reddit', 'Total'],
                        "# Words": [30014, 30099, 30014 + 30099], 
                        "# ADU": [2706, 3827, 2706 + 3827], 
                        "# Posts": [963, 1317, 963 + 1317], 
                        "# Speakers": [465, 1317, 465 + 1317], 
                }
        )

        df_iaa = pd.DataFrame(
                {
                        'Corpus': [ 'PolarIs1', 'US2016reddit', 'Total/Average' ], 
                        'L-' : [  630, 581, 630 + 581],  
                        'L+' : [  1233, 1144, 1233 + 1144 ],  
                        'IAA L' : [  0.618, 0.817, np.mean([0.618, 0.817]) ],  
                        
                        'E-' : [ 440, 847, 440 + 847 ],  
                        'E+' : [ 59, 492, 59 + 492 ],  
                       'IAA E' : [  0.752, 0.793, np.mean([0.752, 0.793]) ],  
                        
                        'P-' : [  653, 1294, 653 + 1294 ],  
                        'P+' : [  152, 190, 152 + 190],  
                        'IAA P' : [  0.417, 0.573, np.mean([0.417, 0.573]) ],    
                        
                        
                }
        )            
        #df_iaa.columns = [ ('Annotation', 'corpus'), ('Logos', 'L-'), ('Logos', 'L+'), ('Logos', 'IAA L' ), 
        #             ('Ethos', 'E-'), ('Ethos', 'E+'), ('Ethos', 'IAA E' ), 
        #            ('Pathos', 'P-'), ('Pathos', 'P+'), ('Pathos', 'IAA P' )  ]
        #df_iaa.columns = pd.MultiIndex.from_tuples(df_iaa.columns, names=[' ','Categories'])

        with st.expander("Data summary"):
                add_spacelines(1)
                st.write( "Datasets used in our technology of LEP Analytics." ) 
                st.dataframe(df_sum.set_index("Corpus"))
                add_spacelines(1)
                st.write( "Annotation of logos, ethos and pathos used in Rhetoric Analytics." )                 
                st.dataframe( df_iaa.set_index('Corpus') )

            
        with st.expander("LEP Categories"):
                add_spacelines(1)
                st.write('''The LEPAn tool makes use of the Aristotelian rhetoric to examine statistical patterns of argumentation in public debates. 
                Three types of rhetorical arguments are distinguished by Aristotle: 
                (i) logotic, which is fact-based, rational argumentation; 
                (ii) ethotic, which is an argument for or against the character (credibility) of the speaker; and 
                (iii) pathotic, which is an emotion-based argumentation that rests on changing the emotional state of the audience.''')
                add_spacelines(1)
                st.image( 'LEPcategoriesAnalytic.png', caption='Three types of rhetorical arguments are distinguished by Aristotle: (1) logotic, (2) ethotic, (3) pathotic.')  # st.image('sunrise.jpg', caption='Sunrise by the mountains') 

            
        add_spacelines(2)
        st.write("**[The New Ethos Lab](https://newethos.org/)**")
        #add_spacelines(1)
        st.write(" ************************************************************* ")
    st.write('<style>div.row-widget.stRadio > div{flex-direction:column;font-size=18px;}</style>', unsafe_allow_html=True)



###################################################


emosn = ['sadness', 'anger', 'fear', 'disgust']
emosp = ['joy'] # 'surprise'
emos_map = {'joy':'emotion_positive', 'surprise':2, 'sadness':'emotion_negative', 'anger':'emotion_negative',
            'fear':'emotion_negative', 'disgust':'emotion_negative', 'neutral':'emotion_neutral'}


def standardize(data):
  from sklearn.preprocessing import StandardScaler
  scaler = StandardScaler()
  data0 = data.copy()
  scaled_values = scaler.fit_transform(data0)
  data0.loc[:, :] = scaled_values
  return data0


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


#@st.cache_data
#def lemmatization(dataframe, text_column = 'sentence', name_column = False):
#  '''Parameters:
#  dataframe: dataframe with your data,
#  text_column: name of a column in your dataframe where text is located'''
#  df = dataframe.copy()
#  lemmas = []
#  for doc in nlp.pipe(df[text_column].astype('str')):
#    lemmas.append(" ".join([token.lemma_ for token in doc if (not token.is_punct and not token.is_stop and not token.like_num and len(token) > 1) ]))
#  if name_column:
#      df[text_column] = lemmas
#  else:
#      df[text_column+"_lemmatized"] = lemmas
#  return df



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
            #df = lemmatization(df, 'sentence_lemmatized', name_column = True)
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
            #dfp = lemmatization(dfp, 'sentence_lemmatized', name_column = True)
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
        ds = "PolarIs1 & US2016reddit"
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
            ds = "PolarIs1 & US2016reddit"
            df['corpus'] = ds
        add_spacelines(1)
        st.write(f'Cases with **{freq_word_pos}** words:')
        st.dataframe(df[ (df['freq_words_'+label_cloud].str.split().map(len) >= 1) ].dropna(axis=1, how='all')[cols_odds1])# .set_index('source')




def generateWordCloud_compare(df_list, selected_rhet_dim, label_cloud, threshold_cloud, rhetoric_dims = ['ethos', 'pathos']):

    if selected_rhet_dim != 'logos':
        df = df_list[0]
        df = clean_text(df, 'sentence_lemmatized', text_column_name = "sentence_lemmatized")
        #st.write( f" **{df.corpus.iloc[0]}** " )
        if not 'neutral' in df['ethos_label'].unique():
            df['ethos_label'] = df['ethos_label'].map(ethos_mapping)
        if not 'negative' in df['pathos_label'].unique():
            df['pathos_label'] = df['pathos_label'].map(valence_mapping)

    elif selected_rhet_dim == 'logos':
        df = df_list[-1] #pd.concat(df_list, axis=0, ignore_index=True)
        #st.write( f" **{df.corpus.iloc[0]}** " )
        df = df.dropna(subset = 'premise')
        df['sentence_lemmatized'] = df['premise'].astype('str').apply(lambda x: re.sub(r"\W+", " ", str(x)))
        #df = lemmatization(df, 'sentence_lemmatized', name_column = True)
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

    #pos_list_freq = df_cloud_words1.word.tolist()
    #freq_word_pos = st.multiselect('Choose word(s) you would like to see data cases for', pos_list_freq, pos_list_freq[:2])
    #df_odds_pos_words = set(freq_word_pos)
    #df['freq_words_'+label_cloud] = df.sentence_lemmatized.apply(lambda x: " ".join( set(x.split()).intersection(df_odds_pos_words) ))
    #add_spacelines(1)
    #st.write(f'Cases with **{freq_word_pos}** words:')
    #dd = df[ (df['freq_words_'+label_cloud].str.split().map(len) >= 1) & (df[selected_rhet_dim] == label_cloud) ][cols_odds1]
    #st.dataframe(dd)# .set_index('source')
    return fig_cloud1, df_cloud_words1, df, cc




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
        #df = lemmatization(df, 'sentence_lemmatized', name_column = True)
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
        #df_dist_hist_all.corpus = df_dist_hist_all.corpus.map( {'PolarIs1':"PolarIs1", 'US2016reddit':'PolarIs5'} )


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
            ds = "PolarIs1 & US2016reddit"
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
            ds = "PolarIs1 & US2016reddit"
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
            ds = "PolarIs1 & US2016reddit"
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
        ds = "PolarIs1 & US2016reddit"
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
            ds = "PolarIs1 & US2016reddit"
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
            ds = "PolarIs1 & US2016reddit"
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
            ds = "PolarIs1 & US2016reddit"
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
            ds = "PolarIs1 & US2016reddit"
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
            ds = "PolarIs1 & US2016reddit"
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
            ds = "PolarIs1 & US2016reddit"
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
            ds = "PolarIs1 & US2016reddit"
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
            ds = "PolarIs1 & US2016reddit"
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
            ds = "PolarIs1 & US2016reddit"
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
                        ds = "PolarIs1 & US2016reddit"
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
                        ds = "PolarIs1 & US2016reddit"
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
            ds = "PolarIs1 & US2016reddit"
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
            ds = "PolarIs1 & US2016reddit"
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
            ds = "PolarIs1 & US2016reddit"
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
            ds = 'PolarIs1 & US2016reddit'
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
            ds = "PolarIs1 & US2016reddit"
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
        box_pol1_log = st.checkbox("PolarIs1", value=True)
        box_pol5_log = st.checkbox("US2016reddit", value=True)

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
                cor1['corpus'] = "PolarIs1"
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
                cor1['corpus'] = "PolarIs1"
                corpora_list.append(cor1)
                summary_corpora_list.append(cor1)
                corpora_list_log[cor1['corpus'].iloc[0]] = cor1


        if box_pol5_log:
                cor55 = load_data(us16)
                cor5 = cor55.copy()
                cor5['Target'] = cor5['Target'].astype('str')
                cor5['corpus'] = "US2016reddit"
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
                cor1['corpus'] = "US2016reddit"
                cor1['kind'] = "logos"
                corpora_list.append(cor1)
                summary_corpora_list.append(cor1)
                corpora_list_log[cor1['corpus'].iloc[0]] = cor1


    else:
        add_spacelines(2)
        st.write('Choose corpora')
        box_pol1_log = st.checkbox("PolarIs1", value=True)
        box_pol5_log = st.checkbox("US2016reddit", value=True)

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
            cor1['corpus'] = "PolarIs1" # Ethos
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
            cor1['corpus'] = "PolarIs1"
            corpora_list.append(cor1)
            summary_corpora_list.append(cor1)
            corpora_list_log[cor1['corpus'].iloc[0]] = cor1


        if box_pol5_log:
            cor55 = load_data(us16)
            cor5 = cor55.copy()
            cor5['Target'] = cor5['Target'].astype('str')
            cor5['corpus'] = "US2016reddit"
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
            cor1['corpus'] = "US2016reddit"
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

        threshold_cloud = st.slider('Select a precision value (threshold) for words in WordCloud', 0, 100, 80)
        st.info(f'Selected precision: **{threshold_cloud}**')

        dict_cond = {}
        nn = 0
        for n in range( int( len(corpora_list) / 2 ) ):
            fig_cloud1, df_cloud_words1, dd, cc = generateWordCloud_compare(corpora_list[nn:nn+2], rhetoric_dims = ['ethos', 'logos', 'pathos'],
                    selected_rhet_dim = selected_rhet_dim, label_cloud=label_cloud, threshold_cloud=threshold_cloud)
            dict_cond[n] = [fig_cloud1, df_cloud_words1, cc, dd]
            nn +=2

        tab_plot, tab_tab, tab_case = st.tabs(['Plots', 'Tables', 'Cases'])            
        with tab_plot:
                cols_columns = st.columns( int( len(corpora_list) / 2 ) )            
                for n, c in enumerate(cols_columns):
                    with c:
                        fig_cloud2 = dict_cond[n][0]
                        cc2 =  dict_cond[n][2]
                        st.write(f"**{cc2}**")
                        st.pyplot(fig_cloud2)

        with tab_tab:
                cols_columns2 = st.columns( int( len(corpora_list) / 2 ) )            
                for n, c in enumerate(cols_columns2):
                    with c:
                        cc2 =  dict_cond[n][2]
                        st.write(f"**{cc2}**")
                        st.write(f'WordCloud frequency table: ')
                        df_cloud_words2 = dict_cond[n][1]
                        st.write(df_cloud_words2)

        with tab_case:
                cols_odds1 = ['source', 'sentence', 'ethos_label', 'pathos_label', 'Target',
                         'freq_words_'+label_cloud]
                cols_columns3 = st.columns( int( len(corpora_list) / 2 ) )            
                for n, c in enumerate(cols_columns3):
                    with c:
                        cc2 =  dict_cond[n][2]
                        st.write(f"**{cc2}**")
                        df = dict_cond[n][-1]
                        df_cloud_words1 =  dict_cond[n][1]
                        pos_list_freq = df_cloud_words1.word.tolist()
                        freq_word_pos = st.multiselect('Choose word(s) you would like to see data cases for', pos_list_freq, pos_list_freq[:2])
                        df_odds_pos_words = set(freq_word_pos)
                        df['freq_words_'+label_cloud] = df.sentence_lemmatized.apply(lambda x: " ".join( set(x.split()).intersection(df_odds_pos_words) ))
                        add_spacelines(1)
                        dd = df[ (df['freq_words_'+label_cloud].str.split().map(len) >= 1) & (df[selected_rhet_dim] == label_cloud) ][cols_odds1]                       
                        st.write(f'Cases with **{freq_word_pos}** words:')
                        st.dataframe(dd)

        

    elif contents_radio_type == 'Single Corpus Analysis' and contents_radio3 == 'WordCloud':
        generateWordCloud(corpora_list, rhetoric_dims = ['ethos', 'logos', 'pathos'], an_type = contents_radio_an_cat)
