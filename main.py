from requests import session
import streamlit as st
import pickle
import altair as alt
import cohere
import pandas as pd
import pygsheets
import config
import os
import numpy as np
import umap
from dotenv import load_dotenv
from tqdm import tqdm
from datasets import load_dataset
import altair as alt
from sklearn.metrics.pairwise import cosine_similarity
from annoy import AnnoyIndex
import warnings
from sklearn.cluster import KMeans
from bertopic._ctfidf import ClassTFIDF
from sklearn.feature_extraction.text import CountVectorizer
from io import StringIO

load_dotenv()

# Paste your API key here. Remember to not share it publicly 

API_KEY = os.environ['API_KEY']
ENV=os.environ['ENV']

co = cohere.Client(API_KEY)


# df=pd.read_csv("data.csv")
# file = open("embeds.obj",'rb')
# file.close()

def reducer(embeds):
    reducer = umap.UMAP(n_neighbors=100) 
    umap_embeds = reducer.fit_transform(embeds)
    print(umap_embeds)
    return(umap_embeds)

def keywords(df,embeds,n_clusters, column):
    kmeans_model = KMeans(n_clusters=n_clusters, random_state=0)
    classes = kmeans_model.fit_predict(embeds)
    # Extract the keywords for each cluster
    documents =  df[column]
    documents = pd.DataFrame({"Document": documents,
                            "ID": range(len(documents)),
                            "Topic": None})
    documents['Topic'] = classes
    documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
    count_vectorizer = CountVectorizer(stop_words="english").fit(documents_per_topic.Document)
    count = count_vectorizer.transform(documents_per_topic.Document)
    words = count_vectorizer.get_feature_names()
    ctfidf = ClassTFIDF().fit_transform(count).toarray()
    words_per_class = {label: [words[index] for index in ctfidf[label].argsort()[-10:]] for label in documents_per_topic.Topic}
    df['cluster'] = classes
    df['keywords'] = df['cluster'].map(lambda topic_num: ", ".join(np.array(words_per_class[topic_num])[:]))
    return(df)

def chart(df, n_clusters,column):
    st.write("plotting")
    selection = alt.selection_multi(fields=['keywords'], bind='legend')

    chart = alt.Chart(df).transform_calculate(
        url='https://news.ycombinator.com/item?id=' + alt.datum.id
    ).mark_circle(size=60, stroke='#666', strokeWidth=1, opacity=0.3).encode(
        x=#'x',
        alt.X('x',
            scale=alt.Scale(zero=False),
            axis=alt.Axis(labels=False, ticks=False, domain=False)
        ),
        y=
        alt.Y('y',
            scale=alt.Scale(zero=False),
            axis=alt.Axis(labels=False, ticks=False, domain=False)
        ),
        href='url:N',
        color=alt.Color('keywords:N', 
                        legend=alt.Legend(columns=1, symbolLimit=0, labelFontSize=14)
                    ),
        opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
        tooltip=[column, 'keywords', 'cluster','Role']
    ).properties(
        width=800,
        height=500
    ).add_selection(
        selection
    ).configure_legend(labelLimit= 0).configure_view(
        strokeWidth=0
    ).configure(background="#FDF7F0").properties(
        title=column
    )

    st.altair_chart(chart,use_container_width=False)

def embeddings(df,column):
    # Get text embeddings via the Embed endpoint
    embeds = []
    for chat in df[column]:
        output = co.embed(
                    model='small',
                    texts=[chat])
        embeds += output.embeddings
    return(embeds)
  



st.write("# co:cluster")


uploaded_file = st.file_uploader("Upload a CSV")
if uploaded_file is not None:
    # Can be used wherever a "file-like" object is accepted:
    df = pd.read_csv(uploaded_file)
    st.write(df)
    with st.form("my_form"):
        column=st.selectbox("Choose Column to Embed",options=df.columns)
        st.session_state['column']=column
        submitted = st.form_submit_button("Submit")
        if submitted:
            st.write("embedding your data......")
            embeds=embeddings(df,column)
            st.write("your data has been embeded!")
            reducer = umap.UMAP(n_neighbors=100) 
            umap_embeds = reducer.fit_transform(embeds)
            df['x'] = umap_embeds[:,0]
            df['y'] = umap_embeds[:,1]
            st.session_state['df']=df
            st.session_state['embeds']=embeds

clusters= st.slider("Cluster", min_value=0, max_value=50)
if (clusters != 0):
    df=st.session_state['df']
    embeds=st.session_state['embeds']
    column=st.session_state['column']
    keywords(df,embeds,clusters,column)
    chart(df,clusters,column)