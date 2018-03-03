from flask import Flask, flash, redirect, render_template, request, session, abort
# importing the requests library
import requests
import requests.auth
# Import the json library
import json
import pandas as pd
import urllib
import numpy as np
# Import Beautiful Soup
from bs4 import BeautifulSoup
import re
import config
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.chunk import ne_chunk
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
import random

# import clustered trailers from pickle
pickle_off = open("finalclustereddf.pickle","rb")
clustereddf = pickle.load(pickle_off)

# import kmeans fit from pickle
pickle_off = open("kmeansfit.pickle","rb")
ogkmeans = pickle.load(pickle_off)

def timewarp(trailer, totaltime, intervaltime):
    points = []
    duration = trailer['durationseconds']
    warper = (duration/totaltime)

    timeinterval = totaltime/intervaltime
    intervallist = list(np.linspace(start=0.0, stop=totaltime, num=(timeinterval+1.0)))
    bucketlist = []
    for first, second in zip(intervallist, intervallist[1:]):
        bucketlist.append((first, second))
    for bucket in bucketlist:
        bucket_sentiments = []
        for sentence in trailer['sentences']:
            if (bucket[0] < round(float(sentence['starttime'])/warper,2) <= bucket[1]):
                bucket_sentiments.append(sentence['sentimentVader'])
        if bucket[0] == 0.0:
            points.append((bucket[0],0.0))
        elif bucket_sentiments:
            points.append((bucket[0],np.average(bucket_sentiments)))
        else:
            points.append((bucket[0],0.0))
    points.append((totaltime,points[-1][1]))
    return points

def gettranscriptandytdetails(youtube_id, ogkmeans):
    d = {}
    youtube_url = "https://www.youtube.com/watch?v=" + youtube_id
    d['youtube_url'] = youtube_url
    transcripturl = "http://video.google.com/timedtext?lang=en&v=" + (youtube_id)
    response = requests.get(transcripturl)
    page = response.text
    souped_page = BeautifulSoup(page,"html5lib")

    if souped_page.find("transcript"):
        transcripthtml = souped_page.find("transcript")
        sentences = souped_page.find_all("text")
        transcripttext = []
        durationtime = []
        for sentence in sentences:
            transcripttext.append(sentence.text.replace('&#39;',"'").replace("\n"," ").replace("... ", "...").replace(".. ", "..").replace("&quot;", '').replace("<i>","").replace("</i>",""))
        transcripttext = " ".join(transcripttext)
    else:
        transcripttext = np.nan
    d['transcripttext'] = transcripttext
    d['id'] = youtube_id

    try:
        d['sentences'] = []
        for index, onesentence in enumerate(sentences):
            sentence = {}
            sentence['id'] = index
            sentence['duration'] = onesentence['dur']
            sentence['starttime'] = onesentence['start']
            sentence['text'] = onesentence.text.replace("&#39;","'").replace("\n"," ").replace("<i>","").replace("</i>","")
            d['sentences'].append(sentence)
    except:
        pass

    apikey = config.api_key
    # Get youtube api info
    base_url = "https://www.googleapis.com/youtube/v3/"
    videoid = youtube_id
    snippeturl = base_url+"videos?part=snippet&id="+videoid+"&key="+apikey
    snippetreq = requests.get(snippeturl).json()
    try:
        d["audio_lang"] = snippetreq['items'][0]['snippet']['defaultAudioLanguage']
        d["description"] = snippetreq['items'][0]['snippet']['description']
        d["title"] = snippetreq['items'][0]['snippet']['title']
        d["tags"] = snippetreq['items'][0]['snippet']['tags']
    except:
        pass
    try:
        statsurl = base_url+"videos?part=statistics&id="+videoid+"&key="+apikey
        statsreq = requests.get(statsurl).json()
        d["commentCount"] = statsreq['items'][0]['statistics']['commentCount']
        d["dislikeCount"] = statsreq['items'][0]['statistics']['dislikeCount']
        d["likeCount"] = statsreq['items'][0]['statistics']['likeCount']
        d["favoriteCount"] = statsreq['items'][0]['statistics']['favoriteCount']
        d["viewCount"] = statsreq['items'][0]['statistics']['viewCount']
    except:
        pass
    try:
        contentdetailsurl = base_url+"videos?part=contentDetails&id="+videoid+"&key="+apikey
        contentreq=requests.get(contentdetailsurl).json()
        d["caption"] = contentreq['items'][0]['contentDetails']['caption']
        d["duration"] = contentreq['items'][0]['contentDetails']['duration']
    except:
        pass

    # Get sentence Vader sentiment
    analyzer = SentimentIntensityAnalyzer()
    speechdurations = []
    try:
        for sentence in d['sentences']:
            speechdurations.append(float(sentence['duration']))
            sentence['sentimentVader'] = analyzer.polarity_scores(sentence['text'])['compound']
    except:
        pass

    # calculating total trailer time in seconds
    if "M" in d['duration']:
        if "S" in d['duration']:
            timearr = (d['duration'].split('PT')[1].replace('S','').split('M'))
            totalduration = (float(timearr[0])*60)+float(timearr[1])
            d['durationseconds'] = totalduration
        else:
            totalduration = float(d['duration'].split('PT')[1].split('M')[0])
            d['durationseconds'] = totalduration*60
    else:
        totalduration = float(vid['duration'].split('PT')[1].split('S')[0])
        d['durationseconds'] = totalduration

    # Getting speechtime
    d['speechduration'] = np.sum(speechdurations)
    d['speechtime'] = float(d['speechduration']/d['durationseconds'])

    # Getting word counts by trailer
    text = TextBlob(d['transcripttext'])
    d['wordcount'] = len(text.words)

    # Warping trailer
    warpedpoints = timewarp(d, 120, 15)
    d['normalizedsentimentpoints'] = warpedpoints
    d['cluster'] = ogkmeans.predict([[x[1] for x in d['normalizedsentimentpoints']]])[0]
    return d

def getsimilarvids(clustereddf, trailer):
    apikey = config.api_key
    newtrailerdata = [trailer['youtube_url'],trailer['wordcount'],trailer['speechtime'],trailer['durationseconds'],trailer['cluster']]
    updateddf = clustereddf.append(pd.Series(newtrailerdata, index=clustereddf.columns), ignore_index=True)
    X = updateddf.iloc[:, 1:]
    scaledX = scale(X)
    scaleddf = pd.DataFrame(scaledX)
    newkmeansfit = KMeans(n_clusters=4, random_state=0).fit(scaleddf)
    updateddf['finalcluster'] = list(newkmeansfit.labels_)
    filtereddf = updateddf[updateddf['finalcluster'] == trailer['cluster']]
    og_yt_url = updateddf.iloc[808,0]
    youtube_urls = list(filtereddf['youtube_url'])
    try:
        youtube_urls.remove(og_yt_url)
    except:
        pass
    list_of_vids = random.sample(youtube_urls, 3)
    list_of_titles = []
    for vid in list_of_vids:
    # Get youtube api info
        base_url = "https://www.googleapis.com/youtube/v3/"
        yt_id = vid.rsplit('?v=',1)[1]
        snippeturl = base_url+"videos?part=snippet&id="+yt_id+"&key="+apikey
        snippetreq = requests.get(snippeturl).json()
        list_of_titles.append(snippetreq['items'][0]['snippet']['title'])
    return list_of_vids, list_of_titles

app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def welcome():
    if request.method == 'POST':
        return redirect(url("/trailerreel"))
    else:
        return render_template('/index.html')

@app.route("/trailerreel", methods=['GET','POST'])
def watchtrailer():
    yt_id = request.args.get('yturl').rsplit('?v=',1)[1]
    yt_url = request.args.get('yturl')
    yt_embed = "https://www.youtube.com/embed/" + yt_id + "?autoplay=1"
    trailerdetails = gettranscriptandytdetails(yt_id, ogkmeans)
    list_of_ytvidurls, list_of_titles = getsimilarvids(clustereddf, trailerdetails)
    indices = list(range(0,len(list_of_ytvidurls)))

    return render_template("/trailerreel.html", yt_url = yt_url, yt_embed = yt_embed, trailerdetails = trailerdetails, list_of_titles = list_of_titles, list_of_ytvidurls = list_of_ytvidurls, indices = indices)


if __name__ == "__main__":
    app.run(debug=True)
