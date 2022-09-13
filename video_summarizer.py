import wave
import contextlib
import speech_recognition as sr
from moviepy.editor import *
import streamlit as st
import numpy as np
import math
import string
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
from nltk.cluster.util import cosine_distance
from summarizer import Summarizer, TransformerSummarizer
import networkx as nx
# import sys
# sys.path.remove('/home/appuser/venv/bin')
from punctuator import Punctuator

p = Punctuator('Demo-Europarl-EN.pcl')

def Transcribe(video_file):
    audio_file = "audio_file.wav"
    audioclip = AudioFileClip(video_file)
    audioclip.write_audiofile(audio_file)

    with contextlib.closing(wave.open(audio_file,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)

    total_duration = math.ceil(duration / 60)
    r = sr.Recognizer()
    transcription = ""
    for i in range(0, total_duration):
        with sr.AudioFile(audio_file) as source:
            audio = r.record(source, offset=i*60, duration=60)
        transcription += r.recognize_google(audio) + " "

    output = p.punctuate(transcription)
    return output

def WordOccurrences(text, sws):
    ps = PorterStemmer()
    word_freq = {}
    total_words = 0
    for word in text:
        word = ps.stem(word)
        #
        if word in sws or word in string.punctuation:
            continue
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1
        total_words += 1
    return word_freq, total_words

def AjiValue(sentence, myword, wordFreq, content_words):
    ps = PorterStemmer()
    fji = 0
    # sentence = sentence.lower()
    sentence = word_tokenize(sentence)
    for word in sentence:
        if ps.stem(word.lower()) == myword:
            fji += 1
    # Fa = len(wordFreq)
    Fa = content_words
    Fw = wordFreq[myword]
    icf = math.log(Fa/Fw)
    return fji * icf

def CreateA_Matrix(sentences, wordFreq, cw):
    A_Matrix = []
    for word in list(wordFreq.keys()):
        row = []
        for sentence in sentences:
            row.append(AjiValue(sentence, word, wordFreq, cw))
        A_Matrix.append(row)
    return A_Matrix

def FindAverage(a):
    sum = 0
    for num in a:
        sum += num
    return sum / len(a)

def DeleteBelowAvg(vt, avg):
    for i in range(vt.shape[0]):
        for j in range(vt.shape[1]):
            sent_score = vt[i][j]
            if sent_score < avg[i]:
                vt[i][j] = 0
    return vt

def FindSentenceLengths(amp_vt):
    sent_len = []
    for j in range(amp_vt.shape[1]):
        # for each column => sentence
        sum = 0
        for i in range(amp_vt.shape[0]):
            # for every value => row of each sentence
            # sum += amp_vt[i][j] ** 2
            sum += amp_vt[i][j]
        # sent_len.append(math.sqrt(sum))
        sent_len.append(sum)
    return sent_len

def CreateConceptMatrix(vt):
    cXc = [[0 for x in range(vt.shape[0])] for y in range(vt.shape[0])]
    for i in range(vt.shape[0]):
        # first concept
        for j in range(vt.shape[0]):
            # second concept
            for k in range(vt.shape[1]):
                # for each sentence
                if vt[i][k] != 0 and vt[j][k] != 0:
                    # if the two concepts are the same concept don't add twice
                    if i == j:
                        cXc[i][j] += vt[i][k]
                    else:
                        cXc[i][j] += (vt[i][k] + vt[j][k])
    return cXc

def FindConceptStrengths(cXc):
    # returns a 1-dim array of each concept's strength (sum of pair value from each other concept)
    c_stren = []
    for i in range(len(cXc)):
        # for every concept (row)
        current_con_stren = 0
        for j in range(len(cXc[i])):
            # and for every concept paired with that previous concept (col)
            current_con_stren += cXc[i][j]
        c_stren.append(current_con_stren)
    return c_stren

def CrossMethod(N, sentences, wordFreq, content_words):
    A_mat = CreateA_Matrix(sentences, wordFreq, content_words)
    u, s_1, vh = np.linalg.svd(A_mat, full_matrices=True)
    s = np.diag(s_1)
    vt = vh
    # find the average value of the each row => of each concept
    avg = []
    for i in range(vt.shape[0]):
        avg.append(FindAverage(vt[i]))

    # delete the sentences that have a score lower than the concept's average
    vt = DeleteBelowAvg(vt, avg)

    # multiply s matrix with vt so that the score of each sentence in each concept is amplified
    # the sentences in the first concept will be the amplified the most as the concepts are in decreasing order of
    # importance
    amp_vt = np.matmul(s, vt)

    # find the length of each sentence
    # go throught the sentences and find the root of the square sum of each sentence's score from each concept
    # therefore, traverse column by column and find length using the values of the rows
    sent_len = FindSentenceLengths(amp_vt)

    # find the top N sentences with the largest length as calculated above
    top_N_sent = sorted(range(len(sent_len)), key=lambda i: sent_len[i], reverse=True)[:N]
    # heapq.nlargest(N, sent_len, key=None)

    # find those sentences from the original sentence matrix and return them
    sentence_selection = []
    for i in range(len(top_N_sent)):
        sentence_selection.append(sentences[top_N_sent[i]])

    indices = {}
    for i in range(len(sentence_selection)):
        indices[i] = sentences.index(sentence_selection[i])

    ordered_selection = []
    for i in range(len(indices)):
        value = min(indices.values())
        key = list(indices.keys())[list(indices.values()).index(value)]
        ordered_selection.append(sentence_selection[key])
        indices[key] = len(sentences) * 2

    return sentence_selection

def TopicMethod(N, sentences, wordFreq, content_words):
    A_mat = CreateA_Matrix(sentences, wordFreq, content_words)
    u, s_1, vh = np.linalg.svd(A_mat, full_matrices=True)
    s = np.diag(s_1)
    vt = vh

    # find the average value of the each row => of each concept
    avg = []
    for i in range(vt.shape[0]):
        avg.append(FindAverage(vt[i]))

    # delete the sentences that have a score lower than the concept's average
    vt = DeleteBelowAvg(vt, avg)

    # multiply s matrix with vt so that the score of each sentence in each concept is amplified
    # the sentences in the first concept will be the amplified the most as the concepts are in decreasing order of
    # importance
    amp_vt = np.matmul(s, vt)

    # create a concept X concept matrix showing total score of the common sentences between two concepts
    con_x_con = CreateConceptMatrix(amp_vt)

    # calculate each concept's strength
    con_strength = FindConceptStrengths(con_x_con)

    # sort the concpets from strongest to weakest
    concepts_from_strongest = sorted(range(len(con_strength)), key=lambda i: con_strength[i], reverse=True)

    con_index = 0
    sentence_selection = []
    # find N sentences for our summary
    for i in range(N):
        # if we went through all concepts, start again from the beginning and pick the next best sentence
        if con_index >= len(concepts_from_strongest):
            con_index = 0
        # returns the sentence with highest score for that concept
        sent_score = max(amp_vt[concepts_from_strongest[con_index]])
        # gets the index of the sentence with that score
        # sent_index = np.where(amp_vt[concepts_from_strongest[con_index]] == sent_score)
        sent_index = list(amp_vt[concepts_from_strongest[con_index]]).index(sent_score)
        # sent_index = amp_vt[con_index].index(sent_score)
        # put the sentence in our selection for summary
        sentence_selection.append(sentences[sent_index])
        # make it zero so that the next time it'll pick the second best sentence
        amp_vt[concepts_from_strongest[con_index]][sent_index] = 0
        # proceed to next concept
        con_index += 1
    return sentence_selection

def Bert(text, N, percentage):
    bert_model = Summarizer()
    # return ''.join(bert_model(text, num_sentences=N, max_length=avg_chars_per_sent))
    return ''.join(bert_model(text, num_sentences=N))

def GPT2(text, N, percentage):
    GPT2_model = TransformerSummarizer(transformer_type="GPT2", transformer_model_key="gpt2-medium")
    # return ''.join(GPT2_model(text, num_sentences=N, max_length=avg_chars_per_sent))
    return ''.join(GPT2_model(text, num_sentences=N))

def XLNet(text, N, percentage):
    model = TransformerSummarizer(transformer_type="XLNet", transformer_model_key="xlnet-base-cased")
    # return ''.join(model(text, num_sentences=N, max_length=avg_chars_per_sent))
    return ''.join(model(text, num_sentences=N))

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
    all_words = list(set(sent1 + sent2))
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    for w in sent1:
        if w not in stopwords and w not in string.punctuation:
            vector1[all_words.index(w)] += 1
    for w in sent2:
        if w not in stopwords and w not in string.punctuation:
            vector2[all_words.index(w)] += 1
    return 1 - cosine_distance(vector1, vector2)

def build_similarity_matrix(sentences, stop_words):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 != idx2:
                similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)
    return similarity_matrix

def TextRank(sentences, N, num_of_chars_original, percentage):
    char_goal = num_of_chars_original * percentage
    sentence_similarity_matrix = build_similarity_matrix(sentences, stopwords.words("english"))
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    try:
        scores = nx.pagerank(sentence_similarity_graph, tol=1e-15, max_iter=10000)
    except:
        scores = nx.pagerank_numpy(sentence_similarity_graph)
    ranked_sentence = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    sentence_selection = []
    char_count = 0
    sent_index = 0
    while char_count <= char_goal:
        sentence_selection.append(ranked_sentence[sent_index][1])
        char_count += len(ranked_sentence[sent_index][1])
        if abs(char_count - char_goal) < abs(char_count + len(ranked_sentence[sent_index+1][1]) - char_goal):
            break
        sent_index += 1

    indices = {}
    for i in range(len(sentence_selection)):
        indices[i] = sentences.index(sentence_selection[i])

    ordered_selection = []
    for i in range(len(indices)):
        value = min(indices.values())
        key = list(indices.keys())[list(indices.values()).index(value)]
        ordered_selection.append(sentence_selection[key])
        indices[key] = len(sentences)*2

    # for i in range(N):
    #     sentence_selection.append(ranked_sentence[i][1])
    return ordered_selection


def Summarize(transcript, txtrank, bert, gpt2, xlnet, cross, topic, perc):
    print("percentage = ", perc)
    nltk.download('punkt')
    nltk.download("stopwords")
    ps = PorterStemmer()
    num_of_chars_original = len(transcript)
    sentences = sent_tokenize(transcript)
    num_of_sent_original = len(sentences)
    sentences_lower = sent_tokenize(transcript.lower())
    words = word_tokenize(transcript.lower())
    num_of_words_original = len(words)
    avg_chars_per_sent = math.ceil(num_of_chars_original / num_of_sent_original)
    wordFreq, content_words = WordOccurrences(sorted(words), stopwords.words('english'))
    percentage = perc/100
    n = math.ceil(percentage*len(sentences))
    if txtrank:
        if 'tk' not in st.session_state:
            st.session_state.tk = " ".join(TextRank(sentences, n, num_of_chars_original, percentage))
        st.subheader("TextRank")
        st.write(st.session_state.tk)
    if bert:
        if 'bt' not in st.session_state:
            st.session_state.bt = Bert(transcript, n, percentage)
        st.subheader("BERT model")
        st.write(st.session_state.bt)
    if gpt2:
        if 'g2' not in st.session_state:
            st.session_state.g2 = GPT2(transcript, n, percentage)
        st.subheader("GPT-2 model")
        st.write(st.session_state.g2)
    if xlnet:
        if 'xl' not in st.session_state:
            st.session_state.xl = XLNet(transcript, n, percentage)
        st.subheader("XLNet model")
        st.write(st.session_state.xl)
    if cross:
        if 'cr' not in st.session_state:
            st.session_state.cr = " ".join(CrossMethod(n, sentences, wordFreq, content_words))
        st.subheader("LSA using Cross method")
        st.write(st.session_state.cr)
    if topic:
        if 'to' not in st.session_state:
            st.session_state.to = " ".join(TopicMethod(n, sentences, wordFreq, content_words))
        st.subheader("LSA using Topic method")
        st.write(st.session_state.to)




def main():
    st.header("Investigating Video Summarization using Speech Recognition")
    st.subheader("A video summariztion system as part of the MSc in A.I Final Project")
    st.markdown("**Student: Loizos Siakallis, Supervisor: Dr Mike Wald**")
    uploaded_file = st.file_uploader("Please upload a video (.mp4):")
    if uploaded_file is not None:
        # for key in st.session_state.keys():
        #     del st.session_state[key]

        f = open(uploaded_file.name, "wb")
        f.write(uploaded_file.getvalue())
        f.close()
        if 'tr' not in st.session_state:
            st.session_state.tr = Transcribe(uploaded_file.name)
        st.subheader("Transcript")
        st.write(st.session_state.tr)

        with st.form(key='method'):
            perc = st.slider('Provide summariztion ratio:', min_value=0, max_value=100)
            st.text("Select Summarization method/s (you can select more than one):")
            txtrank = st.checkbox('TextRank')
            bert = st.checkbox('BERT')
            gpt2 = st.checkbox('GPT-2')
            xlnet = st.checkbox('XLNET')
            cross = st.checkbox('CROSS')
            topic = st.checkbox('TOPIC')
            submitted = st.form_submit_button('Confirm Selection')
            # if submitted or st.session_state.submit_state:
            #     st.session_state.submit_state = True

        Summarize(st.session_state.tr, txtrank, bert, gpt2, xlnet, cross, topic, perc)

if __name__ == '__main__':
    main()
