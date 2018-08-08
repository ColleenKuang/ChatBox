# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings(action='ignore',category=UserWarning,module='gensim')
from flask import Flask, abort, request, render_template
from flask_cors import CORS
from gensim import corpora, similarities, models
import jieba
import json
import os, sys, string
import mysql.connector
from statistics import mode


app = Flask(__name__)
CORS(app)

def getAnswerText(id):
    conn = mysql.connector.connect(user='root', password='', host='localhost', database='xiaoyan', charset='utf8')
    cursor = conn.cursor()
    cursor.execute("SELECT answer_text FROM answer WHERE answer_id = %s", [id])
    data = cursor.fetchone()
    cursor.close()
    conn.close()
    return data[0]

def getAnswerID(id):
    conn = mysql.connector.connect(user='root', password='', host='localhost', database='xiaoyan', charset='utf8')
    cursor = conn.cursor()
    #sql = "SELECT question_answer_id FROM question WHERE question_id = %s"
    cursor.execute("SELECT question_answer_id FROM question WHERE question_id = %s", [id])
    data = cursor.fetchone()
    cursor.close()
    conn.close()
    return data[0]

def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords
def movestopwords(sentence):
    stopwords = stopwordslist('dictionary/stopwords.txt')
    outstr = ''
    for word in sentence:
        if word not in stopwords:
            if word != '\t'and'\n':
                outstr += word
                # outstr += " "
    return outstr

@app.route('/getResult', methods=['POST'])
def getResult():
    dictionary = corpora.Dictionary.load('dictionary/mydict.dic')
    corpus = corpora.MmCorpus('model/lsi/lsi_corpus.mm')
    model = models.LsiModel.load('model/lsi/model.lsi')
    index = similarities.MatrixSimilarity(corpus)
    index.save('similarity/lsi_similarity.sim')
    document =  request.form['question'];
    print (document)
    jieba.add_word('交换生') #强制添加名词‘交换生’
    jieba.suggest_freq(('香港', '浸会'), True) #强制拆分‘香港浸会大学’为 ‘香港’ ‘浸会’ ‘大学’
    jieba.suggest_freq(('香港', '浸会大学'), True) #强制拆分‘香港浸会大学’为 ‘香港’ ‘浸会’ ‘大学’
    jieba.suggest_freq(('浸会', '大学'), True) #强制拆分‘香港浸会大学’为 ‘香港’ ‘浸会’ ‘大学’
    print (jieba.lcut(movestopwords(jieba.lcut(document.replace(" ", "")))))
    bow_vec = dictionary.doc2bow(jieba.lcut(movestopwords(jieba.lcut(document.replace(" ", "")))))
    lsi_vec = model[bow_vec]
    sims = index[lsi_vec]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    print (sims)
    high_confidence_result = []
    middle_confidence_result = []
    #for sim in sims:
    #    if sim[1] > 0.9:
    #        high_confidence_result.append(sim[0] + 1)
    #    elif sim[1] < 0.9 and sim[1] > 0.8:
    #        middle_confidence_result.append(sim[0] + 1)
    for sim in sims[0:1]:
        if sim[1] != 0:
            high_confidence_result.append(sim[0] + 1)
    print (high_confidence_result)
    answer = []
    moddle_answer = []
    for id in high_confidence_result:
        answer.append(getAnswerID(str(id)))

    for id in middle_confidence_result:
        moddle_answer.append(getAnswerID(str(id)))

    mode_id = mode(answer)
    print (mode_id)
    #return json.dumps({'message': 'Success', 'high_confidence': high_confidence_result, 'middle_confidence': middle_confidence_result, 'low_confidence': low_confidence_result})
    return json.dumps(getAnswerText(mode_id), ensure_ascii = False)




if __name__ =='__main__':
    app.run(host="127.0.0.1", port=5000)
