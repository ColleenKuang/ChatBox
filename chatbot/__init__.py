# -*- coding: utf-8 -*-
import os, sys, string
import os.path
import mysql.connector
import gensim
import jieba
from gensim import corpora, similarities, models

conn = mysql.connector.connect(user='root', password='', host='localhost', database='xiaoyan', charset='utf8')

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
    return outstr

cursor = conn.cursor()
documents = []
sql = "SELECT question_text FROM question"
cursor.execute(sql)
alldata = cursor.fetchall()
if alldata:
    for rec in alldata:
        documents.extend(rec)
cursor.close()
conn.close()
jieba.add_word('交换生')
jieba.suggest_freq(('香港', '浸会'), True)
jieba.suggest_freq(('香港', '浸会大学'), True)
jieba.suggest_freq(('浸会', '大学'), True)
texts = [jieba.lcut(document) for document in documents]
seg = [movestopwords(text) for text in texts]
final = [(jieba.lcut(document)) for document in seg]
print (final)
dictionary = corpora.Dictionary(final)
dictionary.save('dictionary/mydict.dic')
corpus = [ dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('corpus/corpus.mm', corpus)
# Reload dictionary and corpus
dictionary = corpora.Dictionary.load('dictionary/mydict.dic')
corpus = corpora.MmCorpus('corpus/corpus.mm')
tfidf = models.TfidfModel(corpus=corpus)
tfidf.save('model/model.tfidf')
# Serialize corpus
tfidf_corpus = tfidf[corpus]
corpora.MmCorpus.serialize('model/tfidf_corpus.mm', tfidf_corpus)

#lsi
lsi = models.LsiModel(corpus = tfidf_corpus,id2word=dictionary,num_topics=120) #TODO
lsi_corpus = lsi[tfidf_corpus]
lsi.save('model/lsi/model.lsi')
corpora.MmCorpus.serialize('model/lsi/lsi_corpus.mm', lsi_corpus)
print ('LSI Topics:')
print (lsi.print_topics(120))

#lda
lda = models.LdaModel(corpus = tfidf_corpus,id2word=dictionary,num_topics=120)
lda_corpus = lda[tfidf_corpus]
lda.save('model/lda/model.lda')
corpora.MmCorpus.serialize('model/lda/lda_corpus.mm', lda_corpus)

index = similarities.MatrixSimilarity(corpus)
index.save('similarity/lsi_similarity.sim')
