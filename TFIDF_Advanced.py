#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 22:19:25 2020

This tutorial codes are modifiled from:
@author: https://github.com/gearmonkey/tfidf-python/blob/master/tfidf.py
"""


#!/usr/bin/env python
# 
# Copyright 2009  Niniane Wang (niniane@gmail.com)
# Reviewed by Alex Mendes da Costa.
#
# Modified in 2012 by Benjamin Fields (me@benfields.net)
#
# This is a simple Tf-idf library.  The algorithm is described in
#   http://en.wikipedia.org/wiki/Tf-idf
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 3 of the License, or (at your option) any later version.
#
# Tfidf is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details:
#
#   http://www.gnu.org/licenses/lgpl.txt

__author__ = "Niniane Wang"
__email__ = "niniane at gmail dot com"

import math
import re
import codecs
from operator import itemgetter

class TfIdf:

  """Tf-idf class implementing http://en.wikipedia.org/wiki/Tf-idf.
  
     The library constructs an IDF corpus and stopword list either from
     documents specified by the client, or by reading from input files.  It
     computes IDF for a specified term based on the corpus, or generates
     keywords ordered by tf-idf for a specified document.
  """

  def __init__(self, corpus_filename = None, stopword_filename = None,
               DEFAULT_IDF = 1.5):
    """Initialize the idf dictionary.  
    
       If a corpus file is supplied, reads the idf dictionary from it, in the
       format of:
         # of total documents
         term: # of documents containing the term
       If a stopword file is specified, reads the stopword list from it, in
       the format of one stopword per line.
       The DEFAULT_IDF value is returned when a query term is not found in the
       idf corpus.
    """
    self.num_docs = 0
    self.term_num_docs = {}     # term : num_docs_containing_term
    self.stopwords = set([])
    self.idf_default = DEFAULT_IDF

    if corpus_filename:
        self.merge_corpus_document(corpus_filename)

    if stopword_filename:
      stopword_file = codecs.open(stopword_filename, "r", encoding='utf-8')
      self.stopwords = set([line.strip() for line in stopword_file])

  def get_tokens(self, str):
    """Break a string into tokens, preserving URL tags as an entire token.
       This implementation does not preserve case.  
       Clients may wish to override this behavior with their own tokenization.
    """
    return re.findall(r"<a.*?/a>|<[^\>]*>|[\w'@#]+", str.lower())

  def merge_corpus_document(self, corpus_filename):
    """slurp in a corpus document, adding it to the existing corpus model
    """
    corpus_file = codecs.open(corpus_filename, "r", encoding='utf-8')

    # Load number of documents.
    line = corpus_file.readline()
    self.num_docs += int(line.strip())

    # Reads "term:frequency" from each subsequent line in the file.
    for line in corpus_file:
      tokens = line.rsplit(":",1)
      #print(tokens)
      #['i', '10\n']
      term = tokens[0].strip()
      try:
          frequency = int(tokens[1].strip())
      except IndexError as err:
          if line in ("","\t"):
              #catch blank lines
              print("line is blank")
              continue
          else:
              raise
      if self.term_num_docs.__contains__(term):
        self.term_num_docs[term] += frequency
      else:
        self.term_num_docs[term] = frequency

  # 添加字符串文本加入Document。
  # usage: p.add_input_document("I also love statistics.")
  def add_input_document(self, input):
    """Add terms in the specified document to the idf dictionary."""
    self.num_docs += 1
    words = set(self.get_tokens(input))
    for word in words:
      if word in self.term_num_docs:
        self.term_num_docs[word] += 1
      else:
        self.term_num_docs[word] = 1
        

  # 添加指定文件，加入Document
  # Usage: p.add_plaintext_document('plaintext.txt')      
  def add_plaintext_document(self, plaintext_filename):
      """Add terms in the specified file to the idf dictionary.
      """
      self.num_docs += 1
      input_file = codecs.open(plaintext_filename, "r", encoding='utf-8')
      allline=""
      for line in input_file:
          allline=allline + line.strip()  
          
      tokens = self.get_tokens(allline.strip())
      tokens_set = set(tokens)
      for word in tokens_set:
          if self.term_num_docs.__contains__(word):
              self.term_num_docs[word] += 1
          else:
              self.term_num_docs[word] = 1
              

  def save_corpus_to_file(self, idf_filename, stopword_filename,
                          STOPWORD_PERCENTAGE_THRESHOLD = 0.01):
    """Save the idf dictionary and stopword list to the specified file."""
    output_file = codecs.open(idf_filename, "w", encoding='utf-8')

    output_file.write(str(self.num_docs) + "\n")
    for term, num_docs in self.term_num_docs.items():
      output_file.write(term + ": " + str(num_docs) + "\n")

    sorted_terms = sorted(self.term_num_docs.items(), key=itemgetter(1),
                          reverse=True)
    stopword_file = open(stopword_filename, "w")
    for term, num_docs in sorted_terms:
      if num_docs < STOPWORD_PERCENTAGE_THRESHOLD * self.num_docs:
        break

      stopword_file.write(term + "\n")

  def get_num_docs(self):
    """Return the total number of documents in the IDF corpus."""
    return self.num_docs

  def get_idf(self, term):
    """Retrieve the IDF for the specified term. 
    
       This is computed by taking the logarithm of ( 
       (number of documents in corpus) divided by (number of documents
        containing this term) ).
     """
    if term in self.stopwords:
      return 0

    if not term in self.term_num_docs:
      return self.idf_default

    return math.log(float(1 + self.get_num_docs()) / 
      (1 + self.term_num_docs[term]))

  # 计算当前“字符串”文档的TFIDF（只计算出现的Term的TFIDF，没有出现的Term对应的TFIDF为0。）
  # Usage: p.get_str_keywords("I like math and chemistry very much.")
  def get_str_keywords(self, curr_doc):
    """Retrieve terms and corresponding tf-idf for the specified document.
       The returned terms are ordered by decreasing tf-idf.
    """
    tfidf = {}
    tokens = self.get_tokens(curr_doc)
    tokens_set = set(tokens)
    for word in tokens_set:
      mytf = float(tokens.count(word)) / len(tokens_set)
      myidf = self.get_idf(word)
      tfidf[word] = mytf * myidf
#    return tfidf
    return sorted(tfidf.items(), key=itemgetter(1), reverse=True)

  
  # 请在该类中增加新的方法，用以计算某本地文档的TFIDF
  # Usage: p.get_doc_keywords('plaintext_TFIDF_comp.txt')
  '''
  def get_doc_keywords(self, curr_doc)

  '''





### Let's test all the methods in the following steps.
            
print("\n########################################################################################")                
print("######################## Here is the experiment !#######################################")                
print("########################################################################################")                

print("\n####################\n We are now building corpus.\n")
p=TfIdf(corpus_filename = 'corpus_initial.txt', stopword_filename = 'teststopwords.txt',
               DEFAULT_IDF = 1.5)
print("We imported ", p.num_docs, " docs from 'corpus_initial.txt'." )
print("The Term_num_docs are", p.term_num_docs)
print("The stopwords include:", p.stopwords)
print('p.get_idf("i"):', p.get_idf("i"), 'p.get_idf("bioinformatics"):', p.get_idf("bioinformatics") )


print("\n####################\n We are trying to add a new corpus in the system.\n")
p.merge_corpus_document("corpus_added.txt")
print("We imported ", p.num_docs, " docs by adding 'corpus_added.txt'. ")
print("The Term_num_docs are", p.term_num_docs)
print("The stopwords include:", p.stopwords)
print('p.get_idf("i"):', p.get_idf("i"), 'p.get_idf("bioinformatics"):', p.get_idf("bioinformatics") )


print("\n####################\n We are trying to add an input sentence in the system.")
p.add_input_document("I also love statistics.")
print("We imported ", p.num_docs, " docs by adding an input sentence 'I also love statistics'. ")
print("The Term_num_docs are", p.term_num_docs)
print("The stopwords include:", p.stopwords)
print('p.get_idf("i"):', p.get_idf("i"), 'p.get_idf("bioinformatics"):', p.get_idf("bioinformatics") )


print("\n####################\n We are trying to add a local document in the system.\n")
p.add_plaintext_document('plaintext_added.txt')
print("We imported ", p.num_docs, " docs by adding the plain text, 'plantext_added.txt'. ")
print("The Term_num_docs are", p.term_num_docs)
print("The stopwords include:", p.stopwords)
print('p.get_idf("i"):', p.get_idf("i"), 'p.get_idf("bioinformatics"):', p.get_idf("bioinformatics") )



print("\n####################\n The idf computation is over, and the corpus construction and management is over. \n")
print("\n####################\n We are showing the TF*IDF computation next.\n")
teststr="I like math and chemistry so much."
print(" For the input sentence '", teststr ,"' being a document, ")
out=p.get_str_keywords("I like math and chemistry so much.\n")
print("TFIDF is:", out)

print("\n########################################################################################")                
print("########################         Assignment!     #######################################")                
print("########################################################################################")                

print("\nFor the input plain text file 'plaintext_TFIDF_comp.txt', please compute the TFIDF.\nPlease start coding from line 200. \n")








p.get_tokens("Trump will be abandoned by the public. Bio-NLP is cute. BRCA1/2")

