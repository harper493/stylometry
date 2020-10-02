#!/usr/bin/python3.8

import sys
import os
import itertools
import argparse
import math
from glob import glob

DEFAULT_MATCH_NUMBER = 50
DEFAULT_THRESHOLD = '0.3'
DEFAULT_NEG_WEIGHT = { 1 : 0.03, 2:0.0, 3:0.001 }
DEFAULT_MAX_ENTRIES = 100

args = None

sentence_terminators = ( '.', '?', '!' )

class n_gram(object):

    def __init__(self, n, max_entries=DEFAULT_MAX_ENTRIES):
        self.entries = {}
        self.n = n
        self.previous = []
        self.max_entries = max_entries
        self.total_entries = 0

    def set_max_entries(self, n):
        self.max_entries = n

    def add_word(self, word, count=1):
        self.previous.append(word)
        if len(self.previous) >= self.n:
            key = tuple(self.previous)
            self.entries[key] = self.entries.get(key, 0) + count
            self.previous = self.previous[1:]
        self.total_entries += count

    def update(self, other):
        if self.n != other.n:
            raise ValueError("incompatible n-gram lengths")
        for k,n in other.entries.items():
            self.entries[k] = self.entries.get(k, 0) + n
            self.total_entries += n

    def __iter__(self):
        self.sorted = [ (k,n) for k,n in self.entries.items() ]
        self.sorted.sort(key=lambda e: e[1], reverse=True)
        for k in self.sorted[:self.max_entries]:
            yield(k[0], self.entries[k[0]])

    def __contains__(self, k):
        return k in self.entries

    def __getitem__(self, k):
        return self.entries.get(k, 0)

    def show_one(self, k):
        count = self.entries.get(k, 0)
        return f'{self.make_title(k)}  {count:5d} {100*count/self.total_entries:6.2f}%'

    def show(self, max_entries=None, tagfn=lambda k: ''):
        m = self.max_entries
        self.max_entries = max_entries or m
        result = '\n'.join([f'{tagfn(k[0])}{self.show_one(k[0])}' for k in self ])
        self.max_entries = m
        return result

    def make_title(self, k):
        return' '.join([ f'{w:20}' for w in k ])

    def compare(self, other):
        self.score = 0
        self.good = set()
        self.bad = set()
        keys = set()
        keys.update(self.entries.keys(), other.entries.keys())
        for k in keys:
            s1 = self.entries.get(k, 0)
            s2 = other.entries.get(k, 0)
            smax = max(s1, s2)
            mult = math.sqrt(smax)
            if s1*s2 == 0:
                ratio = 0
                self.bad.add(k)
            elif s1 > s2:
                ratio = s2 / s1
            else:
                ratio = s1 / s2
            if 1-ratio <= args.threshold:
                self.good.add(k)
            if ratio < 0.5:
                score = (ratio - 0.5) * DEFAULT_NEG_WEIGHT[self.n]
            else:
                score = ratio - 0.5
            score *= mult
            self.score += score
        return max(self.score, 0)

class document(object):

    def __init__(self, filename='', text=None, combine=None):
        self.filename = filename
        self.inputs = []
        self.sentences = 0
        self.sentence_avg = self.sentence_sd = self.comma_avg = self.comma_sd = 0
        self.word_count = 0
        self.n_grams = { i: n_gram(i) for i in range(1,4) }
        if filename:
            text = self.load_file()
        if text:
            self.count(text)
        elif combine:
            self.combine(*combine)

    def count(self, text):
        lines = text.split('\n')
        words = ' '.join(lines).split()
        sentence_length = 0
        sentences = 0
        comma_count = 0
        sentence_total = sentence_tsq = 0
        commas_total = commas_tsq = 0
        for ww in words:
            w = ''
            for ch in ww:
                ch = ch.lower()
                if ch.isalpha():
                    w += ch
                elif ch in sentence_terminators:
                    sentence_length += 1
                    sentences += 1
                    sentence_total += sentence_length
                    sentence_tsq += sentence_length * sentence_length
                    commas_total += comma_count
                    commas_tsq += comma_count * comma_count
                    sentence_length = 0
                    comma_count = 0
                elif ch==',':
                    comma_count += 1
            if w:
                sentence_length += 1
                self.word_count += 1
                for ng in self.n_grams.values():
                    ng.add_word(w)
        if sentences > 0:
            self.sentence_avg = sentence_total / sentences
            self.sentence_sd = self._stddev(sentences, sentence_total, sentence_tsq) / self.sentence_avg
            self.comma_avg = commas_total / sentences
            self.comma_sd = self._stddev(sentences, commas_total, commas_tsq) / self.comma_avg
                
    def combine(self, *inputs):
        self.inputs += inputs
        for d in inputs:
            for me, other in zip(self.n_grams.values(), d.n_grams.values()):
                me.update(other)

    def compare(self, other, word_threshold, bigram_threshold):
        return [ ng.compare(ong) for ng, ong in zip(self.n_grams.values(), other.n_grams.values()) ]
            
    def load_file(self):
        try :
            with open(self.filename, encoding='utf-8') as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(self.filename, encoding='latin-1') as f:
                text = f.read()
        return text

    def make_metrics(self, threshold):
        r = self.inputs[0].compare(self.inputs[1], word_threshold=threshold, bigram_threshold=threshold)
        return r

    def __str__(self):
        return f'{self.filename:50} words : {self.word_count:5}  ' + \
            f'avg sentence : {self.sentence_avg:5.1f} sd { self.sentence_sd:5.2f}  ' + \
            f'commas/sentence : {self.comma_avg:5.2f} sd { self.comma_sd:5.2f}'

    def _make_total(self, which):
        counted = [ (w, n) for w, n in which.items() ]
        counted.sort(key=lambda c: c[1], reverse=True)
        total = sum(which.values())
        return (counted, { c[0] : c[1]/total for c in counted })

    def _add_word(self, w, count=1):
        self.word_count += 1
        try:
            self.words[w] += count
        except KeyError:
            self.words[w] = count
            
    def _add_bigram(self, b, count=1):
        try:
            self.bigrams[b] += count
        except KeyError:
            self.bigrams[b] = count

    def _compare_one(self, coll, other_coll, thresh):
        good = set()
        for k in coll.keys():
            try:
                ratio = coll[k] / other_coll[k]
            except KeyError:
                continue
            if ratio > 1:
                ratio = 1/ratio
            if 1-ratio <= thresh:
                good.add(k)
        return good

    def _stddev(self, n, sum, sumsq):
        return max(math.sqrt(n * sumsq - sum * sum) / (n or 1), 0)

    def show_words(self, how_many=100):
        return self.show_ngrams(1, how_many)

    def show_bigrams(self, how_many=100):
        return self.show_ngrams(2, how_many)

    def show_ngrams(self, n, how_many):
        return self.n_grams[n].show(how_many)

    def show_details(self, n, how_many):
        d1, d2 = self.inputs[0], self.inputs[1]
        result = []
        n_gram = self.n_grams[n]
        n_gram.set_max_entries(how_many)
        for k,v in n_gram:
            prefix = '+++' if k in d1.n_grams[n] and k in d2.n_grams[n] else '   '
            result.append(f'{prefix} {n_gram.make_title(k)}   {d1.n_grams[n][k]:5d} {d2.n_grams[n][k]:5d}')
        return '\n'.join(result)

class document_set(object):

    class one_result(object):

        def __init__(self, d1, d2):
            self.d1, self.d2, self.metrics, self.order, self.distance = d1, d2, [], 0, 0

        def add_metric(self, m):
            self.metrics.append(m)
            self.order = m[args.which_n_gram - 1]

        def __str__(self):
            result = f'{self.d1.filename:50} {self.d2.filename:50} {self.distance:5.2f}   '
            result += '   '.join([ ' '.join([ f'{n:6.2f}' for n in m ]) for m in self.metrics ])
            return result

    def __init__(self, documents=None, filenames=None):
        if documents:
            self.documents = documents
        elif filenames:
            self.documents = [ document(filename=f) for f in filenames ]
        self.result = None

    def compare(self):
        self.result = []
        for d1, d2 in itertools.combinations(self.documents, 2):
            r = document_set.one_result(d1, d2)
            for th in args.thresholds:
                r.combined = document(combine=(d1,d2))
                metrics = r.combined.make_metrics(th)
                r.add_metric(metrics)
            self.result.append(r)
        max_metric = max([ r.order for r in self.result ])
        for r in self.result:
            r.distance = max_metric / (r.order or 1)
        self.result.sort(key=lambda r: r.distance)
        return self.result

    def show_details(self, n, max_entries=DEFAULT_MAX_ENTRIES):
        return self.result[n].combined.show_details(2, max_entries)

    def get_first(self):
        return self.result[0]

class parse_args(object) :

    def __init__(self) :
        p = argparse.ArgumentParser()
        p.add_argument('files', nargs='*')
        p.add_argument('-m', '--matches', type=int, default=DEFAULT_MATCH_NUMBER,
                       help='number of matches to use or show')
        p.add_argument('-o', '--output', type=str, default='', help='output file name')
        p.add_argument('-t', '--threshold', type=str, default=DEFAULT_THRESHOLD, \
                       help='match threshold or list')
        p.add_argument('-v', '--verbose', action='store_true', default=False,
                       help='show details of matching words and bigrams')
        p.add_argument('-B', '--bigram', action='store_true', default=False,
                       help='order results by bigram score')
        p.add_argument('-T', '--trigram', action='store_true', default=False,
                       help='order results by trigram score')
        p.add_argument('-W', '--word', action='store_true', default=False,
                       help='order results by word score')
        a = p.parse_args()
        self.files = [ f for f in itertools.chain(*[ glob(f'{f}/*' if os.path.isdir(f) else f)
                                                     for f in a.files or ('.',) ])
                       if not f.endswith('~') and not os.path.isdir(f) ]
        self.files.sort()
        for f in self.files:
            with open(f, 'r') as ff:
                pass
        self.output = a.output
        self.thresholds = [ float(t) for t in a.threshold.split(',') ]
        self.threshold = self.thresholds[-1]
        self.verbose = a.verbose
        self.matches = a.matches
        self.which_n_gram = 3 if a.trigram else 2 if a.bigram or not a.word else 1
        self.bigram_order = a.bigram or not a.word
        
def do_combinations():
    documents = document_set(filenames=args.files)
    print('\n'.join([ str(d) for d in documents.documents ]))
    print()
    result = documents.compare()
    print('\n'.join([ str(r) for r in result ]))
    print()
    if args.verbose and len(args.files)<=2:
        print()
        print(documents.show_details(0, max_entries=args.matches))

def main():
    if len(args.files)==0:
        print('no input files found')
    elif len(args.files) < 2:
        d = document(filename=args.files[0])
        if args.verbose:
            print(d.show_words())
            print()
            print(d.show_bigrams())
    else:
        do_combinations()


args = parse_args()
main()

