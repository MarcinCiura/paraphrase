#!/usr/bin/python
# -*- coding: utf-8 -*-

import gzip
import re
import sys
import unicodedata

from gensim import models

NUM_SIMILAR_TOKENS = 100

ALPHABETIC = ''.join(
    unichr(x) for x in xrange(0x250)
    if unicodedata.category(unichr(x)) in ('Lu', 'Ll'))

CASE_UNFOLDERS = [
    lambda w: w,
    lambda w: w.title(),
    lambda w: w.upper(),
]

TRIVIAL_BASES = frozenset([u'być', u'jaki', u'który'])

VERB_PREFIXES = [
    u'nie', u'pół', u'współ', u'do', u'na', u'nad', u'nade', u'o',
    u'ob', u'obe', u'od', u'ode', u'po', u'pod', u'pode', u'prze',
    u'przed', u'przy', u'przeciw', u'roz', u'roze', u's', u'ś',
    u'u', u'w', u'we', u'wy', u'z', u'za', u'zd', u'ze',
]

Tokenize = re.compile(u'([^{}]+)'.format(ALPHABETIC), re.UNICODE).split


def BaseIsTrivial(base):
  return (base in TRIVIAL_BASES)


def CategoryIsBlacklisted(category):
  return (category == 'burk')  # Burkinostka.


def AllCategoriesAreInteresting(categories):
  return all(re.match(
      '(subst|depr|ger|adj|adjp|adv|'
      'inf|fin|praet|impt|imps|pact|pant|pcon|ppas):',
      c) for c in categories)


def WordsAreTooSimilar(w1, w2):
  if len(w1) < len(w2):
    w1, w2 = w2, w1
  if w2 in w1:
    return True
  for prefix in VERB_PREFIXES:
    if w1.startswith(prefix) and WordsAreTooSimilar(w1[len(prefix):], w2):
      return True
  return False


def FoldCase(token):
  word = token.lower()
  for case_unfolder in CASE_UNFOLDERS:
    if case_unfolder(word) == token:
      return word, case_unfolder
  return token, CASE_UNFOLDERS[0]


class Lexicon(object):

  def __init__(self):
    self.forms = {}

  def Read(self, f):
    for line in f:
      split = line.split('\t')
      base = split[1].decode('utf-8')
      if BaseIsTrivial(base):
        continue
      form = split[0].decode('utf-8')
      category = split[2]
      if not CategoryIsBlacklisted(category):
        self.forms.setdefault(form, set()).add(category)

  def GetCategories(self, form):
    categories = self.forms.get(form, frozenset())
    if AllCategoriesAreInteresting(categories):
      return categories
    return frozenset()

  def IsValidReplacement(self, form, categories):
    return not categories.isdisjoint(self.GetCategories(form))


def GetReplacements(model, token):
  if token in model.vocab:
    return [w[0] for w in model.similar_by_word(token, NUM_SIMILAR_TOKENS)]
  return []


def main():
  try:
    model = models.KeyedVectors.load_word2vec_format('wiki.pl.vec')
  except IOError:
    sys.exit(
        'Please run\n  wget '
        'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.pl.vec'
        '\n  gzip wiki.pl.vec')
  model.init_sims(replace=True)
  lexicon = Lexicon()
  try:
    with gzip.open('PoliMorf-0.6.7.tab.gz') as f:
      lexicon.Read(f)
  except IOError:
    sys.exit(
        'Please run\n wget -U Mozilla -O PoliMorf-0.6.7.tab.gz '
        "'http://zil.ipipan.waw.pl/PoliMorf?action=AttachFile&do=get&target=PoliMorf-0.6.7.tab.gz'")
  sys.stderr.write('Ready!\n')
  while True:
    try:
      line = raw_input()
    except EOFError:
      return
    for token in Tokenize(line.decode('utf-8')):
      word, case_unfolder = FoldCase(token)
      categories = lexicon.GetCategories(word)
      for replacement in GetReplacements(model, word):
        if (lexicon.IsValidReplacement(replacement, categories) and
            not WordsAreTooSimilar(word, replacement)):
          sys.stdout.write(case_unfolder(replacement))
          break
      else:
        sys.stdout.write(token)
      sys.stdout.flush()
    sys.stdout.write('\n')


if __name__ == '__main__':
  main()
