# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List

from deeppavlov.core.models.component import Component
from deeppavlov.core.common.registry import register

# #################################################################################
""" emoticon recognition via patterns.  tested on english-language twitter, but
probably works for other social media dialects. """

__author__ = "Brendan O'Connor (anyall.org, brenocon@gmail.com)"
__version__= "april 2009"

#from __future__ import print_function
import re,os

mycompile = lambda pat:  re.compile(pat,  re.UNICODE)
#SMILEY = mycompile(r'[:=].{0,1}[\)dpD]')
#MULTITOK_SMILEY = mycompile(r' : [\)dp]')

NormalEyes = r'[:=]'
Wink = r'[;]'

NoseArea = r'(|o|O|-)'   ## rather tight precision, \S might be reasonable...

HappyMouths = r'[D\)\]]'
SadMouths = r'[\(\[]'
Tongue = r'[pP]'
OtherMouths = r'[doO/\\]'  # remove forward slash if http://'s aren't cleaned

Happy_RE =  mycompile( '(\^_\^|' + NormalEyes + NoseArea + HappyMouths + ')')
Sad_RE = mycompile(NormalEyes + NoseArea + SadMouths)

Wink_RE = mycompile(Wink + NoseArea + HappyMouths)
Tongue_RE = mycompile(NormalEyes + NoseArea + Tongue)
Other_RE = mycompile( '('+NormalEyes+'|'+Wink+')'  + NoseArea + OtherMouths )

Emoticon = (
    "("+NormalEyes+"|"+Wink+")" +
    NoseArea +
    "("+Tongue+"|"+OtherMouths+"|"+SadMouths+"|"+HappyMouths+")"
)
Emoticon_RE = mycompile(Emoticon)

#Emoticon_RE = "|".join([Happy_RE,Sad_RE,Wink_RE,Tongue_RE,Other_RE])
#Emoticon_RE = mycompile(Emoticon_RE)

def analyze_tweet(text):
  h= Happy_RE.search(text)
  s= Sad_RE.search(text)
  if h and s: return "BOTH_HS"
  if h: return "HAPPY"
  if s: return "SAD"
  return "NA"

  # more complex & harder, so disabled for now
  #w= Wink_RE.search(text)
  #t= Tongue_RE.search(text)
  #a= Other_RE.search(text)
  #h,w,s,t,a = [bool(x) for x in [h,w,s,t,a]]
  #if sum([h,w,s,t,a])>1: return "MULTIPLE"
  #if sum([h,w,s,t,a])==1:
  #  if h: return "HAPPY"
  #  if s: return "SAD"
  #  if w: return "WINK"
  #  if a: return "OTHER"
  #  if t: return "TONGUE"
  #return "NA"


""" tokenizer for tweets!  might be appropriate for other social media dialects too.
general philosophy is to throw as little out as possible.
development philosophy: every time you change a rule, do a diff of this
program's output on ~100k tweets.  if you iterate through many possible rules
and only accept the ones that seeem to result in good diffs, it's a sort of
statistical learning with in-the-loop human evaluation :)
"""

__author__="brendan o'connor (anyall.org)"

def is_url(s):
  """
  Check token for domain name / ip address / URL
  :param s: input token, str
  :return: if the token is a URL, boolean
  """
  return s.startswith('http:') or s.startswith('https:') or s.startswith('ftp:') \
or s.startswith('ftps:') or s.startswith('smb:') \
or re.match('^([A-Za-z0-9]\.|[A-Za-z0-9][A-Za-z0-9-]{0,61}[A-Za-z0-9]\.){1,}[A-Za-z]{2,6}$', s) \
or re.match('^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$', s)

def is_version(s):
  return re.match(r'^([156789][012]?)\.([014]{2})$', s)

def is_number(s):
  try:
    int(s)
    return True
  except ValueError:
    return False

mycompile = lambda pat:  re.compile(pat,  re.UNICODE)
def regex_or(*items):
  r = '|'.join(items)
  r = '(' + r + ')'
  return r
def pos_lookahead(r):
  return '(?=' + r + ')'
def neg_lookahead(r):
  return '(?!' + r + ')'
def optional(r):
  return '(%s)?' % r


PunctChars = r'''['“".?!,:;]'''
Punct = '%s+' % PunctChars
Entity = '&(amp|lt|gt|quot);'

# one-liner URL recognition:
#Url = r'''https?://\S+'''

# more complex version:
UrlStart1 = regex_or('https?://', r'www\.')
CommonTLDs = regex_or('com','co\\.uk','org','net','info','ca')
UrlStart2 = r'[a-z0-9\.-]+?' + r'\.' + CommonTLDs + pos_lookahead(r'[/ \W\b]')
UrlBody = r'[^ \t\r\n<>]*?'  # * not + for case of:  "go to bla.com." -- don't want period
UrlExtraCrapBeforeEnd = '%s+?' % regex_or(PunctChars, Entity)
UrlEnd = regex_or( r'\.\.+', r'[<>]', r'\s', '$')
Url = (r'\b' +
    regex_or(UrlStart1, UrlStart2) +
    UrlBody +
    pos_lookahead( optional(UrlExtraCrapBeforeEnd) + UrlEnd))

Url_RE = re.compile("(%s)" % Url, re.U|re.I)

Timelike = r'\d+:\d+'
NumNum = r'\d+\.\d+'
NumberWithCommas = r'(\d+,)+?\d{3}' + pos_lookahead(regex_or('[^,]','$'))

Abbrevs1 = ['am','pm','us','usa','ie','eg']
def regexify_abbrev(a):
  chars = list(a)
  icase = ["[%s%s]" % (c,c.upper()) for c in chars]
  dotted = [r'%s\.' % x for x in icase]
  return "".join(dotted)
Abbrevs = [regexify_abbrev(a) for a in Abbrevs1]

BoundaryNotDot = regex_or(r'\s', '[“"?!,:;]', Entity)
aa1 = r'''([A-Za-z]\.){2,}''' + pos_lookahead(BoundaryNotDot)
aa2 = r'''([A-Za-z]\.){1,}[A-Za-z]''' + pos_lookahead(BoundaryNotDot)
ArbitraryAbbrev = regex_or(aa1,aa2)

assert '-' != '―'
Separators = regex_or('--+', '―')
Decorations = r' [  ♫   ]+ '.replace(' ','')

EmbeddedApostrophe = r"\S+'\S+"

ProtectThese = [
    Emoticon,
    Url,
    Entity,
    Timelike,
    NumNum,
    NumberWithCommas,
    Punct,
    ArbitraryAbbrev,
    Separators,
    Decorations,
    EmbeddedApostrophe,
]
Protect_RE = mycompile(regex_or(*ProtectThese))


class Tokenization(list):
  " list of tokens, plus extra info "
  def __init__(self):
    self.alignments = []
    self.text = ""
  def subset(self, tok_inds):
    new = Tokenization()
    new += [self[i] for i in tok_inds]
    new.alignments = [self.alignments[i] for i in tok_inds]
    new.text = self.text
    return new
  def assert_consistent(t):
    assert len(t) == len(t.alignments)
    assert [t.text[t.alignments[i] : (t.alignments[i]+len(t[i]))] for i in range(len(t))] == list(t)

def align(toks, orig):
  s_i = 0
  alignments = [None]*len(toks)
  for tok_i in range(len(toks)):
    while True:
      L = len(toks[tok_i])
      if orig[s_i:(s_i+L)] == toks[tok_i]:
        alignments[tok_i] = s_i
        s_i += L
        break
      s_i += 1
      if s_i >= len(orig): raise AlignmentFailed((orig,toks,alignments))
      #if orig[s_i] != ' ': raise AlignmentFailed("nonspace advance: %s" % ((s_i,orig),))
  if any(a is None for a in alignments): raise AlignmentFailed((orig,toks,alignments))

  return alignments

class AlignmentFailed(Exception): pass

def tokenize(tweet):
  # text = unicodify(tweet)
  text = squeeze_whitespace(tweet)
  t = Tokenization()
  t += simple_tokenize(text)
  t.text = text
  t.alignments = align(t, text)
  return t

def simple_tokenize(text):
  s = text
  s = edge_punct_munge(s)

  # strict alternating ordering through the string.  first and last are goods.
  # good bad good bad good bad good
  goods = []
  bads = []
  i = 0
  if Protect_RE.search(s):
    for m in Protect_RE.finditer(s):
      goods.append( (i,m.start()) )
      bads.append(m.span())
      i = m.end()
    goods.append( (m.end(), len(s)) )
  else:
    goods = [ (0, len(s)) ]
  assert len(bads)+1 == len(goods)

  goods = [s[i:j] for i,j in goods]
  bads  = [s[i:j] for i,j in bads]
  #print goods
  #print bads
  goods = [unprotected_tokenize(x) for x in goods]
  res = []
  for i in range(len(bads)):
    res += goods[i]
    res.append(bads[i])
  res += goods[-1]

  # res = post_process(res)  # no need

  # split tokens like 'webbrowser/mailer/help'. 'debian/main'
  res = post_process_slashes(res)
  # res = post_process_strip(res)
  return res

AposS = mycompile(r"(\S+)('s)$")

def post_process(pre_toks):
  # hacky: further splitting of certain tokens
  post_toks = []
  for tok in pre_toks:
    m = AposS.search(tok)
    if m:
      post_toks += m.groups()
    else:
      post_toks.append( tok )
  return post_toks

InnerSlashes = mycompile(r"([^/]+)(?:/)")
InnerSlashes2 = mycompile(r"([^/]+)(?:/)?")

Split = mycompile (r"([^\[\])(=<>\\\.*:@/-]+)([\[\])(=<>\\\.*:@/-]+)")
Split2 = mycompile(r"([^\[\])(=<>\\\.*:@/-]+)")

def post_process_slashes(pre_toks):
  post_toks = []
  for tok in pre_toks:
    m = Split.search(tok)
    if m and not is_url(tok) and not is_version(tok):       # don't process URLs
      post_toks.extend(Split2.findall(tok))
    else:
      post_toks.append(tok)
  return post_toks

def post_process_strip(pre_toks):
  post_toks = []
  to_remove_tuple = ('=', '\'', '!', '*', '-', ')', '(', '#', '?', '+',
                     '~', '<', '>', '}', '{', '$', '\\', '^', '.', ':', ';', '@', '"', '|', ']', '[')
  for tok in pre_toks:
    tok_ = tok
    for i in range(10):
        if tok.startswith(to_remove_tuple):
          tok_ = tok_[1:]
        if tok.endswith(to_remove_tuple):
          tok_ = tok_[:-1]
    post_toks.append(tok_)
  return post_toks

WS_RE = mycompile(r'\s+')
def squeeze_whitespace(s):
  new_string = WS_RE.sub(" ",s)
  return new_string.strip()

# fun: copy and paste outta http://en.wikipedia.org/wiki/Smart_quotes
EdgePunct      = r"""[  ' " “ ” ‘ ’ < > « » { } ( \) [ \]  ]""".replace(' ','')
#NotEdgePunct = r"""[^'"([\)\]]"""  # alignment failures?
NotEdgePunct = r"""[a-zA-Z0-9]"""
EdgePunctLeft  = r"""(\s|^)(%s+)(%s)""" % (EdgePunct, NotEdgePunct)
EdgePunctRight =   r"""(%s)(%s+)(\s|$)""" % (NotEdgePunct, EdgePunct)
EdgePunctLeft_RE = mycompile(EdgePunctLeft)
EdgePunctRight_RE= mycompile(EdgePunctRight)

def edge_punct_munge(s):
  s = EdgePunctLeft_RE.sub( r"\1\2 \3", s)
  s = EdgePunctRight_RE.sub(r"\1 \2\3", s)
  return s


def unprotected_tokenize(s):
  return s.split()

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = string.replace('`', '\'')

    # string = re.sub(r"\'m", " \'m", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    # string = re.sub(r"%20", " ", string)

    # string = re.sub(r"`", " ` ", string)
    # string = re.sub(r",", " , ", string)
    return string.strip()

def remove_punctuation(string):
    # as Keras filters does
    string = string.replace('\t', '')
    string = string.replace('\n', '')
    string = string.replace(',', '')

    for i in range(2):
        # Then remove all punctuation characters
        string = re.sub(r"(?:\s|^)\.(?:\s|$)", " ", string)
        string = re.sub(r"(?:\s|^)\"(?:\s|$)", " ", string)
        string = re.sub(r"(?:\s|^):(?:\s|$)", " ", string)
        string = re.sub(r"(?:\s|^)'(?:\s|$)", " ", string)
        string = re.sub(r"(?:\s|^)!(?:\s|$)", " ", string)
        string = re.sub(r"(?:\s|^);(?:\s|$)", " ", string)
        string = re.sub(r"(?:\s|^)](?:\s|$)", " ", string)
        string = re.sub(r"(?:\s|^)\[(?:\s|$)", " ", string)
        string = re.sub(r"(?:\s|^)\](?:\s|$)", " ", string)
        string = re.sub(r"(?:\s|^)}(?:\s|$)", " ", string)
        string = re.sub(r"(?:\s|^){(?:\s|$)", " ", string)
        string = re.sub(r"(?:\s|^)`(?:\s|$)", " ", string)
        string = re.sub(r"(?:\s|^)\?(?:\s|$)", " ", string)

        string = re.sub(r"(?:\s|^)\((?:\s|$)", " ", string)
        string = re.sub(r"(?:\s|^)\)(?:\s|$)", " ", string)
        string = re.sub(r"(?:\s|^)#(?:\s|$)", " ", string)

        # string = re.sub(r"(?:\s|^)<(?:\s|$)", " ", string)
        # string = re.sub(r"(?:\s|^)>(?:\s|$)", " ", string)
        # string = re.sub(r"(?:\s|^)-(?:\s|$)", " ", string)
        # string = re.sub(r"(?:\s|^)*(?:\s|$)", " ", string)
        # string = re.sub(r"(?:\s|^)=(?:\s|$)", " ", string)
        # string = re.sub(r"(?:\s|^)+(?:\s|$)", " ", string)
        # string = re.sub(r"(?:\s|^)|(?:\s|$)", " ", string)
        #
        # string = re.sub(r"(?:\s|^)[\[\])(=<>\\.*:;@/_+&~\"'-]{2}(?:\s|$)", " ", string)  # replace '--', '"?'

        # string = re.sub(r"(?:\s|^)'s(?:\s|$)", " ", string)
        # string = re.sub(r"(?:\s|^)'ll(?:\s|$)", " ", string)
        # string = re.sub(r"(?:\s|^)'d(?:\s|$)", " ", string)
        # string = re.sub(r"(?:\s|^)'ve(?:\s|$)", " ", string)
        # string = re.sub(r"(?:\s|^)'re(?:\s|$)", " ", string)


    # string = string.replace("__eot__", "%%%%%EOT%%%%%")
    # string = string.replace("__eou__", "%%%%%EOU%%%%%")
    # string = string.replace("_","")
    # string = string.replace("%%%%%EOT%%%%%"," _eot_ ")
    # string = string.replace("%%%%%EOT%%%%%", " _eou_ ")

    string = string.replace("__eot__", " _eot_ ")
    # string = string.replace("__eou__", " _eou_ ")

    return string

def process_token(c, word):
    """
    Use NLTK to replace named entities with generic tags.
    Also replace URLs, numbers, and paths.
    """
    # nodelist = ['PERSON', 'ORGANIZATION', 'GPE', 'LOCATION', 'FACILITY', 'GSP']
    # if hasattr(c, 'label'):
    #     if c.label() in nodelist:
    #         return "__%s__" % c.label()
    if is_url(word):
        return "__URL__"
    elif is_number(word):
        return "__NUMBER__"
    elif os.path.isabs(word):
        return "__PATH__"
    return word

def process_line(s, clean_string=True):
    """
    Processes a line by iteratively calling process_token.
    """
    if clean_string:
        s = clean_str(s)
    tokens = tokenize(s)
    # sent = nltk.pos_tag(tokens)
    # chunks = nltk.ne_chunk(sent, binary=False)

    # return [process_token(c,token).lower() for c,token in zip(chunks, tokens)]
    return [process_token(None, token).lower() for token in tokens]    # do not use POS tagging

# #################################################################################

from multiprocessing import Pool


@register("twokenize_tokenizer")
class TwokenizeTokenizer(Component):
    """
    TODO: docstrings

    Doesn't have any parameters.
    """
    def __init__(self, **kwargs) -> None:
        self.pool = Pool()

    def map_process_line(self, sample):
        return remove_punctuation(" ".join(process_line(sample, clean_string=True))).split()

    def __call__(self, batch: List[str]) -> List[List[str]]:
        """
        Tokenize given batch

        Args:
            batch: list of texts to tokenize

        Returns:
            tokenized batch
        """
        if isinstance(batch, (list, tuple)):
            return [*self.pool.map(self.map_process_line, batch)]
        else:
            raise NotImplementedError('not implemented for types other than'
                                      ' list or tuple')
