#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import re

f1 = open('synset_words.txt', 'r')
f2 = open('wnjpn-ok.tab', 'r')
#f2 = open('wnjpn-all.tab', 'r')
f3 = open('synset_words_jp.txt', 'w')

en_words = list(f1)
jp_words = list(f2)

jp_dict = {}
for line in jp_words:
    pair = line.split('\t')
    pair[0] = re.sub(r'([0-9]+)-n', r'n\1', pair[0])
    jp_dict[pair[0]] = pair[1].rstrip()

en_list = []
for line in en_words:
    m = re.search(r'(n[0-9]{8}) (.+)\n', line)
    en_list.append((m.group(1), m.group(2).rstrip()))

# synsets.txtの番号を日本語に置き換え
for pair in en_list:
    code = re.sub(r'\n', '', pair[0])
    if code in jp_dict:
        f3.write('%s\t%s\n' % (code, jp_dict[code]))
    else:
        f3.write('%s\t%s\n' % (code, pair[1]))
