#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common definitions for NER
"""

from util import one_hot

LBLS = [
    "1",
    "2",
    "3",
    "4",
    "5",
    ]
NONE = "1"
LMAP = {k: one_hot(5,i) for i, k in enumerate(LBLS)}
NUM = "NNNUMMM"
UNK = "UUUNKKK"

EMBED_SIZE = 50

auth_header_name = 'Glassdoor-NLP-API-Token'
auth_header_env = 'API_HEADER_TOKEN'