from dataclasses import dataclass
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET

import numpy as np


@dataclass(frozen=True)
class SentencePair:
    """
    Contains lists of tokens (strings) for source and target sentence
    """
    source: List[str]
    target: List[str]


@dataclass(frozen=True)
class TokenizedSentencePair:
    """
    Contains arrays of token vocabulary indices (preferably np.int32) for source and target sentence
    """
    source_tokens: np.ndarray
    target_tokens: np.ndarray


@dataclass(frozen=True)
class LabeledAlignment:
    """
    Contains arrays of alignments (lists of tuples (source_pos, target_pos)) for a given sentence.
    Positions are numbered from 1.
    """
    sure: List[Tuple[int, int]]
    possible: List[Tuple[int, int]]


def extract_sentences(filename: str) -> Tuple[List[SentencePair], List[LabeledAlignment]]:
    """
    Given a file with tokenized parallel sentences and alignments in XML format, return a list of sentence pairs
    and alignments for each sentence.

    Args:
        filename: Name of the file containing XML markup for labeled alignments

    Returns:
        sentence_pairs: list of `SentencePair`s for each sentence in the file
        alignments: list of `LabeledAlignment`s corresponding to these sentences
    """
    with open(filename, 'r', encoding="utf-8") as f:
        xml_string = f.read().replace("&", "&amp;") #как я ваще должен был до этого догадаться втф 
    
    sentence_pairs = []
    alignments = []
    root = ET.fromstring(xml_string) 
    for child in root:
        source = child[0].text.split() if child[0].text is not None else []
        target = child[1].text.split() if child[1].text is not None else []
        sentence_pairs.append(SentencePair(source, target))
        sure = [tuple(map(int, p.split('-'))) for p in child[2].text.split()] if child[2].text is not None else []
        possible = [tuple(map(int, p.split('-'))) for p in child[3].text.split()] if child[3].text is not None else []
        alignments.append(LabeledAlignment(sure, possible))
    return (sentence_pairs, alignments)


def get_token_to_index(sentence_pairs: List[SentencePair], freq_cutoff=None) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Given a parallel corpus, create two dictionaries token->index for source and target language.

    Args:
        sentence_pairs: list of `SentencePair`s for token frequency estimation
        freq_cutoff: if not None, keep only freq_cutoff most frequent tokens in each language

    Returns:
        source_dict: mapping of token to a unique number (from 0 to vocabulary size) for source language
        target_dict: mapping of token to a unique number (from 0 to vocabulary size) target language

    """
    source_dict = {}
    target_dict = {}
    
    frequency_dict_source = {}
    frequency_dict_target = {}
    
    for sentence_pair in sentence_pairs:
        for token in sentence_pair.source:
            frequency_dict_source[token] = frequency_dict_source.get(token, 0) + 1
        for token in sentence_pair.target:
            frequency_dict_target[token] = frequency_dict_target.get(token, 0) + 1

    sorted_source_tokens = sorted(frequency_dict_source.items(), key=lambda x: (-x[1], x[0]))
    sorted_target_tokens = sorted(frequency_dict_target.items(), key=lambda x: (-x[1], x[0]))

    if freq_cutoff is not None:
        sorted_source_tokens = sorted_source_tokens[:freq_cutoff]
        sorted_target_tokens = sorted_target_tokens[:freq_cutoff]
    
    for index, (token, _) in enumerate(sorted_source_tokens):
        source_dict[token] = index
    for index, (token, _) in enumerate(sorted_target_tokens):
        target_dict[token] = index
    return (source_dict, target_dict)
    


def tokenize_sents(sentence_pairs: List[SentencePair], source_dict, target_dict) -> List[TokenizedSentencePair]:
    """
    Given a parallel corpus and token_to_index for each language, transform each pair of sentences from lists
    of strings to arrays of integers. If either source or target sentence has no tokens that occur in corresponding
    token_to_index, do not include this pair in the result.
    
    Args:
        sentence_pairs: list of `SentencePair`s for transformation
        source_dict: mapping of token to a unique number for source language
        target_dict: mapping of token to a unique number for target language

    Returns:
        tokenized_sentence_pairs: sentences from sentence_pairs, tokenized using source_dict and target_dict
    """
    tokenized_sentence_pairs = []
    to_add = True
    
    for sentence_pair in sentence_pairs:
        to_add = True
        source_tokens = []
        target_tokens = []
        source_sentence = sentence_pair.source
        target_sentence = sentence_pair.target
        for token in source_sentence:
            index = source_dict.get(token, -1)
            if index == -1:
                to_add = False
                break
            source_tokens.append(index)
        for token in target_sentence:
            index = target_dict.get(token, -1)
            if index == -1:
                to_add = False
                break
            target_tokens.append(index)    
        if (to_add):
            tokenized_sentence_pairs.append(TokenizedSentencePair(np.array(source_tokens), np.array(target_tokens)))
    return tokenized_sentence_pairs
            
            
        
        
