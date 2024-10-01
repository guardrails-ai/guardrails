# This file contains code adapted from the WordTokenizers.jl
# https://github.com/JuliaText/WordTokenizers.jl project.
# It is subject to the license terms in the Apache License file
# found in the top-level directory of this distribution.
# This file has been modified by Guardrails AI on September 27 2024.

import re


def replace_til_no_change(input_text, pattern, replacement):
    while re.search(pattern, input_text):
        input_text = re.sub(pattern, replacement, input_text)
    return input_text


def postproc_splits(sentences):
    """
    Applies heuristic rules to repair sentence splitting errors.
    Developed for use as postprocessing for the GENIA sentence
    splitter on PubMed abstracts, with minor tweaks for
    full-text documents.

    `sentences` should be a string, with line breaks on sentence boundaries.
    Returns a similar string, but more correct.

    Based on
    https://github.com/ninjin/geniass/blob/master/geniass-postproc.pl
    Which is
    (c) 2010 Sampo Pyysalo. No rights reserved, i.e. do whatever you like with this.
    Which draws in part on heuristics included in Yoshimasa Tsuruoka's
    medss.pl script.
    """
    # Remove Windows line endings
    sentences = sentences.replace("\r", "")

    # Breaks sometimes missing after "?", "safe" cases
    sentences = re.sub(r"\b([a-z]+\?) ([A-Z][a-z]+)\b", r"\1\n\2", sentences)
    # Breaks sometimes missing after "." separated with extra space, "safe" cases
    sentences = re.sub(r"\b([a-z]+ \.) ([A-Z][a-z]+)\b", r"\1\n\2", sentences)

    # No breaks producing lines only containing sentence-ending punctuation
    sentences = re.sub(r"\n([.!?]+)\n", r"\1\n", sentences)

    # No breaks inside parentheses/brackets
    # Unlimited length for no intervening parentheses/brackets
    sentences = replace_til_no_change(
        sentences, r"\[([^\[\]\(\)]*)\n([^\[\]\(\)]*)\]", r"[\1 \2]"
    )
    sentences = replace_til_no_change(
        sentences, r"\(([^\[\]\(\)]*)\n([^\[\]\(\)]*)\)", r"(\1 \2)"
    )
    # Standard mismatched with possible intervening
    sentences = replace_til_no_change(
        sentences, r"\[([^\[\]]{0,250})\n([^\[\]]{0,250})\]", r"[\1 \2]"
    )
    sentences = replace_til_no_change(
        sentences, r"\(([^\(\)]{0,250})\n([^\(\)]{0,250})\)", r"(\1 \2)"
    )

    # Guardrails mods for line breaks within quotes
    sentences = replace_til_no_change(
        sentences, r'"([^"\n]{0,250})\n([^"\n]{0,250})"', r'"\1 \2"'
    )
    sentences = replace_til_no_change(
        sentences, r"'([^'\n]{0,250})\n([^'\n]{0,250})'", r"'\1 \2'"
    )

    # Nesting to depth one
    sentences = replace_til_no_change(
        sentences,
        r"\[((?:[^\[\]]|\[[^\[\]]*\]){0,250})\n((?:[^\[\]]|\[[^\[\]]*\]){0,250})\]",
        r"[\1 \2]",
    )
    sentences = replace_til_no_change(
        sentences,
        r"\(((?:[^\(\)]|\([^\(\)]*\)){0,250})\n((?:[^\(\)]|\([^\(\)]*\)){0,250})\)",
        r"(\1 \2)",
    )

    # No break after periods followed by a non-uppercase "normal word"
    sentences = re.sub(r"\.\n([a-z]{3}[a-z-]*[ .:,])", r". \1", sentences)

    # No break after a single letter other than I
    sentences = re.sub(r"(\b[A-HJ-Z]\.)\n", r"\1 ", sentences)

    # No break before coordinating conjunctions (CC)
    coordinating_conjunctions = ["and", "or", "but", "nor", "yet"]
    for cc in coordinating_conjunctions:
        sentences = re.sub(r"\n(" + cc + r" )", r" \1", sentences)

    # No break before prepositions (IN)
    prepositions = [
        "of",
        "in",
        "by",
        "as",
        "on",
        "at",
        "to",
        "via",
        "for",
        "with",
        "that",
        "than",
        "from",
        "into",
        "upon",
        "after",
        "while",
        "during",
        "within",
        "through",
        "between",
        "whereas",
        "whether",
    ]
    for prep in prepositions:
        sentences = re.sub(r"\n(" + prep + r" )", r" \1", sentences)

    # No sentence breaks in the middle of specific abbreviations
    sentences = re.sub(r"(\be\.)\n(g\.)", r"\1 \2", sentences)
    sentences = re.sub(r"(\bi\.)\n(e\.)", r"\1 \2", sentences)
    sentences = re.sub(r"(\bi\.)\n(v\.)", r"\1 \2", sentences)

    # No sentence break after specific abbreviations
    abbreviations = [
        r"e\. ?g\.",
        r"i\. ?e\.",
        r"i\. ?v\.",
        r"vs\.",
        r"cf\.",
        r"Dr\.",
        r"Mr\.",
        r"Ms\.",
        r"Mrs\.",
        r"Prof\.",
        r"Ph\.?D\.",
        r"Jr\.",
        r"St\.",
        r"Mt\.",
        r"etc\.",
        r"Fig\.",
        r"vol\.",
        r"Vols\.",
        r"no\.",
        r"Nos\.",
        r"et\.",
        r"al\.",
        r"i\. ?v\.",
        r"inc\.",
        r"Ltd\.",
        r"Co\.",
        r"Corp\.",
        r"Dept\.",
        r"est\.",
        r"Asst\.",
        r"approx\.",
        r"dr\.",
        r"fig\.",
        r"mr\.",
        r"mrs\.",
        r"ms\.",
        r"prof\.",
        r"rep\.",
        r"jr\.",
        r"sen\.",
        r"st\.",
        r"vs\.",
        r"i\. ?e\.",
    ]
    for abbr in abbreviations:
        sentences = re.sub(r"(\b" + abbr + r")\n", r"\1 ", sentences)

    return sentences


# Original split sentences function from rulebased_split_sentences
def split_sentences(text):
    text = re.sub(r"([?!.])(\s)?", r"\1\n", text)
    text = postproc_splits(text)
    return text.split("\n")
