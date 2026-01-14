# This file contains code adapted from the WordTokenizers.jl
# https://github.com/JuliaText/WordTokenizers.jl project.
# It is subject to the license terms in the Apache License file
# found in the top-level directory of this distribution.
# This file has been modified by Guardrails AI on September 27 2024.

import re


def replace_til_no_change(input_text, pattern, replacement):
    while True:
        new_text = re.sub(pattern, replacement, input_text)
        if new_text == input_text:
            break
        input_text = new_text
    return input_text


def postproc_splits(sentences, separator):
    """Applies heuristic rules to repair sentence splitting errors. Developed
    for use as postprocessing for the GENIA sentence splitter on PubMed
    abstracts, with minor tweaks for full-text documents.

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
    sentences = re.sub(
        r"\b([a-z]+\?)\s+([A-Z][a-z]+)\b", rf"\1{separator}\2", sentences
    )
    # Breaks sometimes missing after ".", "safe" cases
    sentences = re.sub(
        r"\b([a-z]+ \.)\s+([A-Z][a-z]+)\b", rf"\1{separator}\2", sentences
    )

    # No breaks producing lines only containing sentence-ending punctuation
    sentences = re.sub(rf"{separator}([.!?]+){separator}", r"\1" + separator, sentences)

    # No breaks inside parentheses/brackets
    sentences = replace_til_no_change(
        sentences,
        r"\[([^\[\]\(\)]*)" + re.escape(separator) + r"([^\[\]\(\)]*)\]",
        r"[\1 \2]",
    )
    sentences = replace_til_no_change(
        sentences,
        r"\(([^\[\]\(\)]*)" + re.escape(separator) + r"([^\[\]\(\)]*)\)",
        r"(\1 \2)",
    )
    # Standard mismatched with possible intervening
    sentences = replace_til_no_change(
        sentences,
        r"\[([^\[\]]{0,250})" + re.escape(separator) + r"([^\[\]]{0,250})\]",
        r"[\1 \2]",
    )
    sentences = replace_til_no_change(
        sentences,
        r"\(([^\(\)]{0,250})" + re.escape(separator) + r"([^\(\)]{0,250})\)",
        r"(\1 \2)",
    )

    # Line breaks within quotes
    sentences = replace_til_no_change(
        sentences,
        r'"([^"\n]{0,250})' + re.escape(separator) + r'([^"\n]{0,250})"',
        r'"\1 \2"',
    )
    sentences = replace_til_no_change(
        sentences,
        r"'([^'\n]{0,250})" + re.escape(separator) + r"([^'\n]{0,250})'",
        r"'\1 \2'",
    )

    # Nesting to depth one
    sentences = replace_til_no_change(
        sentences,
        r"\[((?:[^\[\]]|\[[^\[\]]*\]){0,250})"
        + re.escape(separator)
        + r"((?:[^\[\]]|\[[^\[\]]*\]){0,250})\]",
        r"[\1 \2]",
    )
    sentences = replace_til_no_change(
        sentences,
        r"\(((?:[^\(\)]|\([^\(\)]*\)){0,250})"
        + re.escape(separator)
        + r"((?:[^\(\)]|\([^\(\)]*\)){0,250})\)",
        r"(\1 \2)",
    )

    # No break after periods followed by a non-uppercase "normal word"
    sentences = re.sub(rf"\.{separator}([a-z]{{3,}}[a-z-]*[ .:,])", r". \1", sentences)

    # No break after a single letter other than I
    sentences = re.sub(rf"(\b[A-HJ-Z]\.){separator}", r"\1 ", sentences)

    # No break before coordinating conjunctions (CC)
    coordinating_conjunctions = ["and", "or", "but", "nor", "yet"]
    for cc in coordinating_conjunctions:
        sentences = re.sub(rf"{separator}({cc}\s)", r" \1", sentences)

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
        sentences = re.sub(rf"{separator}({prep}\s)", r" \1", sentences)

    # No sentence breaks in the middle of specific abbreviations
    sentences = re.sub(rf"(\be\.){separator}(g\.)", r"\1 \2", sentences)
    sentences = re.sub(rf"(\bi\.){separator}(e\.)", r"\1 \2", sentences)
    sentences = re.sub(rf"(\bi\.){separator}(v\.)", r"\1 \2", sentences)

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
        sentences = re.sub(
            rf"(\b{abbr}){separator}", r"\1", sentences, flags=re.IGNORECASE
        )

    return sentences


def split_sentences(text, separator="abcdsentenceseperatordcba"):
    # Use the separator in the regex
    text = re.sub(r"([?!.])(?=\s|$)", rf"\1{separator}", text)
    text = postproc_splits(text, separator)
    return re.split(rf"\n?{separator} ?\n?", text)
