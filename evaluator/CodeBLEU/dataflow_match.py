# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
from parser.DFG import DFG_python, DFG_java
from parser.utils import remove_comments_and_docstrings
from tree_sitter import Language, Parser

from parser.DFG_getter import get_data_flow, normalize_dataflow
import parser

dfg_function = {
    'python': DFG_python,
    'java': DFG_java,
}


def calc_dataflow_match(references, candidate, lang):
    return corpus_dataflow_match([references], [candidate], lang)


def corpus_dataflow_match(references, candidates, lang):
    path = parser.dfg.__path__[0]
    LANGUAGE = Language(path + "/my-languages.so", lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    parser = [parser, dfg_function[lang]]
    match_count = 0
    total_count = 0

    for i in range(len(candidates)):
        references_sample = references[i]
        candidate = candidates[i]
        for reference in references_sample:
            try:
                candidate = remove_comments_and_docstrings(candidate, 'java')
            except:
                pass
            try:
                reference = remove_comments_and_docstrings(reference, 'java')
            except:
                pass

            cand_dfg = get_data_flow(candidate, parser)
            ref_dfg = get_data_flow(reference, parser)

            normalized_cand_dfg = normalize_dataflow(cand_dfg)
            normalized_ref_dfg = normalize_dataflow(ref_dfg)

            if len(normalized_ref_dfg) > 0:
                total_count += len(normalized_ref_dfg)
                for dataflow in normalized_ref_dfg:
                    if dataflow in normalized_cand_dfg:
                        match_count += 1
                        normalized_cand_dfg.remove(dataflow)
    if total_count == 0:
        print(
            "WARNING: There is no reference data-flows extracted from the whole corpus, and the data-flow match score degenerates to 0. Please consider ignoring this score.")
        return 0
    score = match_count / total_count
    return score



