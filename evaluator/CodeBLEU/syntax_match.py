# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

import parser
from parser import DFG_python
from parser import (remove_comments_and_docstrings)
from tree_sitter import Language, Parser

dfg_function = {
    'python': DFG_python,
}


def calc_syntax_match(references, candidate, lang):
    return corpus_syntax_match([references], [candidate], lang)


def corpus_syntax_match(references, candidates, lang):
    path = parser.__path__[0]
    LANGUAGE = Language(path + "/my-languages.so", lang)
    lang_parser = Parser()
    lang_parser.set_language(LANGUAGE)
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

            candidate_tree = lang_parser.parse(bytes(candidate, 'utf8')).root_node

            reference_tree = lang_parser.parse(bytes(reference, 'utf8')).root_node

            def get_all_sub_trees(root_node):
                node_stack = []
                sub_tree_sexp_list = []
                depth = 1
                node_stack.append([root_node, depth])
                while len(node_stack) != 0:
                    cur_node, cur_depth = node_stack.pop()
                    sub_tree_sexp_list.append([cur_node.sexp(), cur_depth])
                    for child_node in cur_node.children:
                        if len(child_node.children) != 0:
                            depth = cur_depth + 1
                            node_stack.append([child_node, depth])
                return sub_tree_sexp_list

            cand_sexps = [x[0] for x in get_all_sub_trees(candidate_tree)]
            ref_sexps = get_all_sub_trees(reference_tree)

            # print(cand_sexps)
            # print(ref_sexps)

            for sub_tree, depth in ref_sexps:
                if sub_tree in cand_sexps:
                    match_count += 1
            total_count += len(ref_sexps)

    score = match_count / total_count
    return score
