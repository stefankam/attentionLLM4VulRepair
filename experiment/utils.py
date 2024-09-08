import torch
from torch_geometric.data import Data
from parser import DFG_getter
import parser
from tree_sitter import Language, Parser

from parser import DFG_python
from parser.DFG import DFG_java

dfg_function = {
    'python': DFG_python,
    'java': DFG_java,
}


def build_graph(dfg, code_tokens, dfg_to_code, tokenizer=None, model=None):
    edges = []
    data = []
    for idx, x in enumerate(dfg):
        tokens_ids = tokenizer.convert_tokens_to_ids(code_tokens[dfg_to_code[idx][0]:dfg_to_code[idx][1]])
        # sum the embeddings of the tokens for each dfg nodes
        context_embeddings = model(torch.tensor(tokens_ids)[None, :])[0].sum(dim=1)
        data.append(context_embeddings)
        for y in x[-1]:
            edges.append((y, idx))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    x = torch.stack(data)
    return Data(x=x.squeeze(), edge_index=edge_index)


def get_graph_dfg_data(code, model, tokenizer, lang='python'):
    max_target_length = 512
    max_source_length = 512
    path = parser.__path__[0]
    LANGUAGE = Language(path + "/my-languages.so", 'python')
    code_parser = Parser()
    code_parser.set_language(LANGUAGE)
    code_parser.set_language(LANGUAGE)
    code_parser = [code_parser, dfg_function[lang]]
    raw_code_tokens, dfg = DFG_getter.get_data_flow(code, code_parser)

    code_tokens = [tokenizer.tokenize('@ ' + x)[1:] if idx != 0 else tokenizer.tokenize(x)
                   for idx, x in enumerate(raw_code_tokens)]

    # ori2cur_pos: code token index -> (start, end)
    ori2cur_pos = {-1: (0, 0)}
    for i in range(len(code_tokens)):
        ori2cur_pos[i] = (ori2cur_pos[i - 1][1], ori2cur_pos[i - 1][1] + len(code_tokens[i]))
    # flat all tokens into one
    code_tokens = [y for x in code_tokens for y in x]
    code_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.eos_token]
    tokens_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    sequence_embeddings = model(torch.tensor(tokens_ids)[None, :])[0]

    reverse_index = {}
    for idx, x in enumerate(dfg):
        reverse_index[x[1]] = idx

    for idx, x in enumerate(dfg):
        dfg[idx] = x[:-1] + ([reverse_index[i] for i in x[-1] if i in reverse_index],)

    # dfg node becomes ("code", code token index, relationship, [par_code dfg index])

    dfg_to_dfg = [x[-1] for x in dfg]

    # tokenized code start and end positions for each dfg nodes
    dfg_to_code = [ori2cur_pos[x[1]] for x in dfg]
    length = len([tokenizer.cls_token])
    dfg_to_code = [(x[0] + length, x[1] + length) for x in dfg_to_code]
    graph_data = build_graph(dfg, code_tokens, dfg_to_code, model=model, tokenizer=tokenizer)
    return graph_data, sequence_embeddings
