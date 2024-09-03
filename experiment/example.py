from transformers import RobertaTokenizer, RobertaModel
import torch
from torch_geometric.data import Data

from model.GAT_model import GATModel
from model.graph_attention_v2 import GraphAttentionV2
from parser import DFG_getter
import parser
from tree_sitter import Language, Parser

from parser import DFG_python
from parser.DFG import DFG_java
from torch_geometric.nn import GATConv

test_code = "experiment/resources/test_code.py"
code = open(test_code, 'r').read()

dfg_function = {
    'python': DFG_python,
    'java': DFG_java,
}


def build_graph(dfg, code_tokens, dfg_to_code, max_target_length=128):
    edges = []
    data = []
    for idx, x in enumerate(dfg):
        tokens_ids = tokenizer.convert_tokens_to_ids(code_tokens[dfg_to_code[idx][0]:dfg_to_code[idx][1]])
        tokens_ids.extend([tokenizer.pad_token_id] * (max_target_length - len(tokens_ids)))
        context_embeddings = model(torch.tensor(tokens_ids)[None, :])[0].squeeze().sum(dim=0)
        data.append(context_embeddings)
        for y in x[-1]:
            edges.append((y, idx))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    x = torch.stack(data)
    return Data(x=x.squeeze(), edge_index=edge_index)


max_target_length = 512
max_source_length = 512
path = parser.__path__[0]
LANGUAGE = Language(path + "/my-languages.so", 'python')
parser = Parser()
parser.set_language(LANGUAGE)
parser.set_language(LANGUAGE)
lang = 'python'
parser = [parser, dfg_function[lang]]
raw_code_tokens, dfg = DFG_getter.get_data_flow(code, parser)
normalized_dfg = DFG_getter.normalize_dataflow(dfg)

tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("microsoft/codebert-base")

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
graph_data = build_graph(dfg, code_tokens, dfg_to_code)

in_channels = graph_data.x.size(-1)
out_channels = 768  # Same dimension as CodeBERT embeddings
num_heads = 8
model = GATModel(in_channels, out_channels)
x, attention = model(graph_data)

graph_embeddings = x

graph_model = GraphAttentionV2(embed_dim=out_channels, num_heads=1)
print(sequence_embeddings.shape)
# sequence_embeddings = sequence_embeddings.unsqueeze(0)
# graph_embeddings = graph_embeddings.unsqueeze(0)
print(sequence_embeddings.shape)
output, attn_weights = graph_model(sequence_embeddings.squeeze(), graph_embeddings.squeeze())