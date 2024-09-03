from transformers import RobertaTokenizer, RobertaModel
from experiment.utils import get_graph_dfg_data
from model.GAT_model import GATModel
from model.graph_attention_v2 import GraphAttentionV2

from parser import DFG_python
from parser.DFG import DFG_java

test_code = "experiment/resources/test_code.py"
code = open(test_code, 'r').read()

dfg_function = {
    'python': DFG_python,
    'java': DFG_java,
}


max_target_length = 512
max_source_length = 512
out_channels = 768  # Same dimension as CodeBERT embeddings
in_channels = 768
num_heads = 8
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
embedding_model = RobertaModel.from_pretrained("microsoft/codebert-base")
graph_data, sequence_embeddings = get_graph_dfg_data(code, embedding_model, tokenizer, lang='python')

assert graph_data.x.size(-1) == in_channels

graph_model = GATModel(in_channels, out_channels)
x, attention = graph_model(graph_data)
graph_embeddings = x

combined_model = GraphAttentionV2(embed_dim=out_channels, num_heads=1)
output, attn_weights = combined_model(sequence_embeddings.squeeze(), graph_embeddings.squeeze())