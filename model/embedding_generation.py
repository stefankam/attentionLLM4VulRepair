from transformers import AutoTokenizer, AutoModel

from model.GAT_model import GATModel


def generate_graph_embeddings(ast_data_list):
    in_channels = ast_data_list[0].num_node_features
    out_channels = 768  # Same dimension as CodeBERT embeddings
    model = GATModel(in_channels, out_channels)
    graph_embeddings = [model(data) for data in ast_data_list]
    return graph_embeddings


def generate_sequence_embeddings(code_snippets, model_name="microsoft/codebert-base"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    inputs = tokenizer(code_snippets, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state
    return embeddings
