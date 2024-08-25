from transformers import RobertaTokenizer, RobertaModel

from parser import DFG_getter
import parser
from tree_sitter import Language, Parser

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
path = parser.__path__[0]
LANGUAGE = Language(path + "/my-languages.so", 'python')
parser = Parser()
parser.set_language(LANGUAGE)
parser.set_language(LANGUAGE)
lang = 'python'
parser = [parser, dfg_function[lang]]
code_tokens, dfg = DFG_getter.get_data_flow(code, parser)
normalized_dfg = DFG_getter.normalize_dataflow(dfg)

tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("microsoft/codebert-base")

code_tokens = [tokenizer.tokenize('@ '+x)[1:] if idx!=0 else tokenizer.tokenize(x)
               for idx,x in enumerate(code_tokens)]
ori2cur_pos = {-1: (0, 0)}
for i in range(len(code_tokens)):
    ori2cur_pos[i]=(ori2cur_pos[i-1][1],ori2cur_pos[i-1][1]+len(code_tokens[i]))
code_tokens=[y for x in code_tokens for y in x]

code_tokens = code_tokens[:max_target_length - 3]
source_tokens = [tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
position_idx = [i+tokenizer.pad_token_id + 1 for i in range(len(source_tokens))]
dfg = dfg[:max_source_length - len(source_tokens)]
source_tokens += [x[0] for x in dfg]
position_idx += [0 for x in dfg]
source_ids += [tokenizer.unk_token_id for x in dfg]
padding_length = max_source_length-len(source_ids)
source_ids += [tokenizer.pad_token_id]*padding_length
source_mask = [1] * (len(source_tokens))
source_mask += [0]*padding_length
reverse_index = {}
for idx, x in enumerate(dfg):
    reverse_index[x[1]] = idx
for idx, x in enumerate(dfg):
    dfg[idx] = x[:-1] + ([reverse_index[i] for i in x[-1] if i in reverse_index],)
dfg_to_dfg = [x[-1] for x in dfg]
print(ori2cur_pos)
dfg_to_code = [ori2cur_pos[x[1]] for x in dfg]
length = len([tokenizer.cls_token])
dfg_to_code = [(x[0] + length, x[1] + length) for x in dfg_to_code]

print("generated dfg: ")
for x in dfg:
    print(x)

print("generated dfg_to_code: ")
for x in dfg_to_code:
    print(x)


