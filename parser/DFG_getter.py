from parser.utils import tree_to_token_index, index_to_code_token


def get_data_flow(code, parser):
    try:
        tree = parser[0].parse(bytes(code, 'utf8'))
        root_node = tree.root_node
        tokens_index = tree_to_token_index(root_node)
        code = code.split('\n')
        code_tokens = [index_to_code_token(x, code) for x in tokens_index]
        index_to_code = {}
        for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
            index_to_code[index] = (idx, code)
        try:
            DFG, _ = parser[1](root_node, index_to_code, {})
        except:
            DFG = []
        DFG = sorted(DFG, key=lambda x: x[1])
        indexs = set()
        for d in DFG:
            if len(d[-1]) != 0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG = []
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        codes = code_tokens
        dfg = new_DFG
    except:
        dfg = []
        code_tokens = []
    #merge nodes
    dic = {}
    for d in dfg:
        if d[1] not in dic:
            dic[d[1]] = d
        else:
            dic[d[1]] = (d[0], d[1], d[2], list(set(dic[d[1]][3] + d[3])), list(set(dic[d[1]][4] + d[4])))
    new_DFG = []
    for d in dic:
        new_DFG.append(dic[d])
    dfg = new_DFG
    return code_tokens, dfg


def normalize_dataflow_item(dataflow_item):
    var_name = dataflow_item[0]
    var_pos = dataflow_item[1]
    relationship = dataflow_item[2]
    par_vars_name_list = dataflow_item[3]
    par_vars_pos_list = dataflow_item[4]

    var_names = list(set(par_vars_name_list + [var_name]))
    norm_names = {}
    for i in range(len(var_names)):
        norm_names[var_names[i]] = 'var_' + str(i)

    norm_var_name = norm_names[var_name]
    relationship = dataflow_item[2]
    norm_par_vars_name_list = [norm_names[x] for x in par_vars_name_list]

    return (norm_var_name, relationship, norm_par_vars_name_list)


def normalize_dataflow(dataflow):
    var_dict = {}
    i = 0
    normalized_dataflow = []
    for item in dataflow:
        var_name = item[0]
        relationship = item[2]
        par_vars_name_list = item[3]
        for name in par_vars_name_list:
            if name not in var_dict:
                var_dict[name] = 'var_' + str(i)
                i += 1
        if var_name not in var_dict:
            var_dict[var_name] = 'var_' + str(i)
            i += 1
        normalized_dataflow.append((var_dict[var_name], relationship, [var_dict[x] for x in par_vars_name_list]))
    return normalized_dataflow
