import code_bert_score
from evaluator.CodeBLEU.code_bleu import calculate_code_bleu, calculate_code_bleu_from_lists
from torch import mean


def get_code_bert(reference_file, prediction_file, lang='python'):
    references = [x.strip() for x in open(reference_file, 'r', encoding='utf-8').readlines()]
    predictions = [x.strip() for x in open(prediction_file, 'r', encoding='utf-8').readlines()]
    return get_code_bert_from_list(references, predictions, lang)


def get_code_bert_from_list(references, predictions, lang='python'):
    precision, recall, F1, F3 = code_bert_score.score(cands=predictions, refs=references, lang=lang)
    avg_pre = mean(precision).item()
    avg_rec = mean(recall).item()
    avg_f1 = mean(F1).item()
    avg_f3 = mean(F3).item()
    return avg_pre, avg_rec, avg_f1, avg_f3


def get_code_bleu(reference_file, prediction_file, lang='python'):
    return calculate_code_bleu(reference_file, prediction_file, lang)


def get_code_bleu_from_list(references, predictions, lang='python'):
    return calculate_code_bleu_from_lists(references, predictions, lang)

def extract_labels(code, label="fix"):
    labels = []
    start_label = "<{}/>".format(label)
    end_label = "</{}>".format(label)
    start_index = 0
    end_index = len(code)
    while start_index >= 0 or end_index >= 0:
        start_index = code.find(start_label, start_index)
        end_index = code.find(end_label, start_index)
        if end_index == -1:
            break
        labels.append(code[start_index:end_index + len(end_label)])
        start_index += len(start_label)

    if not labels:
        return "empty"
    return ' '.join(labels)




def print_metrics(references, predictions, lang):
    code_bleu_score = get_code_bleu_from_list([references], predictions, lang=lang)
    code_bert_score_precision, code_bert_score_recall, code_bert_score_f1, code_bert_score_f3 = (
        get_code_bert_from_list(references, predictions, lang=lang))
    print("Code bleu score : ", code_bleu_score)
    exact_matches_list = [i for i in range(len(references)) if references[i] == predictions[i]]
    print("Average Code Bert score precision : ", code_bert_score_precision)
    print("Average Code Bert score recall : ", code_bert_score_recall)
    print("Average Code Bert score f1 : ", code_bert_score_f1)
    print("Average Code Bert score f3 : ", code_bert_score_f3)
    print("Exact match precision : ", str(len(exact_matches_list) / len(references)))