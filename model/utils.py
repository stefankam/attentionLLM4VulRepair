from evaluator.metrics_getter import get_code_bleu_from_list, get_code_bert_from_list


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