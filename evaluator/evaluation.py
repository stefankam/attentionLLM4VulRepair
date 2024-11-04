import os

from evaluator.metrics_getter import extract_labels, get_code_bleu_from_list, \
    get_code_bert_from_list

vulnerability = "command_injection"
model_name = "s2s"

references_file_path = os.getcwd() + "/model/pretrained_model/{}/{}/output/reference.txt".format(model_name, vulnerability)
prediction_file_path = os.getcwd() + "/model/pretrained_model/{}/{}/output/predictions.txt".format(model_name, vulnerability)

references_labels = [extract_labels(x) for x in open(references_file_path, 'r', encoding='utf-8').readlines()]
predictions_labels = [extract_labels(x) for x in open(prediction_file_path, 'r', encoding='utf-8').readlines()]

exact_matches = 0
for i in range(len(references_labels)):
    if references_labels[i] == predictions_labels[i]:
        exact_matches += 1
print("Exact match rate: " + str((exact_matches * 1.0) / len(references_labels)))


processed_references = [x.strip().replace('\t', '\n').replace(r"<\fix>", "").replace("<fix\>", "")
                        for x in open(references_file_path, 'r', encoding='utf-8').readlines()]
processed_predictions = [x.strip().replace('\t', '\n').replace(r"<\fix>", "").replace("<fix\>", "")
               for x in open(prediction_file_path, 'r', encoding='utf-8').readlines()]

code_bleu_score = get_code_bleu_from_list(processed_references, processed_predictions)
code_bert_score_precision, code_bert_score_recall, code_bert_score_F1, code_bert_score_f3 = (
    get_code_bert_from_list(processed_references, processed_predictions))

print("Code Bleu :" + str(code_bleu_score))
print("Code Bert Precision :" + str(code_bert_score_precision))
print("Code Bert Recall : " + str(code_bert_score_recall))