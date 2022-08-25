'''
Dung Doan
'''

import argparse
from utils import _get_word_ngrams
import re, json

SENTENCE_START = '<S>'
SENTENCE_END = '</S>'

def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


def greedy_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            return selected
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected)

def format_to_lines(input_path, output_path, task):

    file_path = '{}/{}.txt'.format(input_path, task)
    save_path = '{}/{}.label.jsonl'.format(output_path, task)

    fout = open(save_path, 'w')
    with open(file_path, encoding="utf-8") as fin:
        for line in fin:
            data = json.loads(line)

            article_id = data["article_id"]
            article_text = data["article_text"]
            if len(article_text) <= 1:
                continue

            article_sentences = [e.replace('\n', '').replace('  ', '').strip()
                                 for e in article_text]

            article_sentences_split = [e.replace('\n', '').replace('  ', '').strip().split()
                                  for e in article_text]

            abstract_text = data["abstract_text"]
            abstract_sentences = [e.replace(SENTENCE_START, '').replace(SENTENCE_END, '').strip()
                                  for e in abstract_text]

            abstract_sentences_split = [e.replace(SENTENCE_START, '').replace(SENTENCE_END, '').strip().split()
                                  for e in abstract_text]

            sent_labels = greedy_selection(article_sentences_split, abstract_sentences_split, 3)

            dataset = {}
            dataset["id"] = article_id
            dataset["text"] = article_sentences
            dataset["summary"] = abstract_sentences
            dataset["label"] = sent_labels

            fout.write(json.dumps(dataset) + '\n')

    fin.close()
    fout.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing dataset')

    parser.add_argument('--input_path', type=str, default='dataset/arxiv-dataset', help='The dataset directory.')
    parser.add_argument('--ouput_path', type=str, default='dataset/arxiv', help='The dataset directory.')
    parser.add_argument('--task', type=str, default='train', help='dataset [train|val|test]')

    args = parser.parse_args()

    format_to_lines(args.input_path, args.ouput_path, args.task)



