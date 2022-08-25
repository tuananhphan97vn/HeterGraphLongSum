We inherit partly code from https://github.com/dqwang122/HeterSumGraph.

Detail for data format used, dependency, rouge-installation can be found in above link.

The command for training process and evaluation are minimally different with several arguments.

Thanks for their work. 


## Data Processing

1. Download Pubmed and Arxiv dataset from https://github.com/armancohan/long-summarization

2. Preprocess data

For pubmed dataset:
```shell
python preprocess_data.py --input_path dataset/pubmed-dataset --output_path dataset/pubmed --task train
python preprocess_data.py --input_path dataset/pubmed-dataset --output_path dataset/pubmed --task val
python preprocess_data.py --input_path dataset/pubmed-dataset --output_path dataset/pubmed --task test
```
For arxiv dataset:
```shell
python preprocess_data.py --input_path dataset/arxiv-dataset --output_path dataset/arxiv --task train
python preprocess_data.py --input_path dataset/arxiv-dataset --output_path dataset/arxiv --task val
python preprocess_data.py --input_path dataset/arxiv-dataset --output_path dataset/arxiv --task test
```

After getting the standard json format, you can prepare the dataset for the graph by PrepareDataset.sh in the project directory. The processed files will be put under the cache directory.


## Train

For training, you can run commands like this:

```shell
python train.py --cuda --gpu 0 --data_dir <data/dir/of/your/json-format/dataset> --cache_dir <cache/directory/of/graph/features> --embedding_path <glove_path> --model [HSG|HDSG] --save_root <model path> --log_root <log path> --lr_descent --grad_clip -m 6 --save_name folder_name --use_doc --n_iter 2 --passage_length 10 --full_data full

## Test

For evaluation, the command may like this:


python evaluation.py --cuda --gpu 0 --data_dir <data/dir/of/your/json-format/dataset> --cache_dir <cache/directory/of/graph/features> --embedding_path <glove_path>  --model [HSG|HDSG] --save_root <model path>  -m 6 --test_model multi --use_pyrouge --passage_length 10 --doc_max_timesteps 150 --n_iter 2 --use_doc  --gpu 0 --batch_size 32


Some options:

- use_doc: whether to use doc_representation for classification
- passage_length: number of sentence to create one passage 
- niter: number iteration for updating value of weight
- *use_pyrouge*: whether to use pyrouge for evaluation. Default is **False** (which means rouge).
  - Please change Line17-18 in ***tools/utils.py*** to your own ROUGE path and temp file path.
- *limit*: whether to limit the output to the length of gold summaries. This option is only set for evaluation on NYT50 (which uses ROUGE-recall instead of ROUGE-f). Default is **False**.
- *blocking*: whether to use Trigram blocking. Default is **False**.
- save_label: only save label and do not calculate ROUGE. Default is **False**.


