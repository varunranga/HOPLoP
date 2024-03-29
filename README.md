# HOPLoP: Multi-hop Link Prediction over Knowledge Graph Embeddings

Datasets can be downloaded from [here](https://drive.google.com/drive/folders/1URVS3A_xMZly3X6CgkSuoRxiENRYFWaF?usp=sharing):
1. [NELL-995](http://cs.ucsb.edu/~xwhan/datasets/NELL-995.zip)
2. [FB15K-237](https://drive.google.com/file/d/1klWL11nW3ZS6b2MtLW0MHnXu-XlJqDyA/view?usp=sharing)
3. [WN18RR](https://drive.google.com/drive/folders/1fyKRIWWHtwYS9eOHHpSXN3bUQgKG6rDs?usp=sharing)
4. [YAGO3-10](https://drive.google.com/drive/folders/1s_4d78zwZjGnOH7TNk-qn4T0OlGieAel?usp=sharing)

Results from all experiments can be found in the '(M)HOPLoP Results.xlsx' files in this repository.

Logs can be found [here](https://drive.google.com/drive/folders/15apkojiK--j0jdkQKr6TOaMxb4-3eU9v?usp=sharing).

To run experiments:
1. Download the datasets from [here](https://drive.google.com/drive/folders/1URVS3A_xMZly3X6CgkSuoRxiENRYFWaF?usp=sharing) into a 'Datasets' folder in the (M-)HOPLoP folder. 
2. For M-HOPLoP experiments, download our generated embeddings from [here](https://drive.google.com/drive/folders/1vPM_ImtWJY2wLeLd4kz98WRpltiQSVv2?usp=sharing). Place the 'Embeddings' folder in the M-HOPLoP folder. We reuse the embeddings generated for HOPLoP experiments.
3. After giving appropriate permissions, run the script '/(M-)HOPLoP/run.sh' to reproduce all the results.

Some helper scripts can be found [here](https://drive.google.com/drive/folders/1xe_fCwuw6_N-APr0YAVQhNEluLG1SjgB?usp=sharing), which can be used to create new bash scripts and extract information from result files.
