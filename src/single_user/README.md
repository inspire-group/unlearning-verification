This directory contains the source code allowing rebuilding our evaluation infrastructure. 

- [lib/probabilitybuckets_light.py](lib/probabilitybuckets_light.py):
...The Probability Buckets script computing differential privacy bounds, [copied from GitHub](https://github.com/sommerda/privacybuckets). 
- Directory [poisoned_dataset](poisoned_dataset):
...Contains the classs downloading, preprocessing, and providing the different datasets with or without backdoors.
- [generate_paper_data.py](generate_paper_data.py):
...This file generates results for the following datasets: EMNIST, FEMNIST, cifar10, Amazon5
- [generate-differential-privacy-data.py)(generate-differential-privacy-data.py]:
...This file generates the numbers for the differential privacy statement in the paper.
- [models.py](models.py):
...This file contains the models used to evaluate the datasets. 
- [train_20news.py](train_20news.py):
...This script trains ML models for the 20News dataset and generates the raw data used later in the plotting script 'plot-user-level-poison-accuracies_20News.py'
- [train.py](train.py):
...This file contains base code for generating our evaluations. 


#### Citation

If re-used, please reference the source using the following bibtex entry (the paper links to this repo):
```
@inproceedings{sommer2022athena,
  title={{Athena: Probabilistic Verification of Machine Unlearning}},
  author={Sommer, David M. and SOng, Liwei and Wagh, Sameer and Mittal, Prateek},
  journal={Proceedings on Privacy Enhancing Technologies},
  year={2022}
}
``` 
