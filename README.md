# Athena: Probabilistic Verification of Machine Unlearning

This repository contains the relevant code required to reproduce the results in the paper [here](https://petsymposium.org/2022/paperlist.php). This work was published at [Privacy Enhancing Technologies Symposium (PETS) 2022](https://petsymposium.org).

---
### Table of Contents
- [Requirements](#requirements)
- [Source Code](#source-code)
    - [Installing requirements](#installing-requirements)
    - [Running the code](#running)
- [Citation](#citation)
- [Warning](#warning)
---

### Requirements

* The code should work on most Linux distributions (It has been developed and tested with [Ubuntu](http://www.ubuntu.com/) 21.10).
* The following packages are required: pkg-config libhdf5-dev
* The scripts are written for python3.9, the required packages are listed in [requirements.txt](src/requirements.txt)
* The python3 package manager pip3 is required to be installed.


### Source Code
All the source code is provided in `src/` and `Makefile` is provided inside. 
* `src/single_user`: Code for single users. This is the heart of our project.
* `src/multi_user`: This directory contains scripts analyzing the involvment of collaborating users.
* `src/plotting_scripts`: This directory contains plotting scripts.

#### Installing requirements
Install required packages using the following command (For Ubuntu 21.10): `sudo apt install pkg-config libhdf5-dev` 

#### Running the code
This code runs on `python3`. It requires `python3` and `pip` (with the command "pip" refering to python3, not python2) to be installed. After installing the dependencies, you can run the code with the following comand: `cd src/; make`

Each sub-director contains a `README` file to provide pointers on the relevant experiments. 

### Citation
You can cite the paper using the following bibtex entry (the paper links to this repo):
```
@inproceedings{sommer2022athena,
  title={{Athena: Probabilistic Verification of Machine Unlearning}},
  author={Sommer, David M. and SOng, Liwei and Wagh, Sameer and Mittal, Prateek},
  journal={Proceedings on Privacy Enhancing Technologies},
  year={2022}
}
```


#### Warning
This codebase is released solely as a reference for other developers, as a proof-of-concept, and for benchmarking purposes. 

---
For questions, please create git issues.

