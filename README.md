# PriCDR

This repository contains a pytorch implementation of our paper "Differential Private Knowledge Transfer for Privacy-Preserving Cross-Domain Recommendation" accepted by WWW 2022.



## Environment Setup

* Python 3.6+

* PyTorch 1.8+



## Guideline

### data

* We provide a sample dataset for testing in the folder `processed_data`.

### model

* The implementation of PriCDR (`model.py`)
* The implementation of Johnson-Lindenstrauss Transform and Fast Jonhson-Lindenstrauss Tranform (`random_proj_svd.py`)

### utils

* Data preprocessing (`Dataset_PP.py`)
* Model evaluation (`utils.py`)

* Specially, we adopt the evaluation method of CDR from ETL (https://github.com/xuChenSJTU/ETL-master) in this framework.

### Example to run the codes

```python
CUDA_VISIBLE_DEVICES=gpu_num python main.py --eps 0 --sp 0.8 --align 100.0 --et 0.3 
```



## Citation

If you find the code useful, please consider citing the following paper:

```tex
@inproceedings{chen2022differential,
  title={Differential Private Knowledge Transfer for Privacy-Preserving Cross-Domain Recommendation},
  author={Chen, Chaochao and Wu, Huiwen and Su, Jiajie and Lyu, Lingjuan and Zheng, Xiaolin and Wang, Li},
  booktitle={Proceedings of the ACM Web Conference 2022},
  pages={1455--1465},
  year={2022}
}
```