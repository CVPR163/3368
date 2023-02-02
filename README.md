# CVPR 2023-ID-3368

This repo contains the Implementation of CVPR 2023-ID-3368.

## 1.Instructions to Run Our Code

This codebase contain experiments of our proposed BiMeCo based on [LUCIR](https://openaccess.thecvf.com/content_CVPR_2019/papers/Hou_Learning_a_Unified_Classifier_Incrementally_via_Rebalancing_CVPR_2019_paper.pdf) with CIFAR100. 

Our BiMeCo is trained one NVIDIA Tesla V100 GPU.

### Installation

Install the required packages following requirements.txt:

```
pip install -r requirements.txt
```

### CIFAR-100 Experiments w/ LUCIR

No need to download the datasets, everything will be dealt with automatically.

- For 6 tasks, navigate under "src" folder and run:

```
bash BiMeCo_Lucir_C100_T6.sh
```

- For 11 tasks, navigate under "src" folder and run:

```
bash BiMeCo_Lucir_C100_T11.sh
```

- For 26 tasks, navigate under "src" folder and run:

```
bash BiMeCo_Lucir_C100_T26.sh
```
### Results

We have uploaded our training logs in "log". After finishing the training process, you will get the corresponding log files. 

Note: In the log file, 'Average Incremental Accuracy' denotes the final incremental accuracy.

## 2. Acknowledgements

Our code is based on [FACIL](https://github.com/mmasana/FACIL), one of the most well-written CIL library in my opinion:)
