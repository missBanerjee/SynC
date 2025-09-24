Official code of SynC : Language-assisted Feature Representation and Lightweight Active Learning For On-the-Fly Category Discovery
# SynC and SynC-AL
Code for SynC and its extended version SynC-AL

## Requirements
loguru
numpy
pandas
scikit_learn
scipy
torch==1.10.0
torchvision==0.11.1
tqdm

## Data & Pre-trained Weights
We use fine-grained benchmarks in this paper, including:
The Semantic Shift Benchmark (SSB)[CUB and Stanford Cars]
We also use generic object recognition datasets, including:
CIFAR-10/100 and ImageNet


## Config
Set paths to datasets and desired log directories in config.py

## Training
- `nearest_mean.py` for training SynC for all  generic datasets except ImageNet-100
- `nearest_mean_fine.py` for training SynC for all fine-grained datasets
- `nearest_mean_imagenet.py` for training SynC for ImageNet-100
- `nearest_cls_align.py` for classifier alignment (SynC-AL)for all generic datasets except ImageNet-100
- `fine.py` for classifier alignment (SynC-AL)for all fine-grained datasets
- `nearest_cls_align_imagenet.py` for classifier alignment (SynC-AL) for ImageNet-100

## Scripts
In the scripts file,select the python file according to the dataset you desire to train the model.Ensure that correct text embedding file is given as input.Set the dataset name and run. 
bash variable_threshold.sh
For SynC, in the python file, ensure you are running the "test_on_the_fly_CA" during inference. Again in the scripts file,select the python file according to the dataset you desire to train the model.Set the dataset name and run. 
bash classifier_alignment.sh
For SynC-AL, in the python file, ensure you are running the "test_on_the_fly_active3" during inference. Again in the scripts file,select the python file according to the dataset you desire to train the model.Set the dataset name and run. 
bash classifier_alignment.sh
Then check the results starting with 'Train Accuracies after classifier alignment:'

# On-the-fly-Category-Discovery
Code release for "On-the-fly Category Discovery" (CVPR 2023）


**Abstract**: Contemporary deep learning models are very successful in recognizing predetermined categories, but often struggle when confronted with novel ones, constraining their utility in
the real world. Identifying this research gap, On-the-fly Category Discovery aims to enable machine learning systems trained on closed labeled datasets to promptly discern between
novel and familiar categories of the test-images encountered in an online manner (one image at a time), along with clustering the different new classes as and when they are encountered.
To address this challenging task, we propose SynC, a pragmatic yet robust framework that capitalizes on the presence of category names within the labeled datasets and the power-
ful knowledge-base of Large Language Models to obtain unique feature representations for each class. It also dynamically updates the classifiers of both the seen and novel classes for
improved class discriminability. An extended variant, SynC-AL incorporates a lightweight active learning module to mitigate errors during inference, for long-term model deployment.
Extensive evaluation show that SynC and SynC-AL achieve state-of-the-art performance across a spectrum of classification datasets.


## Requirements
- python 3.8
- CUDA 10.2
- PyTorch 1.10.0
- torchvision 0.11.1

## Data & Pre-trained Weights
You may refer to this [repo](https://github.com/sgvaze/generalized-category-discovery) to download the datasets and pre-trained model weights as we followed most of the settings in **Generalized Category Discovery**.



## Contact
Thanks for your attention!
If you have any suggestion or question, you can leave a message here or contact us directly:
- anweshabaner@iisc.ac.in
- somabiswas@iisc.ac.in

## Acknowledgement
Our code is mainly built upon [Generalized Category Discovery](https://github.com/sgvaze/generalized-category-discovery). We appreciate their unreserved sharing.
