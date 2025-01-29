# Lightspeed Geometric Dataset Distances via Projections

## Environment Installation

### Via Conda (recommended)

If you use [ana|mini]conda , you can simply do:

```
conda env create -f environment.yaml python=3.8
conda activate otdd
conda install .
```

(you might need to install pytorch separately if you need a custom install)

### Via pip

First install dependencies. Start by install pytorch with desired configuration using the instructions provided in the [pytorch website](https://pytorch.org/get-started/locally/). Then do:
```

conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch

pip install -r requirements.txt
```
Finally, install this package:
```
pip install .
```


## Experiment Scripts

### Correlation Experiment

For MNIST dataset:
```
python3 correlation_mnist_experiment.py
```

For CIFAR10 dataset:
```
python3 correlation_cifar10_experiment.py
```

### Runtime Experiment

For MNIST dataset:
```
python3 split_mnist_runtime.py --parent_dir saved_runtime_mnist --method hswfs
python3 split_mnist_runtime.py --parent_dir saved_runtime_mnist --method wte
python3 split_mnist_runtime.py --parent_dir saved_runtime_mnist --method sotdd
python3 split_mnist_runtime.py --parent_dir saved_runtime_mnist --method otdd_exact
python3 split_mnist_runtime.py --parent_dir saved_runtime_mnist --method otdd_ga
```

For CIFAR10 dataset:
```
python3 split_cifar10_runtime.py --parent_dir saved_runtime_cifar10 --method hswfs
python3 split_cifar10_runtime.py --parent_dir saved_runtime_cifar10 --method wte
python3 split_cifar10_runtime.py --parent_dir saved_runtime_cifar10 --method sotdd
python3 split_cifar10_runtime.py --parent_dir saved_runtime_cifar10 --method otdd_exact
python3 split_cifar10_runtime.py --parent_dir saved_runtime_cifar10 --method otdd_ga
```

### *NIST Experiment
```
python3 adaptation_nist.py
```

### Augmentation Experiment
```
bash augmentation.sh
```

### Text Classification Experiment

Train baseline:
```
python3 text_cls_baseline.py
```

Pretrain:
```
bash text_pretrain.sh
```

Transfer learning:
```
bash text_transfer.sh
```

Compute distance for each method
```
python3 text_dist.py --method sotdd --max_size 50000
python3 text_dist.py --method otdd --max_size 5000
```