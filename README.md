# MIMR
Model-Independent Mask Refiner 


## Installation

```
conda create --name mimr -y python==3.9 

conda activate mimr
```

then install torch depending on your CUDA version from https://pytorch.org/

```
pip install torch_geometric
```

## Commands

### Training

```
python main.py \
path/to/your/dataset \


```

dataset/thin_object_detection/COIFT/masks