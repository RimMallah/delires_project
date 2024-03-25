# Reproduce the SIDD dataset results / experiments
(Code modified from https://github.com/megvii-research/NAFNet)

## NAFNet
The different implementations of the Simple Gate can be found in basicsr/models/archs/NAFNet_arch.py.

### 1. Setup the environment
* Clone the repository, create a python environment, and install the requirements :
  ```
  python -m venv projectNAFNet
  source projectNAFNet/bin/activate
  git clone https://github.com/RimMallah/delires_project.git
  cd delires_project
  pip install -r requirements.txt
  python setup.py develop --no_cuda_ext
  ```

### 2. Data Preparation for the SIDD dataset

##### Download the train set and place it in ```./datasets/SIDD/Data```:

* [google drive](https://drive.google.com/file/d/1UHjWZzLPGweA9ZczmV8lFSRcIxqiOVJw/view?usp=sharing)

* ```python scripts/data_preparation/sidd.py``` to crop the train image pairs to 512x512 patches and make the data into lmdb format.

##### Download the evaluation data (in lmdb format) and place it in ```./datasets/SIDD/val/```:

  * [google drive](https://drive.google.com/file/d/1gZx_K2vmiHalRNOb1aj93KuUQ2guOlLp/view?usp=sharing)

  * it should be like ```./datasets/SIDD/val/input_crops.lmdb``` and ```./datasets/SIDD/val/gt_crops.lmdb```



### 2. Training

* Change the mode argument in the line 51 of the yaml file (options/train/SIDD/NAFNet-width32.yml) according to the change of architecture (experiments mentioned in the report) you want to test. The options are the following :
  - base : the Simple Gate proposed in the original paper
  - chunk4_multiply_concat : divide the input into 4 chunks, multiply them two by two, and concatenate the results
  - chunk4_keep_concat : divide the input into 4 chunks, multiply 2, and concatenate the results with the two other chunks untouched
  - 5_multiply_concat_rest : multiply two chunks of 5 channels and concatenate the results with the rest of the input untouched
  - multiply_first_two : multiply two chunks of 1 channel and concatenate the results with the rest of the input untouched
  - identity : the identity function (same as 0 multiplication)

* Then run the following command to train the model:
  ```
  torchrun --nproc_per_node=1 --master_port=4321 basicsr/train.py -opt options/train/SIDD/NAFNet-width32.yml --launcher pytorch
  ```

* 1 gpu by default. Set ```--nproc_per_node``` to # of gpus for distributed validation.

  


### 3. Evaluation


##### Download the pretrain model in ```./experiments/pretrained_models/```

* [google drive](https://drive.google.com/file/d/1lsByk21Xw-6aW7epCwOQxvm6HYCQZPHZ/view?usp=sharing)

* or you can use the model you trained in the previous step.
    

##### Testing on SIDD dataset	
```
torchrun --nproc_per_node=1 --master_port=4321 basicsr/test.py -opt ./options/test/SIDD/NAFNet-width32.yml --launcher pytorch
```

* Test by a single gpu by default. Set ```--nproc_per_node``` to # of gpus for distributed validation.


## FFDNet
(Code modified from https://www.ipol.im/pub/art/2019/231/)

* to train the model run the following command (supposing you have already done step 1 and 2 of the NAFNet section):
  ```
  python -m venv projectNAFNet
  source projectNAFNet/bin/activate
  cd delires_project/FFDNet
  python train.py
  ```