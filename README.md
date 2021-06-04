# Bounded logit attention

Implementation of the main experiment from https://arxiv.org/abs/2105.14824

<img width="595" alt="git" src="https://user-images.githubusercontent.com/1204232/120814842-2a6bf480-c547-11eb-8faa-48aa0cc25a5a.png">

## Example usage

Train a model with a bounded logit attention module for the Caltech birds 2011 dataset.  
`python3 run.py --dataset caltech_birds2011 --preset BLA`

Weights of trained models are saved in `weights_dir/`
and logs of loss, accuracy, and top-5 accuracy for the uninterpretable baseline model, the model using the soft training-time explanations, and
the model using the hard test-time explanations are saved in `logs/`.

## Usage

#### Dataset
`-d`,`--dataset`  
Name of the dataset to use, options are `{cats_vs_dogs, stanford_dogs, caltech_birds2011}`

#### Preset
`-s`,`--preset`  
Preset configurations (see paper for details) `{L2X-F , BLA, BLA-T, BLA-PH}`
 - `L2X-F`: learning to explain on feature level (fixed size explanations)
 - `BLA`: bounded logit attention (without thresholding)
 - `BLA-T`: bounded logit attention with thresolding
 - `BLA-PH`: post-hoc setup (head is frozen and only the explanation module is trained)

#### Options
(Overwritten by presets where applicable.)  

`-f`, `--fixed-size`    
Produce fixed size explanations (i.e. learning to explain on feature level, not bounded logit attention)

`-t`, `--thresholding`  
Thresholding using `gamma = 0.02`.

`-p`, `--post-hoc`  
Post-hoc setup, i.e. freeze head during training of explanation module

`-r`, `--force-retraining`  
Retrain, rather than using saved weights from `weights_dir/`
