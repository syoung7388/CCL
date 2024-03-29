# Repository for Contrastive and Consistency Learning(CCL) 

## What is CCL?

Contrastive and Consistency Learning (CCL) is a training method to solve ASR error problems when using pre-trained language models in a modular approach. The CCL method performs token-based contrast learning
followed by consistency learning. 1) Token-based contrastive learning aims to correlate errors in the noisy ASR transcript with the corresponding clean transcript at both utterance and token levels. 2) Consistency learning emphasizes the coherence between clean and noisy latent features to avoid misclassifying the noisy ASR transcriptions. 



## Folder Structure Conventions

    ├── CCL  # CCL foloder(Unzip CCL.zip file to a folder)
        ├── datasets                               # datasets foloder(Unzip datasets.zip file to a folder)
        ├── contrastive_learning.py                # 1) token-based contrastive learning code
        ├── consistency_learning.py                # 2) consistency learning code 
        ├── make_dataset.py                        # dataset, dataloader code 
        ├── models.py                              # models code 
        ├── train_script.sh                        # training scripts (include huyperparameters used per dataset)
        └── README.md



## Setting

Before experimenting, you can make a virtual environment for the project.

```shell
conda create -n slu python=3.8
conda activate slu
pip install -r requirements.txt
```



## Datasets

For training and evaluating NLP tasks, we use benchmark datasets(SLURP, Timers and Such, FSC, SNIPS). Due to capacity issues, we are only releasing the SLURP and FSC datasets. We'll release the rest of the datasets later. 



## Training



#### 1) Token-based contrastive learning

```shell
python contrastive_learning.py
```
Arguments(major components),
* `--dataset`: The dataset paths ('./datasets/slurp' or './datasets/fsc'). 
* `--target`: The ASR module name ('wave2vec2.0' or 'google'). Given a noisy ASR transcript that has been converted to text by the ASR module.
* `--ckpt`: The save paths.
* `--lambda1`: ration for the utterance contrastive objective 
* `--lambda2`: ration for the selective token contrastive objective



#### 2) Consistency learning

```shell
python contrastive_learning.py
```
Arguments(major components),
* `--dataset`: The dataset paths ('./datasets/slurp' or './datasets/fsc'). 
* `--target`: The ASR module name ('wave2vec2.0' or 'google'). Given a noisy ASR transcript that has been converted to text by the ASR module.
* `--ckpt`: The save paths.
* `--tlm_path`: The TLM model (paper: reference network) path saved from token-based contrastive learning. 
* `--ilm_path`: The ILM model (paper: inference network) path saved from token-based contrastive learning.
* `--lambda1`: The ratio of mean squared error between target probabilities in the consistency objective. 
* `--lambda2`:The ratio of mean squared error between the noisy latent feature and the referenced latent feature in the consistency objective. 


##### You can do this by simply running a shell script (includes huyperparameters used per dataset),
```shell
sh train_script.sh
```


## Evaluation

```shell
python do_eval.py
```
Arguments(major components),
* `--dataset`: The dataset paths ('./datasets/slurp' or './datasets/fsc'). 
* `--target`: The ASR module name ('wave2vec2.0' or 'google'). Given a noisy ASR transcript that has been converted to text by the ASR module.
* `--our_model`: the trained model path.
