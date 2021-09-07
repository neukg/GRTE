## EMNLP 2021: A Novel Global Feature-Oriented Relational Triple Extraction Model based on Table Filling.


## Requirements
The main requirements are:
- python 3.6
- torch 1.7.0 
- tqdm
- transformers 3.5.1
- bert4keras

## Usage
* **Get pre-trained BERT model**
Download [BERT-BASE-CASED](https://huggingface.co/bert-base-cased) and put it under `./pretrained`.

* **Train and select the model**
```
python run.py --dataset=WebNLG  --train=train  --rounds=4
python run.py --dataset=WebNLG_star   --train=train  --rounds=2
python run.py --dataset=NYT24   --train=train  --rounds=3
python run.py --dataset=NYT24_star   --train=train  --rounds=2
python run.py --dataset=NYT29   --train=train  --rounds=3
```

* **Evaluate on the test set**
```
python run.py --dataset=WebNLG  --train=test  --rounds=4
python run.py --dataset=WebNLG_star   --train=test  --rounds=2
python run.py --dataset=NYT24   --train=test  --rounds=3
python run.py --dataset=NYT24_star   --train=test  --rounds=2
python run.py --dataset=NYT29   --train=test  --rounds=3
```

### Acknowledgement
Parts of our codes come from [bert4keras](https://github.com/bojone/bert4keras).
