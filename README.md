## pytonh version
```
default=3.8
```
 
## Environment
```shell
pip install -r requirements.txt
```

## Download model and tokenizer
```shell
bash download.sh
```
## preprocess+train
- please make sure env has install install rouge_score. thank you.
### install rouge_score
```shell
pip install rouge_score
```
### train
```shell
python3.8 train.py ${1} ${2}
```
```
${1} --train_file
${2} --eval_file
```
## predict
```shell
python3.8 predict.py ${1} ${2}
```
```
${1} --input_file
${2} --output_file
```

## plot
```
plot.xlsx
```