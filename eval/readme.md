
# To Run

To run the LLM as a judge evaluation please use the scripts here with the evaluation prompt split by each experiment/part in the paper.

## Expt 1 eval
```
python eval_g4_expt1.py --dataset_path "" --sp_type '1'
wait
python eval_g4_expt1.py --dataset_path "" --sp_type '2'

```


## Expt 2 eval
```
python eval_g4_expt2.py --dataset_path "" --model_name ""
wait
python eval_l70_expt2.py --dataset_path "" --model_name ""

```

## Expt 3 eval
```
python eval_g4_expt3.py --dataset_name "" 
```