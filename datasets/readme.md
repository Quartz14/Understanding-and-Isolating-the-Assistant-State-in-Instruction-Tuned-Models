


# Dataset EXP1
- True_False dataset set from [representation-engineering/data/facts](https://github.com/andyzoujm/representation-engineering/tree/main/data/facts)
- We used the general helpful queries dataset used in the paper [Refusal in language models is mediated by a single direction](https://dl.acm.org/doi/10.5555/3737916.3742238)
- Please download the datasets for obtaining the steering vector from the corresponding repository [refusal_direction/dataset/splits](https://github.com/andyrdt/refusal_direction/tree/main/dataset/splits). Specifically the files `harmless_train.json`, `harmless_val.json`.


# Dataset EXP2
- AITA-YTA dataset was downloaded from [ELEPHANT repository](https://github.com/myracheng/elephant)
- IF Eval dataset was obtained fom [Hugging Face](https://huggingface.co/datasets/google/IFEval)
- DeceptionBench dataset was downloaded and preprocessed from [Hugging Face](https://huggingface.co/datasets/skyai798/DeceptionBench)



# Dataset EXP3
- Please download the datasets for the case studies from the same repo [em_organism_dir/data](https://github.com/clarifying-EM/model-organisms-for-EM/blob/main/em_organism_dir/data/training_datasets.zip.enc)
- Persona eval dataset source [anthropics/evals/persona](https://github.com/anthropics/evals/tree/main/persona). The exact eval dataset used in the paper is uploaded above - persona5.jsonl



For all experiments, extract the relevant datasets and modify the paths in the run scripts, code appropriately.