

# Dataset
- We used the general helpful queries dataset used in the paper [Refusal in language models is mediated by a single direction](https://dl.acm.org/doi/10.5555/3737916.3742238)
- Please download the datasets for obtaining the steering vector from the corresponding repository [refusal_direction/dataset/splits](https://github.com/andyrdt/refusal_direction/tree/main/dataset/splits). Specifically the files `harmless_train.json`, `harmless_val.json`.
- Extract the datasets and modify the paths in the run script appropriately.

# To Run
- Adapt the `run.sh` script as needed to run the appropriate experiments.
- Please use the env from `requirements.txt` for all models except Gemma4.
- For any code involving **Gemma4** please use the `requirements_gemma4.txt` env.
