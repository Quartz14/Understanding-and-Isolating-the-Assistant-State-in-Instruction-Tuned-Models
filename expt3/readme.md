
# Code
- This code builds upon the repository: [model-organisms-for-EM](https://github.com/clarifying-EM/model-organisms-for-EM).
- The hyperparameters were set the same as in the paper [Model Organisms for Emergent Misalignment
](https://arxiv.org/pdf/2506.11613) for consistency.

# Dataset
- Please download the datasets for the case studies from the same repo [em_organism_dir/data](https://github.com/clarifying-EM/model-organisms-for-EM/blob/main/em_organism_dir/data/training_datasets.zip.enc)
- Extract the datasets and modify the paths in the config json files appropriately.

# To Run
- Set the appropriate values for the parameters in the config files. Specifically the `model` , `training_file`, `add_system_prompt` and `save_dir`.
- Adapt the `run.sh` script as needed to run the appropriate experiments.