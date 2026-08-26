

# PART 1 Expts

python expt1_tf_steer.py --alpha '0.8' # other alpha values can be tried as needed

# PART 2 Expts

python expt2_gens.py --dataset_path '' --model_path '' --output_path ''
# Steered generations using precomputed vectors, assuming the steering vectors are from extraction_artifacts directory
python expt2_steer.py --alpha '-1' --dataset_path '' --model_path ''

