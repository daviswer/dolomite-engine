from dolomite_engine.hf_models import export_to_huggingface


load_path = "/gpfs/davis/results/granite_tune/7b-4x/hf-convert/"
save_path = "/gpfs/davis/results/granite_tune/7b-4x/hf-dolo-convert/"

# export to HF llama
export_to_huggingface(load_path, save_path, model_type="granitemoeshared")
