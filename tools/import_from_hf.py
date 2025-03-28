from dolomite_engine.hf_models import import_from_huggingface


load_path = "/gpfs/davis/dmf-library/dmf_models/granite-4.0-30b-a6b-base-preview-4k-r250317a"
save_path = "/gpfs/davis/granites/granite-30b-2/"

import_from_huggingface(load_path, save_path, model_type="granitemoeshared")
