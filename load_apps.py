from datasets import load_dataset
import inspect
import lm_eval.datasets.hendrycks_apps.hendrycks_apps

path = inspect.getfile(lm_eval.datasets.hendrycks_apps.hendrycks_apps)

for name in ['introductory', 'interview', 'competition']:
    load_dataset(path, name=name)
