import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from src.figures.show_metrics import show_metrics
from src.metrics.score import score
from src.figures.name_getters import get_model_name, get_dataset_name
models = ['GRAE_100', 'AE', 'EAERMargin', 'TopoAE', 'DiffusionNet', 'UMAP']
datasets = ['Teapot', 'SwissRoll', 'RotatedDigits', 'UMIST', 'Tracking', 'IPSC']

# score(id_='final', models=models, datasets=datasets)
df = show_metrics(id_='final', model_list=models, dataset_list=datasets, split='test')

df = df.replace(to_replace='GRAE_100', value='GRAE')

models = [get_model_name(m) for m in models]
datasets = [get_dataset_name(m) for m in datasets]

sns.set_theme()
sns.set_style('whitegrid')

# print(df.loc['Teapot', :])

fig, axs = plt.subplots(2, 3, figsize=(3 * 5.5, 2 * 4.5) )

for i, ax in enumerate(axs.reshape(-1)):
    d = datasets[i]
    g =sns.barplot(x=models, y=df.loc[d, :]['MSE'], ax=ax)
    g.set_title(d, fontsize=20)
    # g.set_ylim([0, 1])

    if i != 0 and i != 3:
        g.set_ylabel('')

plt.show()
