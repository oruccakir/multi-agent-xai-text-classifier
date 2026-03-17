# COMMANDS

### Create the conda environment
```bash
conda env create -f environment.yml
```
### Activate the conda environmet
```bash
conda activate xai-classifier
```
### Run the app
```bash
streamlit run app/Home.py
```


 python scripts/train_experiment.py --config configs/imdb.yaml                                                                                  
 python scripts/train_experiment.py --config configs/ag_news.yaml                                                                                 
 python scripts/train_experiment.py --config configs/turkish_sentiment.yaml                                                                       
 python scripts/train_experiment.py --config configs/turkish_news.yaml   