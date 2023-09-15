# ProteinStability
Neural network model for protein stability. Code from a rewrite that reproduces work in Wulab. Specifically, please refer to [this article](https://www.biorxiv.org/content/10.1101/2023.08.09.552725v1.full.pdf).


## Prerequisites
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install -r requirements.txt
conda install -c salilab dssp==3.0.0
conda install -c anaconda libboost==1.73.0
```

## Usage
### Processing protein structure data
```bash
python dataset.py
```

### training model
```bash
python model.py
```

