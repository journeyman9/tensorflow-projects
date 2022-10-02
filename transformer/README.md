# Transformer English to Spanish
Learn a transformer from ingested data, plot losses, load model, 
and use to obtain sequence. Configs are found in config.yaml

## Setup

```bash
conda env update -f environment.yml
```

## Usage

Train and save transformer with multi-head attention
```bash
python transformer.py --train
```

Test using batch data
```bash
python transformer.py --test
```
Provide your own string to translate to spanish
```bash
python transformer.py --eval <string sentence.>
```