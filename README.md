# Code for paper "Who leads? Who follows? Temporal dynamics of political dogwhistles in Swedish online communities"

## Building diachronic embeddings
### Collect data from Språkbanken
#### Step 1: Retrieve data

```
python .\data\all_spraak.py collect DIRECTORY_A .\config\CONFIG_FILE
```

Variable `CONFIG_FILE`: 
* For Flashback data `CONFIG_FILE` = fb_plt_spraak_data.json. 
* For Familjeliv data `CONFIG_FILE` = fm_smh_spraak_data.json.

#### Step 2: Systematic sampling of data
Distribute data from `DIRECTORY_A` quarterly in `DRIECTORY_B`.

```
python .\data\systematic_sampler.py DIRECTORY_A DIRECTORY_B --interval quarterly
```
#### Step 3: Preprocess
Preprocess data given configurations in `config\pp_bert-config.json`: remove emojis, remove numbers, remove punctuations, remove urls. 

```
python data\preprocess.py DIRECTORY_B DIRECTORY_C --config config\pp_bert-config.json
```

#### Step 4: Re-organize data for vectorisation
Collect examples for dogwhistle expressions. NB. One and the same sentence can contain more than one dogwhistle term. 
```
python data\context_collector.py DIRECTORY_C DIRECTORY_D utils\dwts.paradigm
```

### Vectorization
Vectorize examples with dogwhistle expressions with Swedish Sentence-BERT (HuggingFace).

```
python bert\bert.py DIRECTORY_D VECTORS cuda --sentence_transformer --model KBLab/sentence-bert-swedish-cased
```

### Diachronic vectors
```
python bert\diachronic_vectors.py VECTORS DIACHRONIC_VECTORS --use_merger
```

## Corpus frequencies
### Step 1. Retrieve corpus frequencies
For Flashback:
```
python .\data\corp_freq.py XXX JSON_FILE_FB --granularity m
```

For Familjeliv:
```
python .\data\corp_freq.py familjeliv-allmanna-samhalle JSON_FILE_FL --granularity m
```

### Step 2: Build dataframe

```
python .\create_frqdf.py JSON_FILE DIACHRONIC_VECTORS CSV_FILE --intervall quarterly
```

## Embeddings for replacements
(Replacment data [csv] is availible on reasonable request.)

### Extract replacements from two-wave data
```
python data\replacement_extractor.py REPLACEMENT_CSV REPLACEMENT_DIR --exclude "ordningreda sjalvstandig" --collapse
```

### Vectorization
```
python bert\bert_repl_vec.py REPLACEMENT_DIR combined --device cuda --sentence_transformer --huggingface_model KBLab/sentence-bert-swedish-cased
```

### In-group and Out-group data
```
python .\bert\io_vec.py .\config\io_base_sbert.json DIACHRONIC_VECTORS --replacement_root ROOT_DIR_REPLACEMENTS --save_vectors --log_name sbert
```

## Analysis 
```
analysis/analysis.ipynb
```