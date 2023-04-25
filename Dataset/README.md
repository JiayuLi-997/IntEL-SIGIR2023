## Dataset

### Dataset Format

We have preprocessed the original Tmall dataset into multiple window formats.

All data for model training is under the `dataset` folder, named in the format of `Tmall_AugSep_window{window_id}/{action_type}`:
- `.csv` format, column name with prefix and suffix:
    - Prefix: `u_` means user features，`i_` means item features，`c_` means context features
    - Suffix: `_i` means integer type，`_f` means float types, `_c` means categorical features，`_s` means sequential features，`_o` means others
- `train/val/test.csv`: interaction data file for train, validation and test
    - One interaction each line, sorted by timestamp
    - Including user ID, item ID, label and timestamp
- `user/item.csv`: user and item features file
    - ID start from 0
    - One user/item each line
- `val/test_iids.csv`: negative samples for validation and test
    - Corresponds to the validation set and test set row by row

### Expected Output Format for IntEL

See the `.csv` files under the `dataset/Tmall_toy` directory.

## Preprocess

### Negative Sampling

You can find the negative sampling process in `preprocess/Tmall.py`. It will generate `val/test_iids.csv` files.

## Training

### DeepFM

You can train the model in any framework with the hyper parameters showed below.

For click:

```yaml
batch_size: 512
lr: 0.001
l2: 0
layers: '[64,64]'
val_metrics: 'ndcg@10.20,hit@10.20'
test_metrics: 'ndcg@5.10.20.50,hit@5.10.20.50,recall@10.20.50,precision@10.20.50'
val_sample_n: 1000
test_sample_n: 1000
```

For favorite:

```yaml
batch_size: 512
lr: 0.0001
l2: 0
layers: '[64]'
val_metrics: 'ndcg@10.20,hit@10.20'
test_metrics: 'ndcg@5.10.20.50,hit@5.10.20.50,recall@10.20.50,precision@10.20.50'
val_sample_n: 1000
test_sample_n: 1000
```

For buy:

```yaml
batch_size: 512
lr: 0.0001
l2: 0
layers: '[64]'
val_metrics: 'ndcg@10.20,hit@10.20'
test_metrics: 'ndcg@5.10.20.50,hit@5.10.20.50,recall@10.20.50,precision@10.20.50'
val_sample_n: 1000
test_sample_n: 1000
```

> `val_sample_n` and `test_sample_n` represents the number of negative samples during validation and testing, respectively.
