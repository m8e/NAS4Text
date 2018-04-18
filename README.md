# NAS4Text
Neural Architecture Search on Text Tasks.

## Requirements

- Python 3
- Pytorch
- tqdm (optional)

## Data Format

See docstring of [`libs/utils/data_processing.py`](libs/utils/data_processing.py).

## Train Child Model

```bash
python train.py [More options]

# Example on de-en iwslt dataset
python child_train.py \
    -T de_en_iwslt \
    -H normal \
    --max-tokens 500 \
    --log-interval 10 \
    -N net_code_example/default.json
```

## Decoding Child Model

TODO

## Train Teacher Model

TODO
