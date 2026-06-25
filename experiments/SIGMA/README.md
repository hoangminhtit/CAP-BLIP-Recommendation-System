# SIGMA: Selective Gated Mamba for Sequential Recommendation
This is the implementation of the submission "SIGMA: Selective Gated Mamba for Sequential Recommendation".
## Configuration of the environment
The hardware and software we used are listed below to facilitate the environment's configuration. The detailed environment setting can be found in the `requirements.txt`. You can use pip install to reproduce the environment.
- Hardware:
  - GPU: one NVIDIA L4
  - CUDA: 11.8
- Software:
  - Python: 3.10.13
  - Pytorch: 2.1.1 + cu118
- Usage
  - Install Causal Conv1d
    - `pip install causal-conv1d==1.1.3.post1`
  - Install Recbole
    - `pip install recbole==1.2.0`
  - Install Mamba
    - `pip install mamba-ssm==1.1.4`
A detailed configuration process in Colab can be found in the `RecMamba.ipynb`
## Datasets

This copy has been adapted to use the preprocessed datasets produced by the
main CAP-BLIP recommendation repo. The original SIGMA code expects RecBole
atomic files, so we provide a converter from:

```text
data/preprocessed/{dataset_code}_min_rating{r}-min_uc{u}-min_sc{s}/dataset_single_export.csv
```

to:

```text
experiments/SIGMA/dataset/{sigma_dataset_name}/{sigma_dataset_name}.inter
```

The `.inter` file uses RecBole typed headers:

```text
user_id:token    item_id:token    rating:float    timestamp:float
```

Because `dataset_single_export.csv` does not store the original timestamp, the
converter assigns a deterministic per-user timestamp that preserves the existing
split order: train interactions first, then validation, then test. The SIGMA
config uses leave-one-out splitting, so this reproduces the same train/valid/test
semantics after conversion.

### Prepare Data

Run the main repo preprocessing first:

```bash
python data_prepare.py \
  --dataset_code beauty \
  --min_rating 3 --min_uc 6 --min_sc 6
```

Then convert it for SIGMA:

```bash
python experiments/SIGMA/prepare_sigma_dataset.py \
  --dataset_code beauty \
  --min_rating 3 --min_uc 6 --min_sc 6 \
  --sigma_dataset_name sigma_beauty
```

This creates:

```text
experiments/SIGMA/dataset/sigma_beauty/sigma_beauty.inter
```

You can use another dataset code, e.g. `games`, `sports`, `fashion`, as long as
`data_prepare.py` has already produced the corresponding `dataset_single_export.csv`.

## Model Training

From the repo root:

```bash
python experiments/SIGMA/model/run.py \
  --dataset sigma_beauty \
  --data_path experiments/SIGMA/dataset
```

For a quick smoke test:

```bash
python experiments/SIGMA/model/run.py \
  --dataset sigma_beauty \
  --data_path experiments/SIGMA/dataset \
  --epochs 1 \
  --train_batch_size 256 \
  --eval_batch_size 512
```

For grouped-user evaluation, use:

```bash
python experiments/SIGMA/model/run_custrainer.py \
  --dataset sigma_beauty \
  --data_path experiments/SIGMA/dataset
```

The main configuration is in `model/config.yaml`. CLI arguments passed to
`run.py` override the dataset, data path, epochs, batch size, and GPU id.

## Citation
If you found the code and the paper are useful, please kindly cite our paper:
```bibtex
@article{liu2024bidirectional,
  title={Bidirectional gated mamba for sequential recommendation},
  author={Liu, Ziwei and Liu, Qidong and Wang, Yejing and Wang, Wanyu and Jia, Pengyue and Wang, Maolin and Liu, Zitao and Chang, Yi and Zhao, Xiangyu},
  journal={arXiv preprint arXiv:2408.11451},
  year={2024}
}
```
