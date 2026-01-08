
## Dual Consistency Matching for Semi-Supervised Semantic Correspondence

### Prepare Datasets

1) Download PF-PASCAL and PF-WILLOW datasets: [Download](https://www.di.ens.fr/willow/research/proposalflow/)
2) Download SPair-71k dataset: [Download](https://cvlab.postech.ac.kr/research/SPair-71k/data/SPair-71k.tar.gz)
3) Unzip the datasets and place them in the `data` directory
4) Organized as follows
```
./data
    PF-PASCAL
    PF-WILLOW
    SPair-71k
```

### Download pretrained parameters of DINOv2
1) DINOv2: [Download](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth)
3) Modify the checkpoint path in `configs/model_dinov2-b14_base.py`


### Environment Settings

```
conda create -n DCM python=3.8.0
conda activate DCM
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

pip install -r requirements.txt
```

### Evaluation on SPair-71k

```
python test.py --config configs/task_dinov2-b14_448x448_spair.py \
    --model_path semi_supervise_spair/best_ema_model.pth  \
    --log_name test
```

### Evaluation on PF-PASCAL

```
python test.py --config  configs/task_dinov2-b14_448x448_pascal.py \
    --model_path semi_supervise_pfpascal/best_ema_model.pth  \
    --log_name test
```


### Train on SPair-71k
```
python train.py \
    --config configs/task_dinov2-b14_448x448_spair.py \
    --work-dir work_dirs/task_dinov2-b14_448x448_spair  \
    --cfg-options train_dataloader.dataset.data_rate=0.5
```

### Train on PF-PASCAL
```
python train.py \
    --config configs/task_dinov2-b14_448x448_pascal.py \
    --work-dir work_dirs/task_dinov2-b14_448x448_pascal  \
    --cfg-options train_dataloader.dataset.data_rate=0.5
```

### Train on AP10K
```
python train.py \
    --config configs/task_dinov2-b14_448x448_ap10k.py \
    --work-dir work_dirs/task_dinov2-b14_448x448_ap10k  \
    --cfg-options train_dataloader.dataset.data_rate=0.5
```