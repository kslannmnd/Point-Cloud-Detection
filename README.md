## Dirs

- `configs/` - Hydra config for environment, dataset, training and inference.
- `src/r3d_point_cloud_detection/` - project code and softgroup process.
- `scripts/` - fast acess scripts.
- `data/r3d/` - r3d / preprocessed s3dis
- `outputs/` - checkpoints, generated configs, inference outputs.


## Experiments + training

- `notebooks/kaggle_end_to_end_softgroup_r3d.ipynb` and `notebooks/colab_end_to_end_softgroup_r3d.ipynb` are for demonstration of all pipeline

- `notebooks/kaggle_expetiments.ipynb` and `notebooks/colab_expetiments.ipynb` are for experiments

Has baseline checkpoint in `outputs/checkpoints/softgroup_s3dis_pretrained.pth`

Experiment configs are in `configs/training/`:

```text
colab_exp_1_pretrained_low_lr.yaml
colab_exp_2_pretrained_medium_lr.yaml
colab_exp_3_two_stage_backbone_full.yaml
```

Checkpoints are written to `outputs/checkpoints/`:

```text
outputs/checkpoints/colab_exp_1_pretrained_low_lr_full_softgroup_latest.pth
outputs/checkpoints/colab_exp_2_pretrained_medium_lr_full_softgroup_latest.pth
outputs/checkpoints/colab_exp_3_two_stage_backbone_full_full_softgroup_latest.pth
```


To reduce runtime in colab there is a quick setup:

```python
N_TEST_ROOMS = 3
N_TRAIN_ROOMS = 6
TRAIN_VARIANT = "n_rooms"
METRIC_VARIANT = "n_rooms"
```

Full setup: se `TRAIN_VARIANT = "all_trainable_rooms"` and `METRIC_VARIANT = "all_test_rooms"`


## Scripts and tools

| Command | Purpose |
| --- | --- |
| `python scripts/prepare_kaggle_env.py` | Prepares kaggle: system packages, Python packages, and environment settings used by softgroup. |
| `python scripts/install_dependencies_and_build_softgroup.py` | Clones softgroup if needed, installs compatible dependencies and builds the softgroup |
| `python scripts/prepare_s3dis.py` | Prepares s3dis into the folder expected by softgroup. |
| `python scripts/download_softgroup_checkpoints.py` | Downloads pretrained softgroup checkpoints under `outputs/checkpoints/`. |
| `python scripts/patch_softgroup_for_kaggle.py` | Applies project patches to softgroup for the current kaggle/colab. |
| `python scripts/generate_softgroup_configs.py` | Generates softgroup train and inference YAML configs. |
| `python scripts/train.py training=exp_1` | Runs experiment `exp_1` and registers produced checkpoints. |
| `python scripts/compute_s3dis_metrics.py metrics.rooms=all_test_rooms` | Runs s3dis inference/metric computation for all selected test rooms and saves summary CSV/JSON files. |
| `python run_r3d_inference.py data/r3d/2026-05-06--12-50-38.r3d inference.checkpoint=outputs/checkpoints/exp_1_backbone_then_full_full_softgroup_latest.pth` | Converts the given `.r3d` frame, runs softgroup instance inference, builds bbox overlays, and writes interactive html. |

Train experiments on s3dis:

| Command | Purpose |
| --- | --- |
| `python scripts/train.py env=kaggle training=exp_1 training.rooms=all_trainable_rooms` | Runs `exp_1` training config on all trainable s3dis rooms in the kaggle. |
| `python scripts/train.py env=kaggle training=template_train training.epochs=4 training.lr=0.001 training.max_rooms=10` | Runs the template training config for 4 epochs with LR `0.001` on at most 10 train rooms. |
| `python scripts/train.py env=kaggle training=exp_1 'training.rooms=[Area_1/office_1,Area_1/office_2]'` | Runs `exp_1` only on the two selected s3dis rooms. |

Run s3dis inference for one room and save interactive bbox html:

| Command | Purpose |
| --- | --- |
| `python scripts/infer.py env=kaggle dataset=s3dis_preprocessed inference=s3dis inference.checkpoint=outputs/checkpoints/exp_1_backbone_then_full_full_softgroup_latest.pth inference.target.kind=room inference.target.area=Area_5 inference.target.room=office_1` | . |

Compute s3dis metrics for all test rooms, or the first `n` test rooms:

| Command | Purpose |
| --- | --- |
| `python scripts/compute_s3dis_metrics.py env=kaggle dataset=s3dis_preprocessed metrics.checkpoint=outputs/checkpoints/exp_1_backbone_then_full_full_softgroup_latest.pth metrics.rooms=all_test_rooms` | For all room in the test split using the checkpoint. |
| `python scripts/compute_s3dis_metrics.py env=kaggle dataset=s3dis_preprocessed metrics.rooms=all_test_rooms metrics.max_rooms=10` | For the first 10 rooms from the test split. |

Compute s3dis metrics for selected rooms from the test area:

| Command | Purpose |
| --- | --- |
| `python scripts/compute_s3dis_metrics.py env=kaggle dataset=s3dis_preprocessed 'metrics.rooms=[Area_5/office_1,Area_5/office_2]'` | . |

Run r3d inference with a s3dis-trained softgroup checkpoint:

| Command | Purpose |
| --- | --- |
| `python run_r3d_inference.py /kaggle/input/my-r3d/scene.r3d env=kaggle dataset=r3d inference=r3d inference.checkpoint=outputs/checkpoints/exp_1_backbone_then_full_full_softgroup_latest.pth` | Converts one r3d frame to softgroup test scene, runs `tools/test.py`, colors points by predicted instances, builds bbox overlays, and writes result in `point_cloud.html`. |

## Visualization tools


| Command | Purpose |
| --- | --- |
| `show_s3dis_rooms Area_1` | Lists available preprocessed s3dis rooms for `Area_1`. |
| `show_room Area_5 office_1` | Opens an interactive point-cloud view |
| `show_room_instance_seg Area_5 office_1` | Shows predicted instance segmentation. |
| `show_room_bbox Area_5 office_1` | Shows predicted bounding boxes. |
| `show_gt_compare_table Area_5 office_1` | Builds a per-object table comparing predictions with ground truth for the room. |
| `show_gt_compare_plot Area_5 office_1 chair_1` | Shows a focused GT-vs-prediction plot for `chair_1` in the selected room. |
| `precompute_s3dis_inference Area_5 office_1,office_2` | Runs and caches softgroup inference. |

