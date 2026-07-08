# Colab Quickstart Notebook

Clean repository notebook for reproducing ADS JSON artifacts and figures in Google Colab.

File: `colab_quickstart.ipynb`

Use the notebook from the root of the cloned repository:

```bash
cd /content/ads-vit-forensics
```

The notebook uses repository-relative script calls such as `python scripts/...`, so the same command form works both locally and in Colab after changing into the repo root.

Main updates in this version:

- Keeps all Google Drive/model/dataset/output paths in one configuration cell.
- Uses the Colab ImageNet-100 validation path `/content/imagenet100_resized/val`.
- Documents two ImageNet-100 setup paths:
  - unpack a prepared Drive tarball, `MyDrive/pe_experiment/imagenet/imagenet100_resized.tar`;
  - or run `00_setup_imagenet.py` using repo metadata files `data/val_labels.txt` and `data/imagenet100_classes.txt`.
- Adds an offline CIFAR-100 fallback that copies a cached `cifar-100-python` folder from Drive into `/tmp/cifar100` and verifies it with `torchvision.datasets.CIFAR100(download=False)`.
- Adds the shared-δ attack-convention regeneration step:
  - `ads_shared_delta_imagenet100.json`
  - `ads_shared_delta_cifar100.json`
- Adds the ROC rank-AUC sensitivity post-processing step:
  - `scripts/compute_roc_rank_auc_sensitivity.py`
  - `ads_roc_rank_auc_sensitivity.json`
- Runs the patched root-level `reproduce.py`, which now verifies the primary n=6 tables, shared-δ artifacts, and ROC sensitivity artifact.
- Keeps the final artifact inventory/zip cells and optional sync-back-to-repo cells.
- Clears outputs and execution counts for clean GitHub display.

Files that should be present in the repository for the notebook workflow:

```text
data/val_labels.txt
data/imagenet100_classes.txt
00_setup_imagenet.py
00_setup_cifar100.py
scripts/compute_roc_rank_auc_sensitivity.py
scripts/ads_shared_delta_attack_convention.py
reproduce.py
```

ImageNet images and the official ILSVRC2012 validation tar are not redistributed in the public repository. CIFAR-100 is normally downloaded by `torchvision`, but the notebook supports a local Drive cache when the upstream endpoint is unavailable.
