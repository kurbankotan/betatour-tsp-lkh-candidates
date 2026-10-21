# Trained checkpoint

`betatour_v2.pt` (23.3 MB) is the trained BetaTour model used in runs v2 to v4 of the paper. It is the file that notebook 02 wrote to Google Drive as

```
MyDrive/TourFold_TSP/v2/runs/TourFold-v2/ckpt.pt
```

and that notebooks 03 and 04 loaded from there. Besides the model weights (`model`) and the model settings (`cfg`), the file holds the state of the optimizer and of the learning-rate schedule (`opt`, `sched`), the training history with one record per epoch (`hist`), the index of the last completed epoch counted from 0 (`epoch`) and the gradient-checkpointing flag (`grad_ckpt`).

Loading the checkpoint:

```python
import sys, torch
sys.path.insert(0, "src")
import tsp_model as M

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, ck = M.load_model("checkpoints/betatour_v2.pt", device)
print(sum(p.numel() for p in model.parameters()))   # 1,875,787
print(ck["epoch"] + 1)                               # 12 completed epochs
```

`load_model` calls `torch.load(..., weights_only=False)`, which unpickles the file, so load only checkpoints from sources you trust.

The two models of run v1 (BetaTour-kNN and the single-instance model, Appendix A.6 of the paper) are not included. On Google Drive they are

```
MyDrive/TourFold_TSP/runs/TourFold/ckpt.pt          (BetaTour-kNN)
MyDrive/TourFold_TSP/runs/single-instance/ckpt.pt   (single-instance model)
```
