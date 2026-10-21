# Trained checkpoints

Place the trained BetaTour checkpoint in this folder as `betatour_v2.pt`. It is the file written by notebook 02 on Google Drive:

```
MyDrive/TourFold_TSP/v2/runs/TourFold-v2/ckpt.pt
```

The two models of run v1 can be added in the same way:

```
MyDrive/TourFold_TSP/runs/TourFold/ckpt.pt          -> betatour_knn_v1.pt   (BetaTour-kNN)
MyDrive/TourFold_TSP/runs/single-instance/ckpt.pt   -> single_instance_v1.pt
```

Loading a checkpoint:

```python
import sys, torch
sys.path.insert(0, "src")
import tsp_model as M

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, ck = M.load_model("checkpoints/betatour_v2.pt", device)
print(sum(p.numel() for p in model.parameters()))   # 1,875,787 for BetaTour
```

Notebooks 03 and 04 load the checkpoint from its Drive path, not from this folder.
