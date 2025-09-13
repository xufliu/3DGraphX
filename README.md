<p align="center">
<h1 align="center">3DGraphX: Explaining 3D Molecular Graph Models via Incorporating Chemical Priors (KDD 2025)</h1>

<p align="center">
    <a href="https://dl.acm.org/doi/10.1145/3690624.3709302"><img src="https://img.shields.io/badge/📄-Paper-blue"></a>
    <a href="https://github.com/xufliu/3DGraphX/blob/main/LICENSE"><img src="https://img.shields.io/github/license/xufliu/3DGraphX"></a>
</p>

![](figs/pipeline.png)

**3DGraphX** provides backbone-agnostic explanations for 3D molecular graph models—currently **SchNet** and **DimeNet++**—by incorporating chemical priors (cluster/ring motifs) and applying node masks at well-defined **hook points** in the backbone. The design separates:

- **Backbone hooks** (how to build edges/attributes, where to apply masks, how to read out), and  
- **Explainers** (how masks are parameterized/optimized).

Two usage modes:
- **Transductive**: optimize a mask directly for a given molecule (inside `forward`).
- **Inductive**: train a small MLP to predict cluster masks across molecules.



## Setup Environment

This is an example for how to set up a working conda environment to run the code.

```shell
conda create -n graphx3d python=3.9
conda activate graphx3d

```

### Install PyTorch + PyG

> Torch/PyG wheels are platform-specific. Install them **before** the rest.

### Install remaining dependencies

We provide the requirement file:
  ```bash
  pip install -r requirements.txt
  ```

## Quickstart (Transductive)

Run an end-to-end explanation on **QM9**:

```bash
# SchNet (uses PyG's pretrained helper under the hood)
python main.py --backbone schnet --explainer transductive --epochs 30 --lr 1e-2

# DimeNet++ (default for --backbone dimenet)
python main.py --backbone dimenet --explainer transductive --epochs 30
```

## Notebook Demo

A single-molecule, step-by-step walkthrough:

- `notebooks/tutorial.ipynb`


## License

Released under the **MIT License**. See [LICENSE](LICENSE).

---

## Citations

Feel free to cite this work if you find it useful to you!

```
@inproceedings{liu20253dgraphx,
  title={3DGraphX: Explaining 3D Molecular Graph Models via Incorporating Chemical Priors},
  author={Liu, Xufeng and Luo, Dongsheng and Gao, Wenhan and Liu, Yi},
  booktitle={Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 1},
  pages={859--870},
  year={2025}
}
```
