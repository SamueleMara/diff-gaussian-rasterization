# Differential Gaussian Rasterization

This repository is based on the official rasterization engine used in:

**"3D Gaussian Splatting for Real-Time Radiance Field Rendering"**  
Kerbl et al., ACM Transactions on Graphics, 2023.

Original implementation:
https://github.com/graphdeco-inria/diff-gaussian-rasterization

If you use this rasterizer in your research, please cite the original paper:

## BibTeX

@Article{kerbl3Dgaussians,
  author  = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
  title   = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
  journal = {ACM Transactions on Graphics},
  volume  = {42},
  number  = {4},
  month   = {July},
  year    = {2023},
  url     = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}

---

# Modifications for Fruit3DGS

This fork includes additional functionality introduced for the **Fruit3DGS** framework.

The modifications are focused on enabling semantic supervision, instance-level reasoning, and 3D fruit localization.

## Added Features

- Semantic supervision support integrated into rasterization outputs
- Top-K Gaussian contributor extraction for instance-aware rendering
- Compatibility with instance embedding clustering pipeline
- Extensions for fruit counting and 3D localization workflows
- Improved integration with custom segmentation-based training pipelines

These changes are specifically designed to support:

**Fruit3DGS: 3D Gaussian Splatting with Semantic-Aware Instance Clustering for Fruit Counting and Localization**

The core rasterization methodology remains based on the original work by Kerbl et al.

---

## Acknowledgement

We gratefully acknowledge the original authors for releasing their implementation.
This fork builds directly upon their rasterization engine while extending it for agricultural 3D perception research.
