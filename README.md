# Physically Driven Hybrid Transformer Hydrological Model (PD-HTHM)

This repository contains the official implementation of the **Physically Driven Hybrid Transformer Hydrological Model (PD-HTHM)**. PD-HTHM is a hybrid hydrological modeling framework developed to improve streamflow simulation under **nonstationary environmental conditions**. Operating at a daily timestep, it uses meteorological inputs—such as precipitation and potential evapotranspiration—and adaptively estimates **time-varying hydrological parameters** to capture evolving catchment responses.

In contrast to purely data-driven sequence models, PD-HTHM integrates **hydrological process constraints** with a transformer-based representation, aiming to achieve both strong predictive performance and physically meaningful parameter dynamics across diverse basins and hydroclimatic regimes.

---

## Paper

If you use this code or any part of this repository, please cite:

**A physics-driven hybrid transformer model for hydrologic simulation under nonstationary environmental conditions.**  
*Journal of Hydrology*, **669**, 135133 (2026).  
DOI: **10.1016/j.jhydrol.2026.135133**

### BibTeX
```bibtex
@article{PDHTHM2026,
  title   = {A physics-driven hybrid transformer model for hydrologic simulation under nonstationary environmental conditions},
  journal = {Journal of Hydrology},
  year    = {2026},
  volume  = {669},
  pages   = {135133},
  doi     = {10.1016/j.jhydrol.2026.135133}
}
