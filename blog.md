---
layout: post
title: "Solving Time-Dependent PDEs with Implicit Neural Spatial Representations"
categories: [deep-learning, scientific-ml, PDEs]
excerpt_separator: "<!--more-->"
---

## 🧩 Introduction

Solving time‑dependent partial differential equations (PDEs) lies at the heart of modeling countless physical phenomena—from the swirling eddies of turbulent fluids to the dynamic deformation of elastic bodies. Traditionally, these simulations break the problem into two pieces:

1. **Time discretization**, marching forward in small steps (e.g. explicit or implicit Euler),  
2. **Spatial discretization**, representing fields on grids, meshes, or particle clouds.

While intuitive, these classical spatial meshes carry three key downsides:

- **Numerical artifacts** (diffusion, dissipation, artificial viscosity) that compromise fidelity,  
- **Growing memory costs** as resolution increases,  
- **Complex adaptivity**, requiring expensive remeshing or data structures.

What if we could bypass explicit meshes altogether? Enter **Implicit Neural Spatial Representations (INSRs)**—a new paradigm that encodes an entire spatial field directly in the weights of a neural network. Instead of storing values at grid nodes, we store a compact function

```math
f_\theta(x)\colon \Omega \to \mathbb{R}^d,
```
---

## 🌌 What is an INSR?

INSR replaces the spatial discretization of fields (e.g. pressure, velocity) with a **continuous function approximated by a neural network**:

$$
x \mapsto f_\theta(x), \quad \text{where } \theta \text{ are trainable weights}
$$

At each timestep, we update this network using gradients derived from the PDE's governing equations.

---

## ⚙️ Method Overview

- Neural network encodes the spatial field at each timestep
- Time integration (e.g., Euler, midpoint, variational integrator) evolves the weights
- Gradient-based optimization minimizes residuals in the PDE
- No training data needed — this is a solver, not a surrogate

---

## 📊 Benchmarks

### 🌀 1. Advection Equation
- Preserves wave shape with high accuracy
- No numerical diffusion
- Midpoint integration

### 🌪️ 2. Incompressible Euler
- Captures multi-scale vortex dynamics
- Outperforms PINNs and DeepONets under same memory limits

### 🧱 3. Elastodynamic Simulations
- Models nonlinear deformations and contact
- Outperforms FEM and MPM in memory-constrained settings

---

## 📈 Benefits

| ✅ Advantages | ❌ Limitations |
|--------------|----------------|
| Mesh-free | Slower runtime |
| Memory efficient | Requires tuning |
| High spatial accuracy | Not real-time yet |
| Easy to integrate with classical schemes | More compute per step |

---

## 🔚 Conclusion

INSRs are a promising alternative to traditional numerical solvers, especially in memory-constrained or adaptive scenarios. They unify deep learning with classical numerical integration — unlocking new ways to simulate complex physical systems.

> 📌 Paper: [Chen et al., ICML 2023](https://arxiv.org/abs/2210.00124)  
> 🎓 Presented at: Advanced Seminar, Universität Stuttgart (2025)

---

## 📬 Stay in Touch

Check out [About Me](/about/) or connect on [GitHub](https://github.com/aveen28) for more deep learning & scientific ML content.
