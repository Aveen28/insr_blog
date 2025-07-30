---
layout: post
title: "Solving Time-Dependent PDEs with Implicit Neural Spatial Representations"
categories: [deep-learning, scientific-ml, PDEs]
excerpt_separator: "<!--more-->"
---

## 🧩 Introduction

Solving time‑dependent partial differential equations (PDEs) is fundamental to understanding and predicting a wide range of real‑world processes—from the sweeping currents in the atmosphere and oceans to the flexing and cracking of materials under stress. At their core, these simulations require two key steps:

1. **Time stepping**, where the system’s state is advanced in small increments to capture its temporal evolution.  
2. **Spatial discretization**, where the continuous physical domain is broken into a finite set of points or elements (grids, meshes, or particles) so that the underlying equations can be solved numerically.

Despite decades of refinement, classical spatial discretizations still face three major challenges:

- **Numerical artifacts** such as artificial diffusion or spurious oscillations, which can distort important features and erode accuracy.  
- **Rapidly growing memory requirements**: as you refine the mesh to resolve finer details, the number of elements skyrockets and so does the cost of storing and manipulating them.  
- **Complex adaptivity**: dynamically refining or coarsening the mesh on the fly is often a monumental coding effort and a source of instabilities.

What if we could remove the mesh entirely and let a single, flexible model represent the spatial field? That’s the promise of **Implicit Neural Spatial Representations (INSRs)**. Rather than assigning a variable to each node in a mesh, we represent the entire field—whether it’s fluid velocity, pressure, or material displacement—as a continuous function encoded in the weights of a neural network. As the simulation marches forward in time, we simply update the network’s parameters according to the governing physics, using well‑established time integrators like explicit or implicit schemes.

This mesh‑free approach brings three standout benefits:

- **Fixed memory footprint**: the network’s size stays constant, regardless of how “smooth” or “complex” the solution becomes.  
- **Adaptive resolution**: the neural network can automatically allocate capacity to where it’s needed most, without the overhead of remeshing.  
- **Self‑contained solver**: no external training data is required—INSR learns the solution “on the fly” by minimizing the physics residual itself.

In this post, we’ll dive into how INSRs work, explore their integration with classic time‑stepping methods, and showcase benchmark results on advection, turbulent vortex flows, and nonlinear elastic deformations. While INSRs may demand more computation per time step, they deliver higher accuracy, lower memory usage, and a simplicity of implementation that opens new doors for scientific simulation. Let’s explore this exciting frontier in mesh‑free numerical methods.  

---

## 🌌 What Is an Implicit Neural Spatial Representation (INSR)?

![Implicit Neural Spatial Representation]({{ site.baseurl }}/images/img_1_insr.png)  
*Figure 1: An INSR encodes an entire spatial field in a neural network.*

An **Implicit Neural Spatial Representation (INSR)** is a mesh‑free way to represent any physical field—velocity, pressure, deformation, etc.—as a single continuous function approximated by a neural network.  Instead of storing values at discrete grid points or mesh vertices, we ask:

> **“Given any point in space, what is the field value there?”**

---

### How It Works

1. **Coordinate Query**  
   - You feed the network the coordinates of one point, e.g. `(x, y)` in 2D or `(x, y, z)` in 3D.

2. **Network Inference**  
   - A multilayer perceptron (MLP) with sinusoidal or ReLU activations processes those coordinates.  
   - **All** of its weights jointly determine the output—there is no local “cell” or “element.”

3. **Field Value Output**  
   - The network returns the physical quantity at that location (a scalar or vector).

---

### Key Properties

- **Continuous & Differentiable**  
  The network defines a \(C^\infty\) function. Computing spatial gradients, divergences, or Laplacians is just auto‑diff.

- **Global Support**  
  Every weight influences the field everywhere. This global coupling lets the model capture long‑range correlations naturally.

- **Fixed Memory Footprint**  
  No matter how finely you sample the domain, you only ever store the network’s weights.

- **Adaptive Detail**  
  During training, the network learns to allocate its capacity to complex regions (shocks, vortices, contact fronts) without needing to refine a mesh.

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
