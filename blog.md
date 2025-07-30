---
layout: post
title: "Solving Time-Dependent PDEs with Implicit Neural Spatial Representations"
categories: [deep-learning, scientific-ml, PDEs]
excerpt_separator: "<!--more-->"
---

# 🧩 Introduction

Solving time‑dependent partial differential equations (PDEs) is fundamental to understanding and predicting a wide range of real‑world processes—from the sweeping currents in the atmosphere and oceans to the flexing and cracking of materials under stress. At their core, these simulations require two key steps:

1. **Time stepping**, where the system’s state is advanced in small increments to capture its temporal evolution.  
2. **Spatial discretization**, where the continuous physical domain is broken into a finite set of points or elements (grids, meshes, or particles) so that the underlying equations can be solved numerically.

What if we could remove the mesh entirely and let a single, flexible model represent the spatial field? That’s the promise of **Implicit Neural Spatial Representations (INSRs)**. Rather than assigning a variable to each node in a mesh, we represent the entire field—whether it’s fluid velocity, pressure, or material displacement—as a continuous function encoded in the weights of a neural network. As the simulation marches forward in time, we simply update the network’s parameters according to the governing physics, using well‑established time integrators like explicit or implicit schemes.

This mesh‑free approach brings three standout benefits:

- **Fixed memory footprint**: the network’s size stays constant, regardless of how “smooth” or “complex” the solution becomes.  
- **Adaptive resolution**: the neural network can automatically allocate capacity to where it’s needed most, without the overhead of remeshing.  
- **Self‑contained solver**: no external training data is required—INSR learns the solution “on the fly” by minimizing the physics residual itself.

In this post, we’ll dive into how INSRs work, explore their integration with classic time‑stepping methods, and showcase benchmark results on advection, turbulent vortex flows, and nonlinear elastic deformations. While INSRs may demand more computation per time step, they deliver higher accuracy, lower memory usage, and a simplicity of implementation that opens new doors for scientific simulation. Let’s explore this exciting frontier in mesh‑free numerical methods.  

---

# Why Rethink Classical PDE Solvers?

![Why Rethink Classical PDE Solvers?]({{ site.baseurl }}/images/img_insr_2.png)
*Figure 2: Classical time‑dependent PDE solvers require (a) spatial discretization via meshes, grids, or point clouds and (b) temporal discretization via time‑stepping.*

While well‑studied, this two‑step process suffers from:  
- **Numerical artifacts** that degrade solution quality  
- **Rapidly growing memory footprints** as resolution increases  
- **Complex adaptivity**, requiring expensive remeshing  
- **Stability constraints** on timestep size  
- **Artificial dissipation or dispersion**  
- **High computational cost** at fine resolution  

This motivates our search for a mesh‑free spatial representation—enter INSR in the next section.

---

# 🌌 What Is an Implicit Neural Spatial Representation (INSR)?

![Implicit Neural Spatial Representation]({{ site.baseurl }}/images/img_1_insr.png)  
*Figure 1: An INSR encodes an entire spatial field in a neural network.*

An **Implicit Neural Spatial Representation (INSR)** is a mesh‑free way to represent any physical field—velocity, pressure, deformation, etc.—as a single continuous function approximated by a neural network.  Instead of storing values at discrete grid points or mesh vertices, we ask:

> **“Given any point in space, what is the field value there?”**


### How It Works

1. **Coordinate Query**  
   - You feed the network the coordinates of one point, e.g. `(x, y)` in 2D or `(x, y, z)` in 3D.

2. **Network Inference**  
   - A multilayer perceptron (MLP) with sinusoidal or ReLU activations processes those coordinates.  
   - **All** of its weights jointly determine the output—there is no local “cell” or “element.”

3. **Field Value Output**  
   - The network returns the physical quantity at that location (a scalar or vector).


### Key Properties

- **Continuous & Differentiable**  
  The network defines a function. Computing spatial gradients, divergences, or Laplacians is just auto‑diff.

- **Global Support**  
  Every weight influences the field everywhere. This global coupling lets the model capture long‑range correlations naturally.

- **Fixed Memory Footprint**  
  No matter how finely you sample the domain, you only ever store the network’s weights.

- **Adaptive Detail**  
  During training, the network learns to allocate its capacity to complex regions (shocks, vortices, contact fronts) without needing to refine a mesh.

---

# Neural Network Architecture

![SIREN-based Implicit Neural Spatial Representation]({{ site.baseurl }}/images/img_insr_3.png)  
*Figure 3: SIREN MLP architecture used for INSRs.*

For our implicit spatial field representation, we adopt the **SIREN** architecture (Sitzmann et al., 2020). SIRENs are multilayer perceptrons with **sinusoidal activations**, which excel at modeling high‑frequency details and provide smooth, infinitely differentiable outputs—ideal for PDE fields.

1. **Input Encoding**  
   - The network takes a spatial coordinate $$(x,y)\in[-1,1]^2$$.  
   - Optionally, a high‑frequency mapping lifts it into a higher‑dimensional space to help represent fine details.

2. **Hidden Layers**  
   - We use **α hidden layers**, each with **β neurons**.  
   - After each linear layer, a sine activation allows the network to capture oscillatory patterns and sharp transitions.

3. **Output Layer**  
   - A final linear layer (optionally followed by sine) outputs the field value (e.g., velocity component or pressure).

### **Why SIREN?**  
- **High accuracy** on continuous signals  
- **Fast convergence** during training  
- **Captures fine spatial details** without explicit meshing  

---

#Time Integration on Neural Spatial Representations

Our goal is to solve time‑dependent PDEs by evolving the weights of an implicit neural field. After introducing how we represent the spatial field with a neural network (Section 3.1), we now show how to step its parameters forward in time using classical integrators.

### Neural Networks as Spatial Representations

We encode each time‑slice of the field \(f^n(x)\) as a neural network \(f_{\theta^n}(x)\), where \(\theta^n\) are the network weights at time \(t_n\). At any query point \(x \in \Omega\), we simply evaluate:
$$
f^n(x) \;=\; f_{\theta^n}(x).
$$
Because the network’s weights implicitly define the field everywhere, memory usage depends only on \(\dim(\theta)\), not on the number of sample points.

### Time Integration

Given the network weights \(\{\theta^k\}_{k=0}^n\) at previous steps, we obtain the next weights \(\theta^{n+1}\) by solving:
$$
\theta^{n+1} \;=\; 
\arg\min_{\theta}
\sum_{x \in \mathcal{M}\subset\Omega}
\mathcal{I}\bigl(\Delta t,\,
\{f_{\theta^k}(x)\}_{k=0}^{n+1},\,
\{\nabla f_{\theta^k}(x)\}_{k=0}^{n+1},\,\ldots\bigr)
\quad,
$$
where \(\mathcal{I}\) is the discrete‐time integrator objective (e.g., explicit/implicit Euler, midpoint, variational, or operator splitting), and \(\mathcal{M}\) is a random mini‑batch of spatial samples.

To illustrate, here is the pseudocode for our integration loop:

![Time Integration Algorithm]({{ site.baseurl }}/images/img_insr_4.png)  
*Algorithm 1: Time integration of network weights via mini‑batch optimization.*

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
