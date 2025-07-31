---
layout: post
title: "Solving Time-Dependent PDEs with Implicit Neural Spatial Representations"
categories: [deep-learning, scientific-ml, PDEs]
excerpt_separator: "<!--more-->"
---

# Introduction

Solving time‑dependent partial differential equations (PDEs) is fundamental to understanding and predicting a wide range of real‑world processes from the sweeping currents in the atmosphere and oceans to the flexing and cracking of materials under stress. At their core, these simulations require two key steps:

1. **Time stepping**: where the system’s state is advanced in small increments to capture its temporal evolution.  
2. **Spatial discretization**: where the continuous physical domain is broken into a finite set of points or elements (grids, meshes, or particles) so that the underlying equations can be solved numerically.

What if we could remove the mesh entirely and let a single, flexible model represent the spatial field? That’s the promise of **Implicit Neural Spatial Representations (INSRs)**. Rather than assigning a variable to each node in a mesh, we represent the entire field whether it’s fluid velocity, pressure, or material displacement as a continuous function encoded in the weights of a neural network. As the simulation marches forward in time, we simply update the network’s parameters according to the governing physics, using well‑established time integrators like explicit or implicit schemes. 

---

# Why Rethink Classical PDE Solvers?

![Why Rethink Classical PDE Solvers?]({{ site.baseurl }}/images/img_insr_2.png)
*Figure 1: Classical time‑dependent PDE solvers require (a) spatial discretization via meshes, grids, or point clouds and (b) temporal discretization via time‑stepping.*

While well‑studied, this two‑step process suffers from:  
- **Numerical artifacts** that degrade solution quality  
- **Rapidly growing memory footprints** as resolution increases  
- **Complex adaptivity**, requiring expensive remeshing  
- **Stability constraints** on timestep size  
- **Artificial dissipation or dispersion**  
- **High computational cost** at fine resolution  

This motivates our search for a mesh‑free spatial representation enter INSR in the next section.

---

# What Is an Implicit Neural Spatial Representation (INSR)?

![Implicit Neural Spatial Representation]({{ site.baseurl }}/images/img_1_insr.png)  
*Figure 2: An INSR encodes an entire spatial field in a neural network.*

An **Implicit Neural Spatial Representation (INSR)** is a mesh‑free way to represent any physical field velocity, pressure, deformation, etc. as a single continuous function approximated by a neural network.  Instead of storing values at discrete grid points or mesh vertices, we ask:

> **“Given any point in space, what is the field value there?”**


### How It Works

1. **Coordinate Query**  
   - You feed the network the coordinates of one point, e.g. `(x, y)` in 2D or `(x, y, z)` in 3D.

2. **Network Inference**  
   - A multilayer perceptron (MLP) with sinusoidal or ReLU activations processes those coordinates.  
   - **All** of its weights jointly determine the output there is no local “cell” or “element.”

3. **Field Value Output**  
   - The network returns the physical quantity at that location (a scalar or vector).


---

# Neural Network Architecture

![SIREN-based Implicit Neural Spatial Representation]({{ site.baseurl }}/images/img_insrs_3.png)  
*Figure 3: SIREN MLP architecture used for INSRs.*

For our implicit spatial field representation, we adopt the **SIREN** architecture. SIRENs are multilayer perceptrons with **sinusoidal activations**, which excel at modeling high‑frequency details and provide smooth, infinitely differentiable outputs ideal for PDE fields.

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

# Time Integration on Neural Spatial Representations

Our goal is to solve time‑dependent PDEs by evolving the weights of an implicit neural field. We now show how to step its parameters forward in time using classical integrators.

### Neural Networks as Spatial Representations

We encode each time‑slice of the field $$f^n(x)$$ as a neural network $$f_{\theta^n}(x)$$, where $$\theta^n$$are the network weights at time $$t_n$$. At any query point $$x \in \Omega$$, we simply evaluate:
$$
f^n(x) \;=\; f_{\theta^n}(x).
$$
Because the network’s weights implicitly define the field everywhere, memory usage depends only on \(\dim(\theta)\), not on the number of sample points.

### Time Integration

Given the network weights $$\{\theta^k\}_{k=0}^n$$ at previous steps, we obtain the next weights $$\theta^{n+1}$$ by solving:

$$
\theta^{n+1} \;=\; 
\arg\min_{\theta}
\sum_{x \in \mathcal{M}\subset\Omega}
\mathcal{I}\bigl(\Delta t,\,
\{f_{\theta^k}(x)\}_{k=0}^{n+1},\,
\{\nabla f_{\theta^k}(x)\}_{k=0}^{n+1},\,\ldots\bigr)
\quad,
$$

where $$\mathcal{I}$$ is the discrete‐time integrator objective (e.g., explicit/implicit Euler, midpoint, variational, or operator splitting), and $$\mathcal{M}$$ is a random mini‑batch of spatial samples.

To illustrate, here is the pseudocode for our integration loop:

![Time Integration Algorithm]({{ site.baseurl }}/images/img_itsrs_4.png)  
*Algorithm 1: Time integration of network weights via mini‑batch optimization.*

### Boundary Conditions

PDEs often come with spatial boundary conditions (e.g., Dirichlet or Neumann). We enforce these by adding a penalty term over boundary samples $$\mathcal{M}_b \subset \partial \Omega$$. Concretely, at each time step we solve:

$$
\theta^{n+1} = \arg\min_{\theta}
\Biggl\{
\sum_{x\in \mathcal{M}}
\mathcal{I}\bigl(\Delta t,\{f_{\theta^k}(x)\}_{k=0}^{n+1},\ldots\bigr)
\;+\;
\lambda
\sum_{x_b\in \mathcal{M}_b}
C\bigl(f_{\theta}(x_b), \nabla f_{\theta}(x_b), \ldots\bigr)
\Biggr\},
$$

where  
- $$\mathcal{I}$$ is the time‐integrator loss,  
- $$C(\cdot)$$ penalizes violation of the prescribed boundary behavior at each boundary point $$x_b$$,  
- $$\lambda$$ balances the physics objective against boundary enforcement.

### Initial Condition

To initialize the network at $$t=0$$, we fit it to a known initial field $$\hat f^0(x)$$ by minimizing the squared error over a batch of sample points $$\mathcal{M}\subset \Omega$$:

$$
\theta^0 \;=\;
\arg\min_{\theta}
\sum_{x\in \mathcal{M}}
\bigl\|\,f_{\theta}(x) \;-\; \hat f^0(x)\bigr\|^2.
$$

We again solve this via Adam on mini‑batches, yielding a network whose predictions exactly match the given initial condition.  

---

# Method Overview

We evaluate our INSR‑based solver across three canonical time‑dependent PDEs. Each will be treated in detail in the following sections, with full equations and algorithmic specifics.

### 1. Advection Equation  
A linear transport problem where a scalar field $$u(x,t)$$ is carried along by a prescribed velocity.  
- **Physical phenomenon**: Passive tracers, pollutant transport, level‑set propagation  
- **Key challenge**: Avoiding numerical diffusion and preserving sharp features over long time marches  
- **Our approach**: We embed $$u(x)$$ in a SIREN network and step forward via an energy‑preserving midpoint or implicit Euler integrator, solving a small optimization at each timestep  

### 2. Incompressible Euler Equations  
The governing equations for ideal (inviscid), divergence‑free fluid flow.  
- **Physical phenomenon**: Vortex dynamics, turbulence onset, vortex–vortex interactions  
- **Key challenge**: Enforcing incompressibility $$\nabla \!\cdot\! u = 0$$ while capturing multiscale vortical structures without excessive smoothing  
- **Our approach**: We represent velocity and pressure each as implicit neural fields, then apply a Chorin‑style operator‑splitting (advection, pressure projection, velocity correction), each cast as an optimization over network weights  

### 3. Elastodynamic Equation  
The second‑order PDE describing large deformations in hyperelastic solids.  
- **Physical phenomenon**: Vibrations, wave propagation, contact and impacts in elastic bodies  
- **Key challenge**: Balancing kinetic and elastic energies, handling collisions or contact constraints without remeshing  
- **Our approach**: We encode the deformation map $$\varphi(x)$$ in a neural network and perform variational time integration (discrete Hamilton’s principle) by minimizing an incremental potential at each timestep, augmented with soft contact penalties  

*The mathematical formulations, sampling strategies, and hyperparameter settings for each method are detailed in the following sections.*  

---

# 1. Advection Equation

The **advection equation** is one of the simplest time‑dependent PDEs, yet it highlights the core challenge of numerical transport:

> **PDE:**  
> $$
> \frac{\partial u(x,t)}{\partial t} \;+\; (a \,\cdot\, \nabla)u(x,t) \;=\; 0,
> $$
> where $$u(x,t)$$ is the advected scalar field and $$a$$ is a constant velocity vector.

Despite its linearity, discretizing this equation on a mesh often introduces **numerical diffusion** (smearing of sharp features) or **numerical dispersion** (unphysical oscillations). Our goal is to show how an Implicit Neural Spatial Representation (INSR) can **dramatically reduce diffusion**, under tight memory budgets, at the cost of extra compute.

### Time Integrators

**Midpoint Rule** (second‑order, energy preserving):  

$$
u^{n+1}(x) = u^n(x) \;+\; \Delta t\,\bigl[a\cdot\nabla\bigl(\tfrac{u^n(x)+u^{n+1}(x)}{2}\bigr)\bigr].
$$

- On the grid, this is a linear solve per step.  
- In INSR, we **optimize** $$\theta$$ so that the above residual is minimized over a set of sample points $$\mathcal{M}$$.

**Implicit Euler** (first‑order, dissipative):  

$$
u^{n+1}(x) = u^n(x) \;+\; \Delta t\,\bigl[a\cdot\nabla u^{n+1}(x)\bigr].
$$

- We include this variant (“Ours–implicit”) to illustrate the effect of artificial damping.

Below, we evaluate INSR on two canonical testbeds: a 1D Gaussian pulse and a 2D two‑vortex flow.

### 1.1 1D Gaussian Pulse

In this test we advect a narrow Gaussian pulse across a one‑dimensional domain to evaluate how well each method preserves sharp features over long times.

#### Problem Setup

1. **Domain:**  
   $$x\in[-2,2]$$

2. **Initial Condition:**  
   A narrow Gaussian centered at \(-1.5\):  
   $$
   u(x,0) = \exp\!\biggl(-\frac{(x + 1.5)^2}{2\sigma^2}\biggr), 
   \quad \sigma=0.1.
   $$

3. **Velocity:**  
   $$a=0.25$$ (moves the pulse to the right)

4. **Time Integration:**  
   $$\Delta t=0.05$$, total $$240$$ steps to $$t=12\,$$s.

5. **Boundary Conditions:**  
   $$u(-2,t)=u(2,t)=0.$$

#### Representations & Memory

To isolate spatial discretization effects, **both** INSR and the finite‑difference grid use **3.520 KB**:

- **INSR (“Ours”):**  
  SIREN MLP with $$\alpha=2$$ hidden layers, $$\beta=20$$ neurons each.

- **Grid:**  
  Uniform grid of $$901$$ points with midpoint integration.

### Quantitative & Qualitative Results

#### Error over Time & Wave Profiles

![1D Transport: MAE & Wave Snapshots]({{ site.baseurl }}/images/img_insr__5.png)  
*Figure 4: (Left) Mean absolute error over time. (Center) Profiles at $$t=3\,$$s. (Right) Profiles at $$t=12\,$$s.*

1. **MAE Curves (left panel):**  
   - **Ours (midpoint, blue):** MAE remains nearly flat under $$0.005$$ throughout $$12\,$$s, showing **virtually no diffusion**.  
   - **Grid (same memory, green):** MAE climbs steadily to ~$$0.015$$, reflecting pulse broadening.  
   - **Ours (implicit, yellow):** MAE grows to ~$$0.08$$, demonstrating **excessive damping** from implicit Euler.

2. **Profiles at $$t=3\,$$s (center panel):**  
   - The midpoint INSR (blue) overlays the exact Gaussian (grey).  
   - The grid (green) shows slight broadening and amplitude loss.  
   - The implicit INSR (yellow) is noticeably lower in amplitude.

3. **Profiles at $$t=12\,$$s (right panel):**  
   - The blue curve remains sharp and centered, whereas the green grid solution is markedly smeared.  
   - The yellow curve is almost flat, indicating near‑total dissipation.
  
#### Memory–Error–Time Trade‑Off

![1D Transport: Quantitative Table]({{ site.baseurl }}/images/img_insrs_6.png)  
*Table 1: Solution profiles of the 1D Gaussian pulse at $$t=3\,$$s (left) and $$t=12\,$$s (right), showing how INSR + midpoint (blue) preserves the amplitude and shape.*

- To achieve the **same** final MAE of ~$$0.003$$, the grid must increase memory by **8×**.  
- INSR’s wall‑clock cost (hours) vs. grid (seconds) underscores the **compute vs. memory** trade‑off.

### 1.2 2D Two‑Vortex Transport

We test on a classic 2D advection benchmark: two Taylor–Green–style vortices of different spatial scales, advected by an incompressible velocity field. This scenario stresses a solver’s ability to capture **multiscale** features without excessive smoothing.

#### Problem Setup

1. **Domain:**  
   $$[0,2\pi]\times[0,2\pi]$$

2. **Initial Field:**  
   Two oppositely rotating Taylor–Green vortices of different scales:  
   $$
   \rho = 1,\quad 
   u_x = \sin x\,\cos y,\quad 
   u_y = -\cos x\,\sin y.
   $$

3. **Time Integration:**  
   Same midpoint rule, $$\Delta t=0.05$$, $$100$$ steps to $$t=5\,$$ s.

4. **Incompressibility Constraint:**  
   Here we only advect a **passive scalar** (the density field), so $$\nabla\!\cdot u=0$$ is satisfied analytically.

#### Representations & Memory

- **INSR:** SIREN with $$\alpha=3$$, $$\beta=32$$ → 25.887 KB.

- **Grid:** $$48\times48$$ nodes → 27.00 KB.

### Quantitative & Qualitative Results

#### Error over Time & Density Snapshots

![2D Transport: MSE & Density Snapshots]({{ site.baseurl }}/images/img_insr_7.png)  
*Figure 5: Mean squared error over 100 timesteps for the 2D Taylor–Green–style two‑vortex advection, comparing INSR (blue) to the coarse grid solver (green).*

1. **MSE Curves (left):**  
   - **Ours (blue):** Mean squared error stays below $$5\times10^{-4}$$.  
   - **Grid (green):** Error rises above $$4\times10^{-3}$$, almost an order larger.

2. **Density at Step 100 (right):**  
   - INSR preserves **both** the large and small vortex structures with crisp edges.  
   - The grid diffusion washes out the **smaller** vortex entirely.

#### Memory–Error–Time Trade‑Off

![2D Transport: Quantitative Table]({{ site.baseurl }}/images/img_insrs_8.png)  
*Table 2: Density magnitude snapshots of the two‑vortex field at step 100.*

- Matching INSR’s accuracy demands an **enormous** memory spike ($$\times500$$).  
- INSR excels at capturing **multiscale** features that simple grids cannot resolve under tight budgets.

---

# 2. Incompressible Euler Equations

The incompressible Euler equations govern the motion of an ideal (zero-viscosity), divergence‑free fluid. They take the form:

$$
\rho_f\!\biggl(\frac{\partial \mathbf{u}}{\partial t} + \mathbf{u}\cdot\nabla\mathbf{u}\biggr)
\;=\;-\,\nabla p \;+\;\rho_f\,\mathbf{g},
\quad
\nabla\cdot\mathbf{u}=0,
$$

where:

- $$\mathbf{u}(x,t)$$ is the velocity field,  
- $$p(x,t)$$ is the pressure that enforces incompressibility,  
- $$\rho_f$$ is the fluid density (here $$\rho_f=1$$),  
- $$\mathbf{g}$$ is external body force (here $$\mathbf{g}=0$$).  

Even without viscosity, the nonlinear advection term $$\mathbf{u}\!\cdot\nabla\mathbf{u}$$ together with the divergence‑free constraint makes this a challenging PDE to solve accurately, especially when capturing fine vortex structures under tight memory constraints.

### Operator‑Splitting Time Integration

We adopt the classic Chorin‑style operator‑splitting scheme, which breaks the nonlinear, coupled system into three linear substeps per timestep:

![Operator Splitting Workflow]({{ site.baseurl }}/images/img_ins_9.png)

*Figure 6: Chorin-Style Operator-Splitting Workflow*

1. **Advection (semi‑Lagrangian):**
   
   $$
   I_{\mathrm{adv}} = \bigl\|\mathbf{u}_{\mathrm{adv}}^{n+1}(x) - \mathbf{u}^n\bigl(x - \Delta t\,\mathbf{u}^n(x)\bigr)\bigr\|_2^2.
   $$
   
   We backtrack each point $$x$$ by $$\Delta t\,\mathbf{u}^n(x)$$, evaluating the MLP directly at the footpoint (no interpolation).

2. **Pressure Projection:**
   
   $$
   I_{\mathrm{pro}} = \bigl\|\nabla^2 p^{\,n+1}(x) - \nabla\!\cdot\mathbf{u}_{\mathrm{adv}}^{n+1}(x)\bigr\|_2^2.
   $$
   
   Optimizing this enforces $$\nabla\!\cdot \mathbf{u}=0$$ by solving for the pressure MLP.

3. **Velocity Correction:**
   
   $$
   I_{\mathrm{cor}} = \bigl\|\mathbf{u}^{n+1}(x) - \bigl(\mathbf{u}_{\mathrm{adv}}^{n+1}(x) - \nabla p^{\,n+1}(x)\bigr)\bigr\|_2^2.
   $$
   
   The final velocity is obtained by subtracting the learned pressure gradient.

Each substep minimizes its residual over a random batch of points $$\mathcal{M}\subset\Omega$$ using Adam.

### Taylor–Green Vortex Benchmark

The Taylor–Green vortex is a classic analytical solution to the incompressible Euler equations in two dimensions. It’s widely used as a benchmark because:

1. **Closed-form solution:**
   You know the exact velocity field at any time, so you can measure numerical error directly.

2. **No external forcing or viscosity:**
   With ρ=1 and g=0, the equations reduce to pure advection plus pressure projection, making it a clean test of your solver’s ability to handle non-linear advection and maintain incompressibility.

3. **Multiscale challenge:**
   Although the initial condition is smooth, the interacting vortices place stress on your spatial discretization: if your mesh (or neural representation) is too coarse or too diffusive, the smaller-scale features will wash out or degrade over time.

You know the exact velocity field at any time, so you can measure numerical error directly.
We validate on the 2D Taylor–Green vortex (zero viscosity) with analytical solution  

$$
\mathbf{u}(x,y,t) = \bigl(\sin x\cos y,\,-\cos x\sin y\bigr), 
\quad (x,y)\in[0,2\pi]^2.
$$  

Timestep: $$\Delta t=0.05$$, 100 steps to $$t=5\,$$s.  

Both INSR and the coarse grid use ~25 KB for the velocity field:
- **INSR:** SIREN with $$\alpha=3$$ hidden layers, $$\beta=32$$ neurons each.
- **Grid:** $$48{\times}48$$ finite‑difference grid with the same operator‑splitting.

### Quantitative & Qualitative Results

#### Error Growth & Velocity Fields

![Euler Error & Fields]({{ site.baseurl }}/images/img_insr_10.png)  
*Figure 7: (Left) Mean squared error over 100 timesteps. (Right) Velocity magnitude at final step: ground truth, INSR, and grid.*

- **INSR (blue):** MSE remains below $$5\times10^{-4}$$.  
- **Grid (green):** Error climbs above $$4\times10^{-3}$$, nearly an order of magnitude larger.  
- **Field snapshots:** INSR preserves the checkerboard’s fine peaks and troughs; the grid smooths them significantly.

#### Memory–Error–Time Trade‑Off

![Euler Quant Table]({{ site.baseurl }}/images/img_insrs_11.png)  
*Table 3: Quantitative comparison.*

- Achieving INSR’s final error on the grid would require **12 MB** vs. **27 KB**, a **450×** memory increase.  
- INSR excels at capturing complex, divergence‑free flows under tight memory constraints.

---

# 3. Elastodynamic Equation

Elastodynamics is the study of how elastic (i.e. deformable but recoverable) solids respond to time-varying loads. Mathematically, it describes the motion of a continuum body whose internal stresses derive from an elastic energy density.

We now turn to the motion of deformable solids, governed by the **elastodynamic equation**:

> **PDE:**  
> $$
> \rho_0\,\ddot\phi(x,t) \;=\;\nabla\!\cdot P\bigl(F(x,t)\bigr)\;+\;\rho_0\,b(x,t),
> $$
> where  
> - $$\phi(x,t)$$ is the deformation map,  
> - $$\rho_0$$ is the reference density,  
> - $$F = \nabla\phi$$ is the deformation gradient,  
> - $$P = \frac{\partial\Psi}{\partial F}$$ is the first Piola–Kirchhoff stress,  
> - $$b$$ is any applied body force.

To close the system, we assume a **stable Neo-Hookean** energy density:

$$
\Psi(F) \;=\;
\frac{\lambda}{2}\,\mathrm{tr}^2\bigl(\Sigma - I\bigr)\;+\;\mu\,\bigl(\det F - 1\bigr)^2,
$$

where $$\Sigma$$ are the singular values of $$F$$, and $$(\lambda,\mu)$$ are the Lamé parameters.

### Variational Time Integration

We adopt a **variational integrator** that marches the deformation and velocity forward by minimizing the incremental action:

$$
I(\phi^{n+1}) \;=\;
\underbrace{\tfrac12\,\rho_0\bigl\|\dot\phi^{n+1} - \dot\phi^n\bigr\|^2}_{\text{kinetic energy}}
\;+\;
\underbrace{\Psi\bigl(\phi^{n+1}\bigr)}_{\text{elastic energy}}
\;-\;
\underbrace{\rho_0\,b^T\,\phi^{n+1}}_{\substack{\text{external}\\[-3pt]\text{force potential}}},
$$

with
$$
\dot\phi^{n+1} \;=\;\frac{\phi^{n+1}-\phi^n}{\Delta t}\,
$$

By replacing the spatial mesh with an implicit neural representation $$\,\phi_\theta(x)\,$$, we solve the above minimization for the network weights $$\theta$$ at each timestep.

### Implementation Workflow

![Elastodynamic INSR Workflow]({{ site.baseurl }}/images/img_insrs_12.png)   
*Figure 8: INSR elastodynamic pipeline.*

We define the domain, set initial/boundary conditions, sample the undeformed volume, impose collision constraints if needed, then compute the deformation via variational optimization.

- **Define spatial domain:**  
  Establish the geometric region of the object in its undeformed (reference) configuration.  
- **Set initial & boundary conditions:**  
  Specify the starting deformation (often identity) and any fixed or driven boundaries.  
- **Sample undeformed domain:**  
  Randomly pick a mini-batch of points inside the domain to evaluate the neural field.  
- **Collision constraints:**  
  Identify points that collide with external geometry and add penalty terms to enforce contact.  
- **Deformation computation:**  
  Optimize the network weights so that the computed deformation at each sample minimizes the variational energy (kinetic + elastic − external potentials), yielding the next time-step deformation.

### Elastic Tension Test

We first evaluate on a classic **2D tensile test**:

1. **Setup:**  
   - A square block with checkerboard texture (or mesh/points)  
   - Clamped on the left and right boundaries, tension applied slowly.
   
2. **Representations (same memory ~56 KB):**  
   - **Ours (INSR):** SIREN MLP, $$\alpha=3$$, $$\beta=68$$.  
   - **FEM (mesh-based):** Tetrahedral mesh with 0.8 K vertices.  
   - **MPM (particle-based):** 1.7 K material points.  

![Elastic Tension Comparison]({{ site.baseurl }}/images/img_insr_13.png)  
*Figure 9: Undeformed (top row) vs. deformed (bottom) states.*

INSR (left) preserves smooth texture and avoids mesh fracture, FEM (center) shows coarseness, and MPM (right) exhibits particle clustering/fracture.

### Error Visualization

To quantify accuracy, we compare against a high-resolution FEM reference and plot the pointwise $$L_2$$ displacement error:

![Elastic Tension Error Field]({{ site.baseurl }}/images/img_insr_14.png)   
*Figure 10: Per-point $$L_2$$ error heatmap.*

INSR’s error (middle) is visibly lower and more uniform than FEM’s (right), especially near high-strain regions.

### Quantitative Results
![Quantitative Result Comaprison]({{ site.baseurl }}/images/img_insrs_15.png) 
*Table 4: Elastic tension test metrics.*

INSR achieves over 2× lower maximum displacement error under the same memory budget, at the cost of longer runtime.

---

# Conclusion

In this work, we have introduced Implicit Neural Spatial Representations (INSRs) as a versatile, mesh‐free alternative for solving time-dependent PDEs. By encoding spatial fields directly in the weights of a neural network and evolving those weights with classical time integrators we demonstrated:

- **Higher accuracy under tight memory budgets.**  
  Across advection, incompressible Euler, and elastodynamics benchmarks, INSRs consistently outperformed grid- and mesh-based methods when constrained to the same storage footprint.

- **Intrinsic adaptivity.**  
  The global support and smoothness of neural fields automatically concentrate representational capacity on the most challenging regions, without any remeshing or adaptive data structures.

- **Flexibility with time integrators.**  
  By coupling INSRs with midpoint, implicit Euler, operator-splitting, and variational integrators, we can leverage decades of numerical analysis to tackle stiff, nonlinear, or contact-driven dynamics.

- **A clear compute–memory trade-off.**  
  While INSRs incur higher wall-clock runtimes (hours vs. seconds), they open a new point on the Pareto frontier of accuracy vs. memory, which may be preferable for simulations where storage or adaptivity are the primary bottleneck.

Looking forward, promising directions include:

- **Hybrid mesh–neural schemes** to combine the speed of local bases with the expressivity of global neural fields.  
- **Hard enforcement of complex boundary conditions** via constraint-preserving architectures.  
- **Theoretical analysis** of convergence, stability, and error bounds for INSR-based time-stepping.  
- **Real-world applications** in fluid–structure interaction, soft robotics, and geophysical flows where adaptivity and reduced memory are critical.

By uniting the strengths of neural representations with the rich heritage of classical time integrators, INSRs chart a new path for high-fidelity, memory-efficient simulation of a broad class of time-dependent phenomena.  

---

# References

- Chen, Z. and Zhang, H. Learning implicit fields for generative shape modeling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 5939–5948, 2019.
- Pfaff, T., Fortunato, M., Sanchez-Gonzalez, A., and Battaglia, P. W. Learning mesh-based simulation with graph networks. arXiv preprint arXiv:2010.03409, 2020.
- Yang, G., Belongie, S., Hariharan, B., and Koltun, V. Geometry processing with neural fields. Advances in Neural Information Processing Systems, 34, 2021.
- Wandel, N., Weinmann, M., and Klein, R. Learning incompressible fluid dynamics from scratch–towards fast, differentiable fluid models that generalize. arXiv preprint arXiv:2006.08762, 2020. 
- Sharp, N. and Jacobson, A. Spelunking the deep: guaranteed queries on general neural implicit surfaces via range analysis. ACM Transactions on Graphics (TOG), 41(4): 1–16, 2022.
- Park, K., Sinha, U., Hedman, P., Barron, J. T., Bouaziz, S., Goldman, D. B., Martin-Brualla, R., and Seitz, S. M. Hypernerf: a higher-dimensional representation for topologically varying neural radiance fields. ACM Transactions on Graphics (TOG), 40(6):1–12, 2021b. 
- Raissi, M., Perdikaris, P., and Karniadakis, G. E. Physicsinformed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378:686–707, 2019. 
- Schwarz, K., Liao, Y., Niemeyer, M., and Geiger, A. Graf: Generative radiance fields for 3d-aware image synthesis. Advances in Neural Information Processing Systems, 33: 20154–20166, 2020. 
