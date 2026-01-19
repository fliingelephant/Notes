#import "@preview/codelst:2.0.2": sourcecode
#import "@preview/cetz:0.3.4": canvas, draw

#set document(title: "JAXMg Tutorial: Distributed Linear Algebra on Multi-GPU", author: "Tutorial")
#set page(paper: "a4", margin: (x: 2cm, y: 2cm))
#set text(font: "New Computer Modern", size: 11pt)
#set heading(numbering: "1.1")
#set par(justify: true)

#show raw.where(block: true): set text(size: 9pt)
#show raw.where(block: false): set text(size: 10pt)
#show link: underline
#show figure: set block(breakable: false)

// Title
#align(center)[
  #text(size: 24pt, weight: "bold")[JAXMg Basics]
  #v(0.5em)
  #text(size: 11pt, style: "italic")[Introduction to sharding and multi-GPU linear solvers]
]

#v(1em)

= Introduction

#link("https://flatironinstitute.github.io/jaxmg/")[*JAXMg*] wraps NVIDIA's *cuSolverMg* (multi-GPU solver library) for JAX, enabling distributed linear algebra operations across multiple GPUs. This tutorial covers the essential concepts for using JAXMg effectively.

#figure(
  table(
    columns: (auto, 1fr, auto),
    align: (center, left, center),
    stroke: 0.5pt,
    inset: 10pt,
    fill: (x, y) => if y == 0 { rgb("#E3F2FD") },
    [*Function*], [*Solves*], [*Complexity*],
    [`potrs`], [$ A x = b $ via Cholesky (A symmetric positive definite)], [$ O(N^3 slash 3) $],
    [`potri`], [$ A^(-1) $ via Cholesky], [$ O(N^3) $],
    [`syevd`], [Eigendecomposition of symmetric matrix], [$ O(N^3) $],
  ),
  caption: [JAXMg distributed linear algebra functions],
)

= Core Concepts: JAX Sharding

== The Mesh

A *mesh* is a logical $n$-dimensional grid of devices with named axes. Think of it as organizing your GPUs into a coordinate system that JAX uses to distribute data and computation.

#sourcecode(numbers-start: 1)[```python
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

# 1D Mesh: 4 GPUs in a line, axis named "x"
devices = jax.devices("gpu")  # [gpu:0, gpu:1, gpu:2, gpu:3]
mesh_1d = jax.make_mesh((4,), ("x",))

# 2D Mesh: 4 GPUs as 2×2 grid, axes "S" and "T"
import numpy as np
mesh_2d = Mesh(np.array(devices).reshape(2, 2), ("S", "T"))
```]

#figure(
  canvas(length: 1cm, {
    import draw: *
    
    // ===== 1D Mesh =====
    let gpu_colors = (rgb("#81C784"), rgb("#4CAF50"), rgb("#388E3C"), rgb("#2E7D32"))
    
    content((3, 5.2), text(size: 12pt, weight: "bold")[1D Mesh: `mesh_1d = jax.make_mesh((4,), ("x",))`], anchor: "center")
    
    for i in range(4) {
      let x = i * 1.8 + 0.3
      // GPU box with gradient-like effect
      rect((x, 3.5), (x + 1.4, 4.5), fill: gpu_colors.at(i), stroke: 1.5pt + black, radius: 4pt)
      content((x + 0.7, 4.0), text(size: 11pt, weight: "bold", fill: white)[GPU #str(i)])
      // Index label below
      content((x + 0.7, 3.2), text(size: 9pt)[index #str(i)], anchor: "center")
    }
    
    // Axis arrow
    line((0.3, 2.7), (7.5, 2.7), stroke: 2pt + rgb("#1565C0"), mark: (end: "stealth", fill: rgb("#1565C0"), scale: 0.8))
    content((7.8, 2.7), text(size: 11pt, fill: rgb("#1565C0"), weight: "bold")[axis "x"], anchor: "west")
    
    // ===== 2D Mesh =====
    content((12, 5.2), text(size: 12pt, weight: "bold")[2D Mesh: `Mesh(devices.reshape(2,2), ("S","T"))`], anchor: "center")
    
    let gpu_colors_2d = (
      (rgb("#64B5F6"), rgb("#42A5F5")),  // Row 0
      (rgb("#1E88E5"), rgb("#1565C0")),  // Row 1
    )
    
    for i in range(2) {
      for j in range(2) {
        let idx = i * 2 + j
        let x = 9.5 + j * 2.2
        let y = 4.5 - i * 1.5
        rect((x, y - 0.6), (x + 1.8, y + 0.6), fill: gpu_colors_2d.at(i).at(j), stroke: 1.5pt + black, radius: 4pt)
        content((x + 0.9, y), text(size: 11pt, weight: "bold", fill: white)[GPU #str(idx)])
      }
    }
    
    // S axis (vertical)
    line((9.0, 4.5), (9.0, 2.4), stroke: 2pt + rgb("#C62828"), mark: (end: "stealth", fill: rgb("#C62828"), scale: 0.8))
    content((8.6, 3.5), text(size: 11pt, fill: rgb("#C62828"), weight: "bold")[S], anchor: "east")
    
    // T axis (horizontal)
    line((9.5, 5.4), (13.3, 5.4), stroke: 2pt + rgb("#2E7D32"), mark: (end: "stealth", fill: rgb("#2E7D32"), scale: 0.8))
    content((11.4, 5.8), text(size: 11pt, fill: rgb("#2E7D32"), weight: "bold")[axis "T"], anchor: "center")
    
    // Index labels
    content((9.0, 4.5), text(size: 8pt)[(0,0)], anchor: "east", padding: 2pt)
    content((9.0, 3.0), text(size: 8pt)[(1,0)], anchor: "east", padding: 2pt)
  }),
  caption: [Device mesh configurations. Left: 1D mesh with 4 GPUs along axis "x". Right: 2D mesh with 4 GPUs arranged in a 2×2 grid with axes "S" (rows) and "T" (columns).],
)

== PartitionSpec: How to Shard

`PartitionSpec` (aliased as `P`) tells JAX which mesh axis to shard each array dimension along. `None` means that dimension is replicated (not sharded).

#figure(
  table(
    columns: (auto, 1fr),
    stroke: 0.5pt,
    inset: 10pt,
    fill: (x, y) => if y == 0 { rgb("#FFF3E0") },
    [*PartitionSpec*], [*Meaning (for 2D array of shape $(N, M)$)*],
    [`P("x", None)`], [Shard dim 0 (rows) across "x", replicate dim 1 (columns)],
    [`P(None, "x")`], [Replicate dim 0 (rows), shard dim 1 (columns) across "x"],
    [`P("S", "T")`], [Shard rows across "S", columns across "T"],
    [`P(None, None)`], [Fully replicate — each device has a complete copy],
    [`P(("S", "T"), None)`], [Flatten S×T mesh into 1D, shard rows across all devices],
  ),
  caption: [PartitionSpec examples for a 2D array],
)

== Placing Data on Devices

#sourcecode[```python
import jax.numpy as jnp

N = 1000
A = jnp.eye(N, dtype=jnp.float64)   # Matrix: 1000 × 1000
b = jnp.ones((N, 1), dtype=jnp.float64)  # Vector: 1000 × 1

# Create mesh and shard data
mesh = jax.make_mesh((4,), ("x",))
A_sharded = jax.device_put(A, NamedSharding(mesh, P("x", None)))  # Row-sharded
b_replicated = jax.device_put(b, NamedSharding(mesh, P(None, None)))  # Replicated
```]

#figure(
  canvas(length: 1cm, {
    import draw: *
    
    // ===== Matrix A: Row-sharded =====
    content((2.5, 6.5), text(size: 12pt, weight: "bold")[Matrix A (1000×1000)], anchor: "center")
    content((2.5, 6.0), text(size: 10pt)[Sharding: `P("x", None)`], anchor: "center")
    
    let colors = (rgb("#BBDEFB"), rgb("#90CAF9"), rgb("#64B5F6"), rgb("#42A5F5"))
    let row_height = 0.9
    
    for i in range(4) {
      let y = 5.0 - i * row_height
      // Matrix shard
      rect((0, y - row_height + 0.1), (5, y), fill: colors.at(i), stroke: 1pt)
      
      // GPU label inside
      content((0.5, y - row_height/2 + 0.05), text(size: 9pt, weight: "bold")[GPU #str(i)], anchor: "west")
      
      // Row range
      let start_row = i * 250
      let end_row = (i + 1) * 250 - 1
      content((5.3, y - row_height/2 + 0.05), text(size: 9pt)[rows #str(start_row)–#str(end_row)], anchor: "west")
    }
    
    // Dimension labels
    line((0, 1.2), (5, 1.2), stroke: 1pt + rgb("#666"))
    content((2.5, 0.9), text(size: 9pt)[1000 columns (replicated on each GPU)], anchor: "center")
    
    // Vertical brace approximation
    line((-0.3, 5.0), (-0.3, 1.4), stroke: 1pt + rgb("#666"))
    content((-0.5, 3.2), text(size: 9pt)[1000\ rows\ total], anchor: "east")
    
    // ===== Vector b: Replicated =====
    content((11, 6.5), text(size: 12pt, weight: "bold")[Vector b (1000×1)], anchor: "center")
    content((11, 6.0), text(size: 10pt)[Sharding: `P(None, None)`], anchor: "center")
    
    // Show 4 identical copies
    for i in range(4) {
      let x = 9 + i * 1.3
      rect((x, 2.0), (x + 0.8, 5.0), fill: rgb("#C8E6C9"), stroke: 1pt)
      content((x + 0.4, 3.5), text(size: 8pt)[full\ copy], anchor: "center")
      content((x + 0.4, 1.7), text(size: 9pt, weight: "bold")[GPU #str(i)], anchor: "center")
    }
    
    // Explanation
    content((11, 1.2), text(size: 9pt)[Each GPU has identical complete copy], anchor: "center")
  }),
  caption: [Data distribution across 4 GPUs. Left: Matrix A is row-sharded — each GPU owns 250 rows but all 1000 columns. Right: Vector b is replicated — each GPU has a complete copy.],
)

= Using `potrs`: Distributed Cholesky Solve

== Basic Usage

#sourcecode[```python
from jaxmg import potrs

# Problem: Solve Ax = b where A is symmetric positive definite
N = 1000
A = jnp.eye(N, dtype=jnp.float64) * 2  # Diagonal matrix
b = jnp.ones((N, 1), dtype=jnp.float64)

# Setup mesh and sharding
mesh = jax.make_mesh((4,), ("x",))
A = jax.device_put(A, NamedSharding(mesh, P("x", None)))
b = jax.device_put(b, NamedSharding(mesh, P(None, None)))

# Solve! 
x = potrs(A, b, T_A=128, mesh=mesh, in_specs=(P("x", None), P(None, None)))
# x ≈ [0.5, 0.5, 0.5, ...]
```]

== Key Parameters

#figure(
  table(
    columns: (auto, 1fr),
    stroke: 0.5pt,
    inset: 10pt,
    fill: (x, y) => if y == 0 { rgb("#E8F5E9") },
    [*Parameter*], [*Description*],
    [`T_A`], [*Tile size* — block width for distributed algorithm. Larger = faster but more memory. Recommended: 128–2048. Auto-pads if shard size not divisible.],
    [`mesh`], [JAX Mesh object defining device topology],
    [`in_specs`], [Tuple of PartitionSpecs: `(P("axis", None), P(None, None))` — matrix row-sharded, RHS replicated],
    [`return_status`], [If `True`, returns `(solution, status_code)` tuple],
    [`pad`], [If `True` (default), auto-pad shards to be divisible by `T_A`],
  ),
  caption: [`potrs` function parameters],
)

== Internal Workflow

#figure(
  canvas(length: 1cm, {
    import draw: *
    
    let box_w = 4.5
    let box_h = 1.0
    let gap = 0.4
    let start_y = 6.5
    
    let steps = (
      ("1. Input", "Each GPU holds local shard\nof A: shape (N/ndev, N)", rgb("#E3F2FD")),
      ("2. Padding", "If shard_size % T_A ≠ 0,\nadd padding rows", rgb("#FFF3E0")),
      ("3. Cholesky", "cuSolverMg computes\nA = L·L^T (distributed)", rgb("#E8F5E9")),
      ("4. Forward", "Solve L·y = b\n(triangular solve)", rgb("#F3E5F5")),
      ("5. Backward", "Solve L^T·x = y\n(triangular solve)", rgb("#F3E5F5")),
      ("6. Output", "Solution x replicated\nto all GPUs", rgb("#FFEBEE")),
    )
    
    for (i, (title, desc, color)) in steps.enumerate() {
      let y = start_y - i * (box_h + gap)
      
      // Box
      rect((0, y - box_h/2), (box_w, y + box_h/2), fill: color, stroke: 1.5pt, radius: 5pt)
      
      // Title
      content((0.2, y + 0.2), text(size: 11pt, weight: "bold")[#title], anchor: "west")
      
      // Description on right
      content((box_w + 0.5, y), text(size: 9pt)[#desc], anchor: "west")
      
      // Arrow to next
      if i < steps.len() - 1 {
        line((box_w/2, y - box_h/2 - 0.05), (box_w/2, y - box_h/2 - gap + 0.05), 
             stroke: 1.5pt, mark: (end: "stealth", fill: black, scale: 0.6))
      }
    }
    
    // GPU diagram on far right
    content((12, 6.5), text(size: 10pt, weight: "bold")[4 GPUs working together:], anchor: "center")
    
    for i in range(4) {
      let y = 5.5 - i * 0.9
      rect((10.5, y - 0.3), (13.5, y + 0.3), fill: rgb("#90CAF9"), stroke: 1pt, radius: 3pt)
      content((12, y), text(size: 9pt)[GPU #str(i): rows #str(i*250)–#str((i+1)*250-1)], anchor: "center")
    }
    
    // Arrows showing communication
    line((13.7, 5.2), (13.7, 2.6), stroke: 1pt + rgb("#666"), mark: (start: "stealth", end: "stealth", fill: rgb("#666"), scale: 0.4))
    content((14.5, 3.9), text(size: 8pt)[inter-GPU\ comm], anchor: "west")
  }),
  caption: [Distributed Cholesky solve workflow. The matrix is distributed across GPUs; cuSolverMg handles the distributed factorization and triangular solves with automatic inter-GPU communication.],
)

== Tile Size Selection

The tile size `T_A` controls the block granularity of the distributed algorithm:

#sourcecode[```python
N = 10000
ndev = 4
shard_size = N // ndev  # 2500 rows per GPU

# Good choices: T_A divides shard_size evenly (no padding needed)
T_A = 250   # 2500 / 250 = 10 tiles ✓
T_A = 500   # 2500 / 500 = 5 tiles  ✓
T_A = 2048  # 2500 / 2048 = 1.22 → padding added automatically ⚠
```]

#block(
  fill: rgb("#FFF9C4"),
  inset: 12pt,
  radius: 5pt,
  width: 100%,
  stroke: 1pt + rgb("#FBC02D"),
)[
  #text(weight: "bold")[💡 Tips for choosing `T_A`:]
  - Use `T_A ≥ 128` for good performance (small tiles = slow)
  - Larger `T_A` → fewer communication rounds, but more memory per tile
  - Best: choose `T_A` that divides `N // ndev` evenly to avoid padding overhead
  - Common good choices: 256, 512, 1024, 2048
]

#pagebreak()
= The `shard_map` Pattern

== What is `shard_map`?

`jax.shard_map` lets you write code that runs *per-shard* on each device, with explicit collective operations for cross-device communication:

#sourcecode[```python
from functools import partial

@partial(jax.shard_map,
         mesh=mesh,
         in_specs=P("x", None),   # Input: row-sharded
         out_specs=P(None))       # Output: replicated
def my_distributed_fn(local_A):
    # This code runs on EACH GPU independently
    # local_A shape: (N/ndev, M) — just this GPU's shard
    
    my_idx = jax.lax.axis_index("x")  # Which GPU am I? (0, 1, 2, or 3)
    
    # Collective: sum across all GPUs
    global_sum = jax.lax.psum(local_sum, axis_name="x")
    
    # Collective: gather all shards
    full_A = jax.lax.all_gather(local_A, axis_name="x", axis=0, tiled=True)
    
    return result
```]

== Common Collective Operations

#figure(
  table(
    columns: (auto, 1fr, auto),
    stroke: 0.5pt,
    inset: 10pt,
    fill: (x, y) => if y == 0 { rgb("#E1F5FE") },
    [*Operation*], [*Description*], [*Output Shape*],
    [`psum(x, "axis")`], [Sum values across all devices on axis], [Same shape, replicated],
    [`pmean(x, "axis")`], [Mean across devices], [Same shape, replicated],
    [`all_gather(x, "axis")`], [Concatenate shards from all devices], [Larger (combined)],
    [`axis_index("axis")`], [Get this device's index on axis], [Scalar: 0 to ndev-1],
    [`pbroadcast(x, "axis")`], [Broadcast from source device], [Same on all devices],
  ),
  caption: [JAX collective operations for `shard_map`],
)

#figure(
  canvas(length: 1cm, {
    import draw: *
    
    // ===== psum example =====
    content((3, 6), text(size: 11pt, weight: "bold")[`psum`: Sum across devices], anchor: "center")
    
    let vals = ("3", "7", "2", "8")
    for i in range(4) {
      rect((i * 1.5 + 0.2, 4.5), (i * 1.5 + 1.2, 5.3), fill: rgb("#BBDEFB"), stroke: 1pt, radius: 3pt)
      content((i * 1.5 + 0.7, 4.9), text(size: 10pt)[#vals.at(i)])
      content((i * 1.5 + 0.7, 4.2), text(size: 8pt)[GPU#str(i)], anchor: "center")
    }
    
    // Arrow down
    line((3, 4.0), (3, 3.5), stroke: 1.5pt, mark: (end: "stealth", fill: black, scale: 0.5))
    content((3.3, 3.75), text(size: 8pt)[psum], anchor: "west")
    
    // Result: all get 20
    for i in range(4) {
      rect((i * 1.5 + 0.2, 2.5), (i * 1.5 + 1.2, 3.3), fill: rgb("#C8E6C9"), stroke: 1pt, radius: 3pt)
      content((i * 1.5 + 0.7, 2.9), text(size: 10pt, weight: "bold")[20])
    }
    content((3, 2.1), text(size: 9pt)[All devices get sum = 3+7+2+8 = 20], anchor: "center")
    
    // ===== all_gather example =====
    content((11, 6), text(size: 11pt, weight: "bold")[`all_gather`: Collect all shards], anchor: "center")
    
    let shard_labels = ("A₀", "A₁", "A₂", "A₃")
    for i in range(4) {
      rect((9 + i * 1.2, 4.5), (9 + i * 1.2 + 0.9, 5.3), fill: rgb("#FFCCBC"), stroke: 1pt, radius: 3pt)
      content((9 + i * 1.2 + 0.45, 4.9), text(size: 10pt)[#shard_labels.at(i)])
      content((9 + i * 1.2 + 0.45, 4.2), text(size: 8pt)[GPU#str(i)], anchor: "center")
    }
    
    // Arrow down
    line((11, 4.0), (11, 3.5), stroke: 1.5pt, mark: (end: "stealth", fill: black, scale: 0.5))
    content((11.5, 3.75), text(size: 8pt)[all_gather], anchor: "west")
    
    // Result: each gets full array
    for i in range(4) {
      rect((9 + i * 1.2, 2.2), (9 + i * 1.2 + 0.9, 3.4), fill: rgb("#B2DFDB"), stroke: 1pt, radius: 3pt)
      content((9 + i * 1.2 + 0.45, 2.8), text(size: 7pt)[A₀A₁\ A₂A₃])
    }
    content((11, 1.8), text(size: 9pt)[All devices get concatenated full array], anchor: "center")
  }),
  caption: [Collective operations illustrated. Left: `psum` sums values and replicates result. Right: `all_gather` collects all shards into complete array on each device.],
)

== Using `potrs_shardmap_ctx`

For advanced use inside your own `shard_map` (when you need custom logic before/after the solve):

#sourcecode[```python
from jaxmg import potrs_shardmap_ctx

def custom_solve(local_A, b):
    """Runs inside shard_map context — local_A is just this GPU's shard"""
    local_A = local_A * some_scaling_factor  # Custom pre-processing
    x, status = potrs_shardmap_ctx(local_A, b, T_A=256)  # No extra shard_map
    return x, status

result = jax.shard_map(
    partial(custom_solve, b=b_replicated),
    mesh=mesh,
    in_specs=P("x", None),
    out_specs=(P(None, None), P(None)),
    check_vma=False,  # Required for FFI calls
)(A_sharded)
```]

#pagebreak()

= Application: MinSR with JAXMg

== The MinSR Algorithm

In Variational Monte Carlo, *minSR* (minimum-step Stochastic Reconfiguration) computes parameter updates efficiently when $N_s lt.double N_p$ (fewer samples than parameters):

#align(center)[
  #block(
    fill: rgb("#F3E5F5"),
    inset: 15pt,
    radius: 8pt,
    stroke: 1.5pt + rgb("#7B1FA2"),
  )[
    $ delta theta = tau dot X^dagger (X X^dagger + lambda I)^(-1) dot E^"loc" $
  ]
]

#table(
  columns: (auto, 1fr),
  stroke: none,
  inset: 5pt,
  [$X in RR^(N_s times N_p)$], [Centered Jacobian (samples × parameters)],
  [$T = X X^dagger in RR^(N_s times N_s)$], [*Gram matrix* — much smaller than $N_p times N_p$!],
  [$lambda$], [Regularization (diagonal shift, typically $10^(-4)$ to $10^(-2)$)],
  [$E^"loc"$], [Centered local energy vector],
)

== Multi-Node Mesh Setup

For clusters with multiple nodes, use a 2D mesh where one axis is within-node and another is across-nodes:

#sourcecode[```python
def get_device_grid():
    """Organize devices: rows = local GPUs, columns = nodes"""
    by_proc = {}
    for d in jax.devices():
        by_proc.setdefault(d.process_index, []).append(d)
    hosts = sorted(by_proc)
    return np.array([[by_proc[h][x] for h in hosts] 
                     for x in range(jax.local_device_count())]).T

def create_2d_mesh():
    dev_grid = get_device_grid()  # shape: (n_nodes, n_gpus_per_node)
    return Mesh(dev_grid, ("S", "T"))
    # "S" = across nodes, "T" = within node
```]

#figure(
  canvas(length: 1cm, {
    import draw: *
    
    content((7, 7), text(size: 14pt, weight: "bold")[Multi-Node Setup: 2 Nodes × 4 GPUs = 8 Total], anchor: "center")
    
    // ===== Node 0 =====
    rect((-0.3, 1.5), (5.8, 5.5), stroke: (dash: "dashed", thickness: 1.5pt), fill: rgb("#E8F5E9").lighten(50%), radius: 8pt)
    content((2.75, 5.9), text(size: 12pt, weight: "bold")[Node 0 (Process 0)], anchor: "center")
    
    let gpu_colors_0 = (rgb("#81C784"), rgb("#66BB6A"), rgb("#4CAF50"), rgb("#43A047"))
    for i in range(4) {
      let x = i * 1.4 + 0.2
      rect((x, 2.0), (x + 1.1, 5.0), fill: gpu_colors_0.at(i), stroke: 1.5pt, radius: 5pt)
      content((x + 0.55, 4.3), text(size: 11pt, weight: "bold", fill: white)[GPU], anchor: "center")
      content((x + 0.55, 3.5), text(size: 14pt, weight: "bold", fill: white)[#str(i)], anchor: "center")
      content((x + 0.55, 2.5), text(size: 8pt, fill: white)[local], anchor: "center")
    }
    
    // ===== Node 1 =====
    rect((7.2, 1.5), (13.3, 5.5), stroke: (dash: "dashed", thickness: 1.5pt), fill: rgb("#E3F2FD").lighten(50%), radius: 8pt)
    content((10.25, 5.9), text(size: 12pt, weight: "bold")[Node 1 (Process 1)], anchor: "center")
    
    let gpu_colors_1 = (rgb("#64B5F6"), rgb("#42A5F5"), rgb("#2196F3"), rgb("#1E88E5"))
    for i in range(4) {
      let x = 7.7 + i * 1.4
      rect((x, 2.0), (x + 1.1, 5.0), fill: gpu_colors_1.at(i), stroke: 1.5pt, radius: 5pt)
      content((x + 0.55, 4.3), text(size: 11pt, weight: "bold", fill: white)[GPU], anchor: "center")
      content((x + 0.55, 3.5), text(size: 14pt, weight: "bold", fill: white)[#str(i+4)], anchor: "center")
      content((x + 0.55, 2.5), text(size: 8pt, fill: white)[local], anchor: "center")
    }
    
    // Network connection
    line((6.0, 3.5), (7.0, 3.5), stroke: 2pt + rgb("#FF5722"), mark: (start: "stealth", end: "stealth", fill: rgb("#FF5722"), scale: 0.6))
    content((6.5, 4.0), text(size: 9pt, fill: rgb("#FF5722"), weight: "bold")[Network], anchor: "center")
    
    // Axis labels
    line((0.75, 1.0), (12.25, 1.0), stroke: 2pt + rgb("#1565C0"), mark: (end: "stealth", fill: rgb("#1565C0"), scale: 0.7))
    content((6.5, 0.5), text(size: 11pt, fill: rgb("#1565C0"), weight: "bold")[axis "S" (across nodes)], anchor: "center")
    
    line((-0.8, 5.0), (-0.8, 2.0), stroke: 2pt + rgb("#C62828"), mark: (end: "stealth", fill: rgb("#C62828"), scale: 0.7))
    content((-1.5, 3.5), text(size: 11pt, fill: rgb("#C62828"), weight: "bold")[axis\ "T"], anchor: "east")
  }),
  caption: [2D mesh for multi-node cluster. Axis "S" spans across nodes (inter-node communication via network). Axis "T" spans GPUs within each node (fast NVLink/PCIe). Total 8 devices in a (2, 4) grid.],
)

#pagebreak()

== MinSR Data Flow

#figure(
  canvas(length: 1cm, {
    import draw: *
    
    // Title
    content((7, 8), text(size: 14pt, weight: "bold")[MinSR Computation Pipeline], anchor: "center")
    
    let box_w = 4.0
    let box_h = 1.2
    
    // Step 1: Jacobian
    rect((0, 6), (box_w, 6 + box_h), fill: rgb("#E3F2FD"), stroke: 1.5pt, radius: 5pt)
    content((box_w/2, 6.6), text(size: 10pt, weight: "bold")[Jacobian X], anchor: "center")
    content((box_w/2, 6.2), text(size: 9pt)[(N_s × N_p)], anchor: "center")
    content((box_w + 0.3, 6.6), text(size: 8pt)[Sharded: `P(("S","T"), None)`\ Each GPU: (N_s/8, N_p)], anchor: "west")
    
    // Arrow
    line((box_w/2, 6 - 0.1), (box_w/2, 5.2), stroke: 1.5pt, mark: (end: "stealth", fill: black, scale: 0.5))
    content((box_w/2 + 0.3, 5.6), text(size: 8pt)[X·X#super[T]], anchor: "west")
    
    // Step 2: Gram matrix
    rect((0, 4), (box_w, 4 + box_h), fill: rgb("#FFF3E0"), stroke: 1.5pt, radius: 5pt)
    content((box_w/2, 4.6), text(size: 10pt, weight: "bold")[Gram T = XX^T], anchor: "center")
    content((box_w/2, 4.2), text(size: 9pt)[(N_s × N_s)], anchor: "center")
    content((box_w + 0.3, 4.6), text(size: 8pt)[Row-sharded: `P("T", None)`\ Much smaller than N_p × N_p!], anchor: "west")
    
    // Arrow
    line((box_w/2, 4 - 0.1), (box_w/2, 3.2), stroke: 1.5pt, mark: (end: "stealth", fill: black, scale: 0.5))
    content((box_w/2 + 0.3, 3.6), text(size: 8pt)[+ λI], anchor: "west")
    
    // Step 3: Regularized
    rect((0, 2), (box_w, 2 + box_h), fill: rgb("#F3E5F5"), stroke: 1.5pt, radius: 5pt)
    content((box_w/2, 2.6), text(size: 10pt, weight: "bold")[T + λI], anchor: "center")
    content((box_w/2, 2.2), text(size: 9pt)[SPD matrix], anchor: "center")
    
    // Arrow to potrs
    line((box_w + 0.1, 2.6), (6.5, 2.6), stroke: 2pt + rgb("#1565C0"), mark: (end: "stealth", fill: rgb("#1565C0"), scale: 0.6))
    
    // potrs box
    rect((6.5, 2), (10.5, 3.2), fill: rgb("#C8E6C9"), stroke: 2pt + rgb("#2E7D32"), radius: 8pt)
    content((8.5, 2.85), text(size: 11pt, weight: "bold")[JAXMg `potrs`], anchor: "center")
    content((8.5, 2.35), text(size: 9pt)[Distributed Cholesky], anchor: "center")
    
    // E_loc input
    rect((6.5, 4.5), (10.5, 5.5), fill: rgb("#FFCDD2"), stroke: 1.5pt, radius: 5pt)
    content((8.5, 5), text(size: 10pt, weight: "bold")[E^loc vector (N_s)], anchor: "center")
    line((8.5, 4.5 - 0.1), (8.5, 3.3), stroke: 1.5pt, mark: (end: "stealth", fill: black, scale: 0.5))
    
    // Arrow from potrs
    line((10.5 + 0.1, 2.6), (12, 2.6), stroke: 1.5pt, mark: (end: "stealth", fill: black, scale: 0.5))
    
    // v vector
    rect((12, 2), (14, 3.2), fill: rgb("#B3E5FC"), stroke: 1.5pt, radius: 5pt)
    content((13, 2.85), text(size: 10pt, weight: "bold")[v], anchor: "center")
    content((13, 2.35), text(size: 9pt)[(N_s)], anchor: "center")
    
    // Arrow down
    line((13, 2 - 0.1), (13, 1.2), stroke: 1.5pt, mark: (end: "stealth", fill: black, scale: 0.5))
    content((13.3, 1.6), text(size: 8pt)[X#super[T]·v], anchor: "west")
    
    // Final output
    rect((11.5, 0), (14.5, 1), fill: rgb("#DCEDC8"), stroke: 2pt + rgb("#689F38"), radius: 8pt)
    content((13, 0.5), text(size: 11pt, weight: "bold")[δθ (N_p)], anchor: "center")
    
    // psum annotation
    content((10.5, 0.5), text(size: 8pt)[`psum` reduces\ across GPUs], anchor: "east")
  }),
  caption: [MinSR data flow. The Jacobian X is sharded by samples. The Gram matrix T=XX^T is computed distributedly. JAXMg's `potrs` solves (T+λI)v = E^loc. Final update δθ = X^T v uses distributed matrix-vector product with `psum` reduction.],
)

== Key Code Snippets

*Sharding the Jacobian:*
#sourcecode[```python
O_LT = jax.lax.with_sharding_constraint(
    O_LT, NamedSharding(mesh_2d, P(("S", "T"), None))
)  # Samples sharded across all 8 GPUs, params replicated
```]

*Distributed Gram matrix (memory-efficient streaming):*
#sourcecode[```python
def streamed_gram(O_L, chunk_size):
    """Compute T = X @ X.T without materializing full intermediate"""
    def step(acc, chunk):
        full_chunk = jax.lax.all_gather(chunk, ('S', 'T'), tiled=True)
        return acc + chunk @ full_chunk.T, None
    return jax.lax.scan(step, zeros, O_L_chunked)[0]
```]

*Distributed Cholesky solve:*
#sourcecode[```python
v = potrs(T_reg, E_loc[:, None], T_A=2048, mesh=mesh_2d,
          in_specs=(P("T", None), P(None, None)))
```]

*Final update with reduction:*
#sourcecode[```python
def update(O_LT, v):
    return jax.lax.psum(v @ O_LT, axis_name=("S", "T"))
delta_theta = jax.shard_map(update, mesh_2d, 
    in_specs=(P(("S","T"), None), P(("S","T"))), out_specs=P(None))(O_LT, v)
```]

#pagebreak()

= Complete Example

#sourcecode[```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jaxmg import potrs

# Setup
jax.config.update("jax_enable_x64", True)
devices = jax.devices("gpu")
ndev = len(devices)
mesh = jax.make_mesh((ndev,), ("x",))

# Create Gram matrix (like in minSR: T = X @ X.T)
N_samples, N_params = 4000, 100000
key = jax.random.PRNGKey(0)
X = jax.random.normal(key, (N_samples, N_params)) / jnp.sqrt(N_samples)
T = X @ X.T  # (4000, 4000) — much smaller than (100000, 100000)!

# Add regularization
T_reg = T + 1e-4 * jnp.eye(N_samples)

# RHS (centered local energies)
b = jax.random.normal(key, (N_samples, 1))

# Shard for distributed solve
T_sharded = jax.device_put(T_reg, NamedSharding(mesh, P("x", None)))
b_replicated = jax.device_put(b, NamedSharding(mesh, P(None, None)))

# Distributed Cholesky solve: (T + λI) v = b
v = potrs(T_sharded, b_replicated, T_A=256, mesh=mesh,
          in_specs=(P("x", None), P(None, None)))

# Final minSR update: δθ = X.T @ v
delta_theta = X.T @ v  # (100000, 1)
print(f"Update shape: {delta_theta.shape}")
```]

= Quick Reference

#figure(
  table(
    columns: (auto, 1fr),
    stroke: 0.5pt,
    inset: 12pt,
    fill: (x, y) => if y == 0 { rgb("#E3F2FD") } else if calc.odd(y) { rgb("#FAFAFA") },
    [*Aspect*], [*Rule*],
    [Matrix sharding], [Always `P(<axis>, None)` — rows sharded, columns replicated],
    [RHS vector], [Always `P(None, None)` — fully replicated on all devices],
    [Tile size `T_A`], [Use 128–2048; choose to divide `N // ndev` evenly if possible],
    [Multi-node mesh], [2D mesh: `("across_nodes", "within_node")` axes],
    [Memory efficiency], [Use streaming Gram computation for large N_params],
    [Output], [Solution is always replicated to all devices],
  ),
  caption: [JAXMg quick reference],
)

#v(1em)

#block(
  fill: rgb("#E8F5E9"),
  inset: 15pt,
  radius: 8pt,
  width: 100%,
  stroke: 1.5pt + rgb("#4CAF50"),
)[
  #text(size: 12pt, weight: "bold")[✓ Summary]
  
  JAXMg makes NVIDIA's multi-GPU linear algebra "just work" with JAX:
  
  1. *Create a mesh* — organize your GPUs into a logical grid
  2. *Shard your data* — use `PartitionSpec` to distribute arrays
  3. *Call the solver* — `potrs`, `potri`, or `syevd` handle everything
  
  The library manages distributed Cholesky factorization, inter-GPU communication, padding, and memory automatically. You write high-level JAX code; JAXMg handles the CUDA complexity.
]
