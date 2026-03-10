# Dataset, Tasks, and Network Architectures Across Energy-Based Models

Audience: software engineers implementing models step by step.
Base dataset: **Fashion‑MNIST / MNIST (28×28 grayscale images)**.

Each image:

28 × 28 = **784 pixels**

Images are flattened when required:

784‑dimensional vector.

General rule used throughout the document:

- associative memory models → store **one prototype per class**
- generative energy models → learn **full dataset distribution**
- attention / transformers → operate on **embeddings or tokens**

---

# 1. Hopfield Network

## Why these sizes

Hopfield networks directly store the **input pattern itself** as the network state.

Fashion‑MNIST images:

28 × 28 = **784 pixels**

Flattened representation:

**784‑dimensional state vector**.

Therefore the Hopfield network must contain:

**N = 784 neurons**

so each neuron represents **one pixel** of the stored pattern.

The theoretical storage capacity of a classical Hopfield network is approximately:

$$
P_{max} \approx 0.138 N
$$

For N = 784:

$$
P_{max} \approx 108
$$

In our experiment we store only **10 prototypes**, which keeps the system well within the stable regime.

---

## Network architecture

Single recurrent layer with **784 neurons**.

Connections:

- fully connected
- symmetric weights

Constraints:

w_ij = w_ji

w_ii = 0

Weight matrix size:

**784 × 784**

Neuron state:

s ∈ {−1, +1}^{784}

---

## Energy

$$
E(s) = -\frac{1}{2} \sum_{i,j} w_{ij} s_i s_j
$$

---

## Dataset usage

Store **one prototype per class**.

Example:

10 classes → **P = 10 memories**.

Each memory is a **784‑dimensional binary vector**.

---

## Retrieval update

$$
s_i \leftarrow \text{sign}\left(\sum_j w_{ij}s_j\right)
$$

Iterate until convergence.

---

## Task

Pattern completion / denoising:

corrupted image → Hopfield dynamics → stored prototype

---

# 2. Boltzmann Machine

## Why these sizes

Visible units correspond directly to the **data dimensionality**.

Fashion‑MNIST:

**784 visible units**.

Hidden units capture latent dependencies between pixels.

We choose **256 hidden units** as a reasonable demonstration scale.

Total stochastic units:

784 + 256 = **1040 units**.

This keeps sampling computationally manageable while still demonstrating probabilistic energy modeling.

---

## Network architecture

Undirected stochastic graph.

Layers:

Visible layer: **784 units**  
Hidden layer: **256 units**

Connections may include:

- visible ↔ hidden
- hidden ↔ hidden
- visible ↔ visible

Each node is a **binary stochastic variable**.

---

## Energy

$$
E(v,h) = -v^TWh - b^Tv - c^Th
$$

---

## Dataset usage

Use the **full training dataset (~60k images)**.

The model learns:

P(image)

---

## Task

Generative modeling.

Training objective:

$$
\max_\theta \sum \log P(v)
$$

Typical demonstrations:

- generate new samples
- reconstruct corrupted images

---

# 3. Restricted Boltzmann Machine

## Why these sizes

Visible units again match input dimensionality:

**784 visible neurons**.

Hidden units represent latent features.

We use **256 hidden neurons** because it:

- captures meaningful visual structure
- keeps training stable
- keeps parameter count manageable

Weight matrix size:

784 × 256 = **200,704 parameters**.

---

## Network architecture

Bipartite graph.

Visible layer: **784 units**  
Hidden layer: **256 units**

Connections only:

visible ↔ hidden

No visible‑visible or hidden‑hidden connections.

---

## Energy

$$
E(v,h) = -v^TWh - b^Tv - c^Th
$$

---

## Conditional distributions

$$
P(h_j=1|v)=\sigma(W_j v + c_j)
$$

$$
P(v_i=1|h)=\sigma(W_i^T h + b_i)
$$

---

## Dataset usage

Use the **full dataset**.

---

## Task

Feature learning and generative modeling.

Pipeline:

image → hidden representation → reconstructed image

---

# 4. Dense Associative Memory

## Why these sizes

Instead of storing raw pixels we store **compressed embeddings**.

Encoder network:

784 → 256 → **128‑dim embedding**

Reasons for 128 dimensions:

- compact representation
- sufficient expressive capacity
- easy to visualize and store in memory modules

Memory matrix size:

P × d

With:

P = 10 memories  
d = 128

Memory matrix:

**10 × 128**

---

## Network architecture

Pipeline:

image → encoder → embedding

Encoder structure:

Linear(784 → 256) → ReLU  
Linear(256 → 128)

Memory module stores embeddings.

---

## Energy

$$
E = -\sum_{\mu} F\left(\sum_i \xi_i^\mu s_i\right)
$$

Typical choice:

F(x) = exp(x)

---

## Task

Associative retrieval in embedding space.

query embedding → retrieve stored memory.

---

# 5. Modern Hopfield Network

## Why these sizes

Modern Hopfield networks operate on **continuous embeddings**.

We reuse the same representation size used above:

embedding dimension **d = 128**.

Stored memories:

P = 10

Memory matrix:

**X ∈ R^{10 × 128}**

Modern Hopfield capacity grows **exponentially with embedding dimension**, so storing 10 memories is trivial.

---

## Network architecture

Input:

query vector (128‑dim).

Memory matrix:

10 stored vectors.

Retrieval step computes similarity between query and memories.

---

## Update rule

$$
x' = \text{softmax}(\beta X^T x) X
$$

---

## Dataset usage

image → encoder → 128‑dim embedding

---

## Task

Vector retrieval using attention‑like weighting.

---

# 6. Transformer Attention

## Why these sizes

Transformers operate on **token sequences**, so images must be tokenized.

We split a 28 × 28 image into **4 × 4 patches**.

Patch grid:

28 / 4 = 7

7 × 7 = **49 patches**.

Each patch contains:

4 × 4 = **16 pixels**.

Patch embedding projects 16 values into a higher dimension.

Chosen embedding size:

**64**

Reasons:

- divisible by number of attention heads
- small enough for demonstration

Attention heads:

4 heads

Per‑head dimension:

64 / 4 = **16**.

Sequence length:

**49 tokens**.

---

## Network architecture

Pipeline:

image → patches → patch embeddings → transformer layer

Patch embedding layer:

16 → 64

Transformer parameters:

embedding dimension: 64  
heads: 4

---

## Attention

$$
\text{Attention}(Q,K,V) = \text{softmax}(QK^T)V
$$

---

## Task

Context aggregation between image patches.

---

# 7. Large Language Models

## Why these sizes

LLMs operate on token sequences.

For demonstration we define a **small transformer model**.

Vocabulary size:

30k tokens

Embedding dimension:

256

Attention heads:

8

Per‑head dimension:

256 / 8 = 32

Feed‑forward dimension:

1024 (≈4× embedding size).

Transformer depth:

4 layers.

This configuration is small but still demonstrates the architecture of modern LLMs.

---

## Network architecture

Pipeline:

text → tokenizer → embeddings → transformer blocks → output logits

Transformer block contains:

- multi‑head attention
- feedforward network
- residual connections
- layer normalization

---

## Objective

$$
P(x_t | x_1, x_2, ..., x_{t-1})
$$

---

# 8. JEPA

## Why these sizes

JEPA operates entirely in **latent representation space**.

We use a CNN encoder producing **128‑dimensional embeddings**.

Reasons:

- enough capacity for image semantics
- small enough for fast experimentation
- consistent with earlier embedding sizes

---

## Network architecture

Two encoders with identical architecture:

Context encoder  
Target encoder

CNN encoder structure:

Conv(32 filters) → ReLU  
Conv(64 filters) → ReLU  
Flatten  
Linear → **128‑dim representation**

Predictor network:

MLP 128 → 128

---

## Objective

$$
E(x,y)
$$

Minimize distance between predicted representation and target representation.

---

## Task

Predict representation of masked image regions instead of predicting pixels.

---

# End Result

The pipeline illustrates the conceptual evolution:

Hopfield memory → probabilistic energy models → latent feature learning → continuous associative memory → attention → transformers → representation prediction.

