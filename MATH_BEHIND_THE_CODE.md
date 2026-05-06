# Math Behind the Code

In this file we analyze the model architecture by reducing the code to the mathematical operations it implements, while also studying important scaling factors, tensor shapes, time and memory complexity, and the reasoning behind these design choices. We will go in chronological order.

## Token Embeddings

We start with a batch of token sequences. If the original input is text, then it is first tokenized into discrete token ids. The model itself does not directly receive raw text; it receives these token ids.

Let $B$ be the batch size, $T$ be the fixed context length, and $V$ be the vocabulary size. Then the input token ids form the tensor

$$
X_{\text{token ids}} \in \{0, 1, \dots, V - 1\}^{B \times T}.
$$

Here $X_{\text{token ids}}^{b,t}$ is the token id at batch index $b$ and sequence position $t$. Each entry is an integer index into the vocabulary.

The token embedding table is a learned matrix

$$
W_{\text{token embedding table}} \in \mathbb{R}^{V \times d_{\text{hidden}}}.
$$

Each row of this matrix is the vector representation of one token in the vocabulary. The embedding operation maps each discrete token id into a vector in $\mathbb{R}^{d_{\text{hidden}}}$:

$$
X_{\text{token ids}} \in \{0, 1, \dots, V - 1\}^{B \times T}
\longrightarrow
X_{\text{hidden}}^{(0)} \in \mathbb{R}^{B \times T \times d_{\text{hidden}}}.
$$

More explicitly, for every batch index $b$, token position $t$, and embedding coordinate $c$,

$$
X_{\text{hidden}}^{(0),b,t,c} = W_{\text{token embedding table}}^{X_{\text{token ids}}^{b,t},\ c}.
$$

So the token id $X_{\text{token ids}}^{b,t}$ selects a row of the embedding table, and that selected row becomes the initial hidden vector at position $(b,t)$.

Complexity:

- Time: $O(BT d_{\text{hidden}})$
- Activation memory: $O(BT d_{\text{hidden}})$
- Parameter memory: $O(V d_{\text{hidden}})$

Mathematically, an embedding lookup can be viewed as multiplying a one-hot token vector by the embedding matrix. Computationally, it is implemented as a direct row lookup.

## Transformer Blocks as Iterative Maps

After token embedding, the model has the initial hidden state

$$
X_{\text{hidden}}^{(0)} \in \mathbb{R}^{B \times T \times d_{\text{hidden}}}.
$$

The transformer applies a stack of layers that repeatedly update this hidden state. If $X_{\text{hidden}}^{(i)}$ is the hidden state entering layer $i$, then the layer computes

$$
X_{\text{hidden}}^{(i+1)} = F_i(X_{\text{hidden}}^{(i)}).
$$

In this model, each block uses pre-normalization:

$$
Y^{(i)} = X_{\text{hidden}}^{(i)} + \mathrm{Attention}(\mathrm{RMSNorm}(X_{\text{hidden}}^{(i)})).
$$

$$
X_{\text{hidden}}^{(i+1)} = Y^{(i)} + \mathrm{MoE}(\mathrm{RMSNorm}(Y^{(i)})).
$$

So the layer function $F_i$ is built from RMSNorm, multi-head latent attention, residual addition, and a mixture-of-experts feedforward transformation. We analyze these pieces one at a time.

## RMSNorm

RMSNorm normalizes each hidden vector by its root mean square magnitude. For a single hidden vector

$$
x \in \mathbb{R}^{d_{\text{hidden}}},
$$

the root mean square is

$$
\mathrm{RMS}(x) = \sqrt{\frac{1}{d_{\text{hidden}}}\sum_{c=1}^{d_{\text{hidden}}} x_c^2 + \epsilon}.
$$

Then RMSNorm computes

$$
\mathrm{RMSNorm}(x)_c = g_c \frac{x_c}{\mathrm{RMS}(x)},
$$

where

$$
g \in \mathbb{R}^{d_{\text{hidden}}}
$$

is a learned scale vector. For a full batch, RMSNorm acts independently on each token position:

$$
X \in \mathbb{R}^{B \times T \times d_{\text{hidden}}}
\longrightarrow
\mathrm{RMSNorm}(X) \in \mathbb{R}^{B \times T \times d_{\text{hidden}}}.
$$

Unlike LayerNorm, RMSNorm does not subtract the mean. It only controls the vector magnitude. This keeps the scale of the residual stream more stable while preserving the direction of the hidden vector. In this model, RMSNorm is used before attention and before MoE, so each sublayer receives inputs with a controlled scale before applying large learned transformations.

Complexity:

- Time: $O(BT d_{\text{hidden}})$
- Activation memory: $O(BT d_{\text{hidden}})$
- Parameter memory: $O(d_{\text{hidden}})$

The reason for using pre-normalization instead of post-normalization is mainly to make optimization more stable in deep residual networks. [TODO: discuss pre-norm vs post-norm later in a consolidated design-choice section.]


## Multi-Head Latent Attention (MHLA) with Rotary Position Embedding (RoPE)

Let the input to attention be

$$
X \in \mathbb{R}^{B \times T \times d_{\text{hidden}}}.
$$

This attention layer uses RoPE, which is how positional information is injected into the attention mechanism. Before applying RoPE, the model first constructs the non-rotary and rotary pieces of the queries and keys, along with the latent representation that will produce the remaining keys and values.

### Pre-RoPE part of Attention

---

The hidden dimension is split across $n_{\text{heads}}$ attention heads:

$$
d_{\text{head}} = \frac{d_{\text{hidden}}}{n_{\text{heads}}}.
$$

In this model, each head is further split into a non-rotary part and a rotary part:

$$
d_{\text{head}} = d_{\text{nope}} + d_{\text{rope}}.
$$

The name `nope` means "no positional encoding": this part of the query/key vector is not rotated by RoPE. The `rope` part is the part that will receive rotary position information.

Before RoPE is applied, the query is projected into these two pieces separately:

$$
Q_{\text{nope}} = \mathrm{reshape}(XW_{Q,\text{nope}}) \in \mathbb{R}^{B \times n_{\text{heads}} \times T \times d_{\text{nope}}},
$$

where

$$
W_{Q,\text{nope}} \in \mathbb{R}^{d_{\text{hidden}} \times (n_{\text{heads}}d_{\text{nope}})},
$$

and

$$
Q_{\text{rope, pre}} = \mathrm{reshape}(XW_{Q,\text{rope}}) \in \mathbb{R}^{B \times n_{\text{heads}} \times T \times d_{\text{rope}}},
$$

where

$$
W_{Q,\text{rope}} \in \mathbb{R}^{d_{\text{hidden}} \times (n_{\text{heads}}d_{\text{rope}})}.
$$

The key also has a rotary part, but unlike the query RoPE projection, this rotary key is shared across heads before being expanded later:

$$
K_{\text{rope, pre}} = \mathrm{reshape}(XW_{K,\text{rope}}) \in \mathbb{R}^{B \times 1 \times T \times d_{\text{rope}}},
$$

where

$$
W_{K,\text{rope}} \in \mathbb{R}^{d_{\text{hidden}} \times d_{\text{rope}}}.
$$

The non-rotary key and the value do not come directly from $X$. Instead, the model first compresses $X$ into a latent representation:

$$
L_{\text{pre}} = XW_L \in \mathbb{R}^{B \times T \times d_{\text{latent}}},
$$

where

$$
W_L \in \mathbb{R}^{d_{\text{hidden}} \times d_{\text{latent}}}.
$$

This latent vector is then RMS-normalized:

$$
L = \mathrm{RMSNorm}(L_{\text{pre}}).
$$

This is the "latent" part of multi-head latent attention: instead of storing or deriving all key/value information directly at full hidden width, the model first stores a smaller latent state of width $d_{\text{latent}}$. Later, this latent state is expanded into the non-rotary keys and values.

From this normalized latent representation, the non-rotary key is produced:

$$
K_{\text{nope}} = \mathrm{reshape}(LW_{K,\text{nope}}) \in \mathbb{R}^{B \times n_{\text{heads}} \times T \times d_{\text{nope}}},
$$

where

$$
W_{K,\text{nope}} \in \mathbb{R}^{d_{\text{latent}} \times (n_{\text{heads}}d_{\text{nope}})}.
$$

The value vectors are also produced from the same latent representation:

$$
V_{\text{attn}} = \mathrm{reshape}(LW_V) \in \mathbb{R}^{B \times n_{\text{heads}} \times T \times d_{\text{head}}},
$$

where

$$
W_V \in \mathbb{R}^{d_{\text{latent}} \times (n_{\text{heads}}d_{\text{head}})}.
$$

So before applying RoPE, the attention layer has constructed:

- $Q_{\text{nope}} \in \mathbb{R}^{B \times n_{\text{heads}} \times T \times d_{\text{nope}}}$
- $Q_{\text{rope, pre}} \in \mathbb{R}^{B \times n_{\text{heads}} \times T \times d_{\text{rope}}}$
- $K_{\text{rope, pre}} \in \mathbb{R}^{B \times 1 \times T \times d_{\text{rope}}}$
- $L \in \mathbb{R}^{B \times T \times d_{\text{latent}}}$
- $K_{\text{nope}} \in \mathbb{R}^{B \times n_{\text{heads}} \times T \times d_{\text{nope}}}$
- $V_{\text{attn}} \in \mathbb{R}^{B \times n_{\text{heads}} \times T \times d_{\text{head}}}$

Time complexity before applying RoPE:

- $Q_{\text{nope}}$: $O(BT d_{\text{hidden}} n_{\text{heads}} d_{\text{nope}})$
- $Q_{\text{rope, pre}}$: $O(BT d_{\text{hidden}} n_{\text{heads}} d_{\text{rope}})$
- $K_{\text{rope, pre}}$: $O(BT d_{\text{hidden}} d_{\text{rope}})$
- $L_{\text{pre}}$: $O(BT d_{\text{hidden}} d_{\text{latent}})$
- RMSNorm on $L_{\text{pre}}$: $O(BT d_{\text{latent}})$
- $K_{\text{nope}}$: $O(BT d_{\text{latent}} n_{\text{heads}} d_{\text{nope}})$
- $V_{\text{attn}}$: $O(BT d_{\text{latent}} n_{\text{heads}} d_{\text{head}})$

### Applying RoPE

---

RoPE is applied only to the rotary parts of the query and key:

$$
Q_{\text{rope, pre}} \in \mathbb{R}^{B \times n_{\text{heads}} \times T \times d_{\text{rope}}},
$$

$$
K_{\text{rope, pre}} \in \mathbb{R}^{B \times 1 \times T \times d_{\text{rope}}}.
$$

The dimension $d_{\text{rope}}$ must be even, because RoPE rotates pairs of coordinates. For each pair index

$$
m \in \{0, 1, \dots, \frac{d_{\text{rope}}}{2} - 1\},
$$

the model defines the angular frequency

$$
\theta_m = 10000^{-\frac{2m}{d_{\text{rope}}}}.
$$

At sequence position $t$, the rotation angle for pair $m$ is

$$
\phi_{t,m} = t\theta_m.
$$

Now take a rotary vector

$$
z_t \in \mathbb{R}^{d_{\text{rope}}}.
$$

Group its coordinates into pairs:

$$
(z_{t,2m}, z_{t,2m+1}).
$$

RoPE rotates each pair by the position-dependent angle $\phi_{t,m}$:

$$
\begin{aligned}
\begin{bmatrix}
z'_{t,2m} \\
z'_{t,2m+1}
\end{bmatrix}
&=
\begin{bmatrix}
\cos(\phi_{t,m}) & -\sin(\phi_{t,m}) \\
\sin(\phi_{t,m}) & \cos(\phi_{t,m})
\end{bmatrix}
\begin{bmatrix}
z_{t,2m} \\
z_{t,2m+1}
\end{bmatrix}.
\end{aligned}
$$

Equivalently,

$$
z'_{t,2m} = z_{t,2m}\cos(\phi_{t,m}) - z_{t,2m+1}\sin(\phi_{t,m}),
$$

$$
z'_{t,2m+1} = z_{t,2m}\sin(\phi_{t,m}) + z_{t,2m+1}\cos(\phi_{t,m}).
$$

Applying this rotation to the query and key rotary parts gives

$$
Q_{\text{rope}} = \mathrm{RoPE}(Q_{\text{rope, pre}}) \in \mathbb{R}^{B \times n_{\text{heads}} \times T \times d_{\text{rope}}},
$$

and

$$
K_{\text{rope}} = \mathrm{RoPE}(K_{\text{rope, pre}}) \in \mathbb{R}^{B \times 1 \times T \times d_{\text{rope}}}.
$$

RoPE injects position without adding a separate position vector to $X$. Instead, it makes the query-key dot product depend on relative position through rotations of the query and key coordinate pairs.

Time complexity:

- Computing angles: $O(T d_{\text{rope}})$
- Rotating $Q_{\text{rope, pre}}$: $O(BT n_{\text{heads}} d_{\text{rope}})$
- Rotating $K_{\text{rope, pre}}$: $O(BT d_{\text{rope}})$


### Post-RoPE part of Attention

---

After RoPE, the full query and key vectors are assembled by concatenating their non-rotary and rotary parts:

$$
Q = [Q_{\text{nope}};\ Q_{\text{rope}}],
\qquad
K = [K_{\text{nope}};\ \widetilde{K}_{\text{rope}}].
$$

The query already has a separate RoPE part for each head:

$$
Q_{\text{rope}} \in \mathbb{R}^{B \times n_{\text{heads}} \times T \times d_{\text{rope}}}.
$$

The rotary key was produced with only one head dimension:

$$
K_{\text{rope}} \in \mathbb{R}^{B \times 1 \times T \times d_{\text{rope}}}.
$$

So the rotary key is shared across heads by expanding it:

$$
K_{\text{rope}}
\longrightarrow
\widetilde{K}_{\text{rope}} \in \mathbb{R}^{B \times n_{\text{heads}} \times T \times d_{\text{rope}}}.
$$

Then the non-rotary and rotary parts are concatenated:

$$
Q = \mathrm{concat}(Q_{\text{nope}}, Q_{\text{rope}}) \in \mathbb{R}^{B \times n_{\text{heads}} \times T \times d_{\text{head}}},
$$

$$
K = \mathrm{concat}(K_{\text{nope}}, \widetilde{K}_{\text{rope}}) \in \mathbb{R}^{B \times n_{\text{heads}} \times T \times d_{\text{head}}}.
$$

The values were already constructed from the latent representation:

$$
V_{\text{attn}} \in \mathbb{R}^{B \times n_{\text{heads}} \times T \times d_{\text{head}}}.
$$

For each head, attention computes the scaled query-key similarity matrix:

$$
S = \frac{QK^T}{\sqrt{d_{\text{head}}}} \in \mathbb{R}^{B \times n_{\text{heads}} \times T \times T}.
$$

The factor $\frac{1}{\sqrt{d_{\text{head}}}}$ controls the variance of the dot products. If the query and key coordinates have roughly constant variance, then their dot product is a sum of $d_{\text{head}}$ terms, so its variance grows like $O(d_{\text{head}})$. Without scaling, larger head dimensions would produce larger attention logits, pushing the softmax into a saturated regime where one token gets almost all the probability mass and gradients become less useful.

A causal mask is applied so position $t$ can only attend to positions $\leq t$:

$$
S_{t,j} = -\infty \quad \text{for } j > t.
$$

Then attention weights are computed with softmax:

$$
A = \mathrm{softmax}(S) \in \mathbb{R}^{B \times n_{\text{heads}} \times T \times T}.
$$

The attention output is the weighted sum of values:

$$
O = AV_{\text{attn}} \in \mathbb{R}^{B \times n_{\text{heads}} \times T \times d_{\text{head}}}.
$$

The heads are then rearranged back into the hidden dimension:

$$
O_{\text{merged}} \in \mathbb{R}^{B \times T \times (n_{\text{heads}}d_{\text{head}})} = \mathbb{R}^{B \times T \times d_{\text{hidden}}}.
$$

Finally, the model applies an output mixing projection and dropout:

$$
\mathrm{Attention}(X) = \mathrm{Dropout}(O_{\text{merged}}W_O) \in \mathbb{R}^{B \times T \times d_{\text{hidden}}},
$$

where

$$
W_O \in \mathbb{R}^{d_{\text{hidden}} \times d_{\text{hidden}}}.
$$

Time complexity:

- Expanding and concatenating into $Q$ and $K$: $O(BT n_{\text{heads}} d_{\text{head}})$
- Attention scores $QK^T$: $O(B n_{\text{heads}} T^2 d_{\text{head}})$
- Attention output $AV_{\text{attn}}$: $O(B n_{\text{heads}} T^2 d_{\text{head}})$
- Output mixing projection: $O(BT d_{\text{hidden}}^2)$

The important quadratic term is the attention matrix:

$$
A \in \mathbb{R}^{B \times n_{\text{heads}} \times T \times T},
$$

which is why attention becomes expensive as the context length $T$ grows.

## Mixture of Experts Feedforward

After attention and the residual update, the block applies RMSNorm and then a mixture-of-experts feedforward layer. Let the input to MoE be

$$
X \in \mathbb{R}^{B \times T \times d_{\text{hidden}}}.
$$

Flatten the batch and sequence dimensions:

$$
N = BT,
\qquad
X_{\text{flat}} \in \mathbb{R}^{N \times d_{\text{hidden}}}.
$$

The router assigns each token a distribution over $E = n_{\text{experts}}$ experts. First it computes router scores:

$$
R = X_{\text{flat}}W_R \in \mathbb{R}^{N \times E},
$$

where

$$
W_R \in \mathbb{R}^{d_{\text{hidden}} \times E}.
$$

The router probabilities are

$$
P = \mathrm{softmax}(R) \in \mathbb{R}^{N \times E}.
$$

For each token $i$, only the top $k = n_{\text{top experts}}$ experts are selected:

$$
\mathcal{E}_i = \mathrm{TopK}(P_i, k).
$$

The selected expert weights are renormalized over only the chosen experts:

$$
\alpha_{i,e} = \frac{P_{i,e}}{\sum_{e' \in \mathcal{E}_i} P_{i,e'}}
\qquad \text{for } e \in \mathcal{E}_i.
$$

Each expert is a gated feedforward network. For expert $e$, with input token vector $x_i \in \mathbb{R}^{d_{\text{hidden}}}$,

$$
\begin{aligned}
\mathrm{Expert}_e(x_i)
&=
\left[\left(x_i W_{e,\text{up}}\right)
\odot
\mathrm{SiLU}\left(x_i W_{e,\text{gate}}\right)\right]
W_{e,\text{down}},
\end{aligned}
$$

where

$$
W_{e,\text{up}}, W_{e,\text{gate}} \in \mathbb{R}^{d_{\text{hidden}} \times d_{\text{intermediate}}},
$$

and

$$
W_{e,\text{down}} \in \mathbb{R}^{d_{\text{intermediate}} \times d_{\text{hidden}}}.
$$

The final MoE output for token $i$ is the weighted sum of its selected expert outputs:

$$
y_i = \sum_{e \in \mathcal{E}_i} \alpha_{i,e}\mathrm{Expert}_e(x_i).
$$

After reshaping back to batch and sequence form, the MoE output is

$$
\mathrm{MoE}(X) \in \mathbb{R}^{B \times T \times d_{\text{hidden}}}.
$$

The implementation uses a capacity per expert:

$$
C_{\text{expert}} = \left\lfloor \text{capacity factor} \cdot \frac{Nk}{E} \right\rfloor.
$$

This means each expert processes at most $C_{\text{expert}}$ routed token slots. If too many tokens are routed to one expert, only the highest-weight assignments are kept for that expert.

The router also contributes a load-balancing loss. Let

$$
p_e = \frac{1}{N}\sum_{i=1}^N P_{i,e}
$$

be the average router probability mass for expert $e$, and let

$$
f_e
$$

be the fraction of selected expert assignments that go to expert $e$. The load-balancing term is

$$
\mathcal{L}_{\text{load}} = E \sum_{e=1}^{E} p_e f_e.
$$

This discourages the router from collapsing onto only a few experts.

Time complexity:

- Router projection: $O(N d_{\text{hidden}} E)$
- Router softmax and top-$k$: $O(NE)$
- Expert computation: $O(E C_{\text{expert}} d_{\text{hidden}} d_{\text{intermediate}})$
- Since $E C_{\text{expert}} \approx Nk$, expert computation is approximately $O(Nk d_{\text{hidden}} d_{\text{intermediate}})$

The main idea is that MoE increases parameter capacity by having many experts, but each token only uses $k$ of them.

## Final Normalization, Unembedding, and Loss

After all transformer blocks, the model has the final hidden state

$$
X_{\text{hidden}}^{(n_{\text{layers}})} \in \mathbb{R}^{B \times T \times d_{\text{hidden}}}.
$$

A final RMSNorm is applied:

$$
\widetilde{X} = \mathrm{RMSNorm}(X_{\text{hidden}}^{(n_{\text{layers}})}) \in \mathbb{R}^{B \times T \times d_{\text{hidden}}}.
$$

Then the model maps each hidden vector back to vocabulary space. Since the token embedding table has shape

$$
W_{\text{token embedding table}} \in \mathbb{R}^{V \times d_{\text{hidden}}},
$$

and the model ties the unembedding weights to the embedding weights, the vocabulary logits are

$$
Z = \widetilde{X}W_{\text{token embedding table}}^T \in \mathbb{R}^{B \times T \times V}.
$$

For each position $(b,t)$, the vector

$$
Z^{b,t,:} \in \mathbb{R}^{V}
$$

contains the raw scores for the next token over the whole vocabulary.

During training, the target tensor is

$$
Y \in \{0, 1, \dots, V - 1\}^{B \times T},
$$

where $Y^{b,t}$ is the correct next token id for position $(b,t)$. The cross-entropy loss is

$$
\begin{aligned}
\mathcal{L}_{\text{CE}}
&=
-\frac{1}{BT}
\sum_{b=1}^{B}
\sum_{t=1}^{T}
\log
\left(
\frac{\exp(Z^{b,t,Y^{b,t}})}
{\sum_{v=1}^{V}\exp(Z^{b,t,v})}
\right).
\end{aligned}
$$

The full training loss also includes the MoE load-balancing penalty:

$$
\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda_{\text{load}}\mathcal{L}_{\text{load}},
$$

where $\lambda_{\text{load}}$ controls the strength of the load-balancing term.

Time complexity:

- Final RMSNorm: $O(BT d_{\text{hidden}})$
- Unembedding logits: $O(BT d_{\text{hidden}} V)$
- Cross entropy over vocabulary: $O(BT V)$

The unembedding is expensive when $V$ is large because every token position produces a score for every vocabulary item.

## Design Choices and Deeper Explanations

### Why Residual Connections?

Each transformer block updates the residual stream by adding learned changes to it:

$$
X \longmapsto X + f(X).
$$

In this model, each block has two residual updates:

$$
Y^{(i)} = X_{\text{hidden}}^{(i)} + \mathrm{Attention}(\mathrm{RMSNorm}(X_{\text{hidden}}^{(i)})),
$$

$$
X_{\text{hidden}}^{(i+1)} = Y^{(i)} + \mathrm{MoE}(\mathrm{RMSNorm}(Y^{(i)})).
$$

The residual path gives information and gradients a direct route through the network. Instead of forcing every layer to completely rewrite the representation, each sublayer can learn an update $\Delta X$ to the existing representation.

### Why Residual Scaling?

Because residual branches are repeatedly added, their variances can accumulate with depth. If each branch contributes a roughly independent update with similar variance, then after many residual additions the residual stream can grow in scale.

This model has two residual branches per layer: one attention branch and one MoE branch. Across $n_{\text{layers}}$ layers, that gives roughly

$$
2n_{\text{layers}}
$$

residual-producing branches. To control the initial scale of these added updates, the output projections on residual branches are initialized with the extra factor

$$
(2n_{\text{layers}})^{-1/2}.
$$

So if the usual initialization scale is $\sigma$, the residual branch output projection uses

$$
\sigma_{\text{residual}} = \sigma(2n_{\text{layers}})^{-1/2}.
$$

The goal is not to remove residual updates, but to make their initial contribution small enough that the residual stream stays well-scaled at the start of training.

### Why Pre-Norm Instead of Post-Norm?

In a pre-norm block, the sublayer receives normalized input, but the residual path itself remains a direct identity path:

$$
X \longmapsto X + f(\mathrm{RMSNorm}(X)).
$$

This makes optimization more stable because gradients can flow through the residual addition without having to pass through a normalization operation at every layer. Post-norm can work, but in deeper transformers it is often harder to optimize because the residual path is repeatedly transformed by normalization after each addition.

### Why Scale Attention by $1 / \sqrt{d_{\text{head}}}$?

The query-key score is a dot product:

$$
q \cdot k = \sum_{c=1}^{d_{\text{head}}} q_c k_c.
$$

If the coordinates have roughly constant variance, then the variance of this sum grows like $O(d_{\text{head}})$. Dividing by $\sqrt{d_{\text{head}}}$ keeps the score scale roughly stable as the head dimension changes. This prevents the softmax from becoming too sharp just because $d_{\text{head}}$ is large.

### Why Causal Masking?

This is an autoregressive language model, so token position $t$ is trained to predict the next token using only earlier context. The causal mask enforces

$$
\text{position } t \text{ can attend only to positions } \leq t.
$$

Without this mask, the model could look at future tokens during training, making the next-token prediction task invalid.

### Why RoPE?

Attention by itself is permutation-equivariant: without positional information, it does not know whether a token came earlier or later in the sequence. RoPE injects position by rotating query and key coordinates as a function of token position.

The important effect is that the dot product between a rotated query and a rotated key depends on their relative positions. So RoPE gives attention a way to reason about distance and order without adding a separate learned position vector to the hidden state.

### Why Dropout?

Dropout randomly removes part of a sublayer's output during training. In this model it is applied after attention output mixing and after the MoE output. This makes the residual updates less brittle: the model cannot rely too heavily on any single activation pathway, which helps regularization.

At inference time, dropout is disabled, so the full learned computation is used.

### Why MoE Instead of a Dense MLP?

A dense MLP applies the same feedforward network to every token. MoE instead contains many expert MLPs, but each token is routed to only $k$ experts. This increases the number of parameters available to the model without making every token pay the compute cost of every expert.

The tradeoff is that routing introduces extra complexity: the model must decide which experts to use, pack tokens by expert, respect capacity limits, and avoid routing collapse.

### Why Load Balancing?

Without load balancing, the router can collapse onto a small number of experts. Then most experts receive little or no training signal, and the active experts become overloaded.

The load-balancing term encourages two things to agree: how much probability mass the router assigns to each expert, and how many selected token assignments each expert actually receives. This pushes the model toward using the expert pool more evenly.

### Why Weight Tying?

The model uses the same matrix for token embedding and unembedding. The embedding table maps token ids into hidden vectors, while the transposed same matrix maps hidden vectors back to vocabulary logits.

This reduces the number of parameters and forces the input-token geometry and output-token geometry to share the same representation space.

### Where and Why RMSNorm Is Used

RMSNorm is used wherever the model wants to control the scale of a representation before an important learned transformation. In the transformer blocks, it is used before attention:

$$
\mathrm{Attention}(\mathrm{RMSNorm}(X)).
$$

This gives the attention projections inputs with a stable magnitude while keeping the residual path itself as a direct identity path.

It is also used before MoE:

$$
\mathrm{MoE}(\mathrm{RMSNorm}(Y)).
$$

This is important because the router and experts are sensitive to input scale. If the hidden vectors entering MoE grow too large or too small, routing logits and expert activations can become poorly scaled.

RMSNorm is also used on the latent representation inside attention:

$$
L = \mathrm{RMSNorm}(L_{\text{pre}}).
$$

Here it stabilizes the compressed key/value latent space before that latent vector is expanded into non-rotary keys and values.

Finally, RMSNorm is used before unembedding:

$$
\widetilde{X} = \mathrm{RMSNorm}(X_{\text{hidden}}^{(n_{\text{layers}})}).
$$

This controls the scale of the final hidden vectors before they are projected into vocabulary logits.

RMSNorm is not applied after every operation because normalization is not free and too much normalization can interfere with the residual stream. The model mainly normalizes before large learned transformations, while leaving residual additions themselves unnormalized so information and gradients can flow directly through the network.

### Why Latent Key/Value Representations?

Instead of directly producing all key and value information from the full hidden dimension, the model first compresses the hidden state into

$$
L \in \mathbb{R}^{B \times T \times d_{\text{latent}}}.
$$

The non-rotary keys and values are then produced from this smaller latent representation. This makes the key/value pathway more compact and is especially useful for inference, where past key/value information may be stored in a cache.

The tradeoff is that $d_{\text{latent}}$ becomes a bottleneck: it saves memory and parameters, but it also limits how much information can pass through the latent key/value path.

### Why Split RoPE and NoPE Dimensions?

The attention head is split as

$$
d_{\text{head}} = d_{\text{nope}} + d_{\text{rope}}.
$$

The RoPE dimensions carry position-aware information, while the NoPE dimensions remain unrotated. This lets the model mix two kinds of similarity: content similarity that is not directly position-rotated, and position-aware similarity through RoPE.

Using RoPE on only part of the head also reduces the amount of rotary computation.

### Why Share the RoPE Key Across Heads?

The query has a separate RoPE component for each head, but the rotary key is produced with only one head dimension and then shared across all heads.

This reduces the amount of key-side positional information that must be produced and stored. Each head still has its own query, but they compare against a shared positional key component.

### Why Top-$k$ Routing?

Using all experts for every token would turn MoE back into a very expensive dense ensemble. Top-$k$ routing keeps the computation sparse:

$$
\text{each token uses only } k \text{ experts}.
$$

This gives the model access to many expert parameters while keeping per-token compute closer to a small number of MLPs.

### Why Expert Capacity?

Routing can be uneven: many tokens might choose the same expert. Expert capacity puts a limit on how many routed token slots each expert processes.

This keeps computation bounded and makes the packed expert tensors have predictable shapes. The tradeoff is that if an expert receives too many assignments, some lower-weight assignments are dropped.

### Why Gated Experts?

Each expert uses a gated MLP:

$$
\left(xW_{\text{up}}\right) \odot \mathrm{SiLU}(xW_{\text{gate}}).
$$

The gate controls which intermediate features are emphasized for each token. Compared with a plain MLP, the multiplicative interaction gives the expert a richer way to modulate features before projecting back to the hidden dimension.

### Why No Bias in Many Linear Layers?

Many projections use no bias. In attention and routing, this keeps the transformations slightly simpler and centered around learned linear geometry.

In the expert MLPs, no bias is also useful for the packed-capacity implementation: padded expert slots contain zeros, and without bias those padded slots produce zero contribution instead of accidentally producing nonzero expert outputs.
