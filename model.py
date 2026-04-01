class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.scale  = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x):  # [B,T,C]
        assert x.size(-1) == self.scale.size(0), 'dimension mismatch in RMSNorm'
        input_dtype = x.dtype
        x = x.float()
        var = (x.pow(2)).mean(dim=-1, keepdim=True)  # [B,T,1]
        x = x * torch.rsqrt(var + self.eps)  # [B,T,C] * [B,T,1] --broadcast-> [B,T,C] * [B,T,C(rep)] = [B,T,C]
        x = self.scale * x # [C] * [B,T,C] --broadcast-> [B(rep),T(rep),C] * [B,T,C] = [B,T,C]
        x = x.to(input_dtype)
        return x

class Multi_Headed_Latent_Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_nope = nn.Linear(d_hidden, n_heads * d_nope, bias=False)
        self.q_rope = nn.Linear(d_hidden, n_heads * d_rope, bias=False)

        self.k_rope = nn.Linear(d_hidden, 1 * d_rope, bias=False)
        
        self.latent_proj = nn.Linear(d_hidden, d_latent, bias=False)
        self.latent_norm = RMSNorm(d_latent)
        self.k_nope = nn.Linear(d_latent, n_heads * d_nope, bias=False)
        self.v = nn.Linear(d_latent, n_heads * d_head, bias=False)
        
        self.mix   = nn.Linear(d_hidden, d_hidden, bias=False)
        self.mix.residual = True
        # self.register_buffer("tril", torch.tril(torch.ones(block_size,block_size)))
        self.drop  = nn.Dropout(dropout)

        m = torch.arange(d_rope//2, dtype=torch.float32) # [d_rope/2]
        theta = 10000.0 ** (-2 * m / d_rope)  # [d_rope/2]
        self.register_buffer("rope_theta", theta, persistent=False) # [d_rope/2]

    def apply_RoPE(self, x, position_id): # x.shape = [B,n_heads (or any),T,d_rope]
        B,n,T,d = x.shape
        input_dtype = x.dtype
        x = x.float()
        x = x.reshape(B,n,T,d//2,2)  # [B,n,T,d/2,2]

        if position_id is None: # training phase and pre-fill phase
            t = torch.arange(T, dtype=torch.float32, device=x.device) # [T]
            phi = self.rope_theta[None,:] * t[:,None]  # [1,d_rope/2] * [T,1] --broadcast-> [T(rep),d_rope/2] * [T,d_rope/2(rep)] = [T,d_rope/2]
            cos = phi.cos().view(1,1,T,d//2) # [1,1,T,d/2]
            sin = phi.sin().view(1,1,T,d//2) # [1,1,T,d/2]
        else:
            assert T == 1, 'decode phase must have single token in a batch'
            t = torch.arange(position_id, position_id+1, dtype=torch.float32, device=x.device) # [T=1]
            phi = self.rope_theta[None,:] * t[:,None]  # [1,d_rope/2] * [T=1,1] --broadcast-> [1,d_rope/2] * [1,d_rope/2(rep)] = [T=1,d_rope/2]
            cos = phi.cos().view(1,1,T,d//2) # [1,1,T=1,d/2]
            sin = phi.sin().view(1,1,T,d//2) # [1,1,T=1,d/2]         
            
        xrot_even = x[:,:,:,:,0] * cos - x[:,:,:,:,1] * sin   #  [B,n,T,d/2] * [1,1,T,d/2] --broadcast-> [B,n,T,d/2] * [B(rep),n(rep),T,d/2] = [B,n,T,d/2]
        xrot_odd  = x[:,:,:,:,0] * sin + x[:,:,:,:,1] * cos   #  [B,n,T,d/2]
        xrot = torch.stack((xrot_even, xrot_odd), dim=-1) # [B,n,T,d/2,2]
        xrot = xrot.reshape(B, n, T, d)
        xrot = xrot.to(input_dtype)
        return xrot
        
    def forward(self, x, cache, position_id, layer_idx):
        B,T,C = x.shape # [B,T,C=d_hidden]

        # query
        query_nope = self.q_nope(x) # [B,T,n_heads * d_nope]
        query_nope = query_nope.view(B,T,n_heads,d_nope).transpose(1,2)  # [B,T,n_heads * d_nope] --view-> [B,T,n_heads,d_nope] --tr-> [B,n_heads,T,d_nope]

        query_rope = self.q_rope(x) # [B,T,n_heads * d_rope]
        query_rope = query_rope.view(B,T,n_heads,d_rope).transpose(1,2)  # [B,T,n_heads * d_rope] --view-> [B,T,n_heads,d_rope] --tr-> [B,n_heads,T,d_rope]
        query_rope = self.apply_RoPE(query_rope, position_id)

        query = torch.cat((query_nope,query_rope),dim=-1) 

        # latent
        latent = self.latent_proj(x) # [B,T,d_latent]
        latent = self.latent_norm(latent)

        # key_rope
        key_rope = self.k_rope(x) # [B,T,1 * d_rope]
        key_rope = key_rope.view(B,T,1,d_rope).transpose(1,2)  # [B,T,1 * d_rope] --view-> [B,T,1,d_rope] --tr-> [B,1,T,d_rope]
        key_rope = self.apply_RoPE(key_rope, position_id) # [B,1,T,d_rope]

        T_k = T
        # Cache Latent and key_rope
        if cache is not None: # prefill or decode phase
            if position_id is None: # prefill phase
                cache[layer_idx]['kv_latent'] = latent    # [B,T,d_latent]
                cache[layer_idx]['k_rope']    = key_rope  # [B,1,T,d_rope]
            else:  # decode phase
                assert T == 1, 'decode phase must have single token in a batch'
                # Concat to store new latent and key_rope
                cache[layer_idx]['kv_latent']  = torch.cat((cache[layer_idx]['kv_latent'], latent), dim=1)    # [B,T_prev,d_latent] cat [B,1,d_latent] = [B,T_prev+1,d_latent]
                cache[layer_idx]['k_rope'] = torch.cat((cache[layer_idx]['k_rope'], key_rope), dim=2)  # [B,1,T_prev,d_rope] cat [B,1,1,d_rope] = [B,1,T_prev+1,d_rope]
                # Crop to fit within context window
                cache[layer_idx]['kv_latent'] = cache[layer_idx]['kv_latent'][:,-block_size:,:]  
                cache[layer_idx]['k_rope'] = cache[layer_idx]['k_rope'][:,:,-block_size:,:]
                # Retrieve all of latent and key_rope
                latent   = cache[layer_idx]['kv_latent']
                key_rope = cache[layer_idx]['k_rope']
                T_k = latent.size(1)

        # value
        value = self.v(latent) # [B,T_k,n_heads * d_head]
        value = value.view(B,T_k,n_heads,d_head).transpose(1,2)  # [B,T_k,n_heads * d_head] --view-> [B,T_k,n_heads,d_head] --tr-> [B,n_heads,T_k,d_head]
        
        # key_nope 
        key_nope = self.k_nope(latent) # [B,T_k,n_heads * d_nope]
        key_nope = key_nope.view(B,T_k,n_heads,d_nope).transpose(1,2)  # [B,T_k,n_heads * d_nope] --view-> [B,T_k,n_heads,d_nope] --tr-> [B,n_heads,T_k,d_nope]

        # share key_rope across heads and concat with key_nope to get key
        key_rope = key_rope.expand(-1,n_heads,-1,-1)  # [B,n_heads(rep),T_k,d_rope] (! No more inplace writes on this hereafter)
        key = torch.cat((key_nope,key_rope),dim=-1) # [B,n_heads,T_k,d_nope] cat [B,n_heads,T_k,d_rope] = [B,n_heads,T_k,d_nope+d_rope=d_head]
        
        # wei = query @ key.transpose(-1,-2) * d_head**-0.5   # [B,n_heads,T of query,T of key] # scaled attention [To control variance]
        # wei = wei.masked_fill(self.tril[:T,:T]==0, float('-inf'))
        # wei = F.softmax(wei,-1)
        # wei = self.drop(wei)
        # attn = wei @ value

        # Flash attention (torch.compile doesn't know how to fuse the above 5 operations of attention and so the below does that) # [speedup] useless for mps
        if position_id is None: # training or prefill
            attn = F.scaled_dot_product_attention(query, key, value, is_causal=True) # do as usual with causal mask in train and prefil phase
        else: # decode
            attn = F.scaled_dot_product_attention(query, key, value, is_causal=False) # no causal mask in decode phase as the single query must attend to all keys as they are already in the querie's past
        
        attn = (attn).transpose(1,2).reshape(B,T,C)   # [B,n_heads,T_q,T_k] @ [B,n_heads,T_k,d_head] = [B,n_heads,T_q,d_head] --tr-> [B,T_q,n_heads,d_head] --view-> [B,T_q,n_heads*d_head = C]
        attn = self.mix(attn)
        attn = self.drop(attn)
        return attn, cache

class MLPexperts(nn.Module): # No bias must be used here cuz I rely on that to have contributions of padded tokens to be zero.
    def __init__(self):
        super().__init__()
        self.up_proj   = nn.Parameter(torch.randn(n_experts, d_hidden, d_intermediate) * 0.02)  # to maintain uniformity of std=0.02
        self.gate_proj = nn.Parameter(torch.randn(n_experts, d_hidden, d_intermediate) * 0.02)
        self.act       = nn.SiLU()
        self.down      = nn.Parameter(torch.randn(n_experts, d_intermediate, d_hidden) * 0.02 * ((2 * n_layers) ** -0.5)) # extra scaling for the residual connections scaling 
    def forward(self, x):  # x.shape = [n_expert,capacity,d_hidden]
        up_proj   = x @ self.up_proj  # [n_experts,capacity,d_hidden] @ [n_experts,d_hidden, d_intermediate] = [n_experts,capacity,d_intermediate]
        gate_proj = x @ self.gate_proj # [n_experts,capacity,d_hidden] @ [n_experts,d_hidden, d_intermediate] = [n_experts,capacity,d_intermediate]
        out = up_proj * self.act(gate_proj)
        out = out @ self.down # [n_experts,capacity,d_intermediate] @ [n_experts,d_intermediate, d_hidden] = [n_experts,capacity,d_hidden]
        return out

class Router(nn.Module):
    def __init__(self):
        super().__init__()
        self.router_logits = nn.Linear(d_hidden, n_experts, bias=False)
    def forward(self,x): # x.shape = [N,d_hidden]
        router_logits = self.router_logits(x) # [N,n_experts]
        router_probs = F.softmax(router_logits, dim=-1) # [N,n_experts]
        
        values, top_expert_ids = router_probs.topk(n_top_experts, dim=-1) # [N,n_top_experts], [N,n_top_experts]
        if n_top_experts == 1:
            top_expert_weights = values # [N,n_top_experts]
        else:
            top_expert_weights = values / values.sum(dim=-1, keepdim=True) # [N,n_top_experts]

        expert_prob_mass = router_probs.mean(dim=0) # [n_experts]

        return top_expert_weights, top_expert_ids, expert_prob_mass

class MoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.router = Router()
        self.MLPexperts = MLPexperts()
        self.drop = nn.Dropout(dropout)
    def forward(self, x, total_load): # x.shape = [B,T,d_hidden]
        B,T,d = x.shape
        x = x.reshape(B*T,d) # [N=B*T,d_hidden]
        
        top_expert_weights, top_expert_ids, expert_prob_mass = self.router(x) # [N,n_top_experts], [N,n_top_experts], [n_experts]
        token_ids = torch.arange(B*T, device=x.device).unsqueeze(-1).expand(-1,n_top_experts)  # [N,n_top_experts(rep)]
        
        flat_weights = top_expert_weights.reshape(-1)  # [N*n_top_experts]
        flat_expert_ids = top_expert_ids.reshape(-1)  # [N*n_top_experts]
        flat_token_ids = token_ids.reshape(-1)  # [N*n_top_experts]
        
        counts = torch.bincount(flat_expert_ids, minlength=n_experts)
        expert_assignment_fraction = counts / counts.sum() # [n_experts]
        load_balance_loss = n_experts * (expert_prob_mass * expert_assignment_fraction).sum()
        total_load += load_balance_loss

        idx = flat_expert_ids.argsort()
        flat_weights    = flat_weights[idx]
        flat_expert_ids = flat_expert_ids[idx]  
        flat_token_ids  = flat_token_ids[idx]

        cum_assigns = torch.cumsum(counts, dim=0)
        boundary_pos = torch.cat((torch.tensor([0], device=x.device), cum_assigns))  

        capacity = max(1, int(capacity_factor * (B*T * n_top_experts) / n_experts))
        
        out = torch.zeros_like(x)
        x_packed = torch.zeros(n_experts, capacity, d_hidden, device=x.device)  # [n_experts,capacity,d_hidden]
        weights_packed = torch.zeros(n_experts, capacity, device=x.device)  # [n_experts,capacity]
        token_ids_packed = torch.zeros(n_experts, capacity, dtype=torch.long, device=x.device)  # [n_experts,capacity]

        nonempty = torch.nonzero(counts > 0).squeeze(-1)
        for e in nonempty:
            start = boundary_pos[e].item()  
            end   = boundary_pos[e+1].item()  
            selected_token_ids = flat_token_ids[start:end]
            selected_weights = flat_weights[start:end]
            if end - start > capacity:
                selected_weights, idx = selected_weights.topk(capacity)
                selected_token_ids = selected_token_ids[idx]
            Len = len(selected_token_ids)
            xe = x[selected_token_ids]
            
            x_packed[e,0:Len,:] = xe
            weights_packed[e,0:Len] = selected_weights
            token_ids_packed[e,0:Len] = selected_token_ids

        y_packed = self.MLPexperts(x_packed)   # [n_experts,capacity,d_hidden]
        
        y_unpacked = y_packed.reshape(-1,d_hidden)  # [n_experts*capacity,d_hidden]
        weights_unpacked = weights_packed.reshape(-1) # [n_experts*capacity]
        token_ids_unpacked = token_ids_packed.reshape(-1) # [n_experts*capacity]

        contrib = weights_unpacked.unsqueeze(-1) * y_unpacked # [M_valid, d_hidden] Note: the padded tokens would just contribute zero in this implementation as there is no bias in the mlp.
        #out[token_ids_masked] += contrib !!! Wrong thing to do because indices in token_ids_masked can repeat so it will only add the most recent one which is not what we want.
        out.index_add_(0, token_ids_unpacked, contrib)  # Correct thing to do: This does out[token_ids_masked[i]] += contrib[i] for all i

        out = out.reshape(B,T,d)  # [B,T,d_hidden]
        out = self.drop(out)
        return out, total_load

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_attn_rmsn  = RMSNorm(d_hidden)
        self.attn = Multi_Headed_Latent_Attention()
        self.pre_moe_rmsn  = RMSNorm(d_hidden)
        self.moe = MoE()
    def forward(self,x, total_load, cache, position_id, layer_idx):
        x_attn, cache = self.attn(self.pre_attn_rmsn(x), cache, position_id, layer_idx)
        x = x + x_attn
        x_moe, total_load = self.moe(self.pre_moe_rmsn(x), total_load) 
        x = x + x_moe
        return x, total_load, cache, position_id

class MyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, d_hidden)
        self.blocks = nn.ModuleList([Block() for _ in range(n_layers)])
        self.rmsn3 = RMSNorm(d_hidden)
        self.unembedding = nn.Linear(d_hidden, vocab_size, bias=False)
        
        self.apply(self._init_weights) # custom initilizatio 

        self.unembedding.weight = self.token_embedding_table.weight # weight tying
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if getattr(module, 'residual', False):
                std *= (2 * n_layers) ** -0.5 # [To control variance]
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
 
    def forward(self,x, targets=None, cache=None, position_id=None):
        if position_id is not None:
            assert cache is not None, 'position_id given without a cache. decode phase must have both and prefil phase must only have cache'
        B, T = x.shape
        token_embedding = self.token_embedding_table(x)
        x = token_embedding
        total_load = torch.tensor([0.0], device=x.device)
        layer_idx = 0
        for block in self.blocks:
            x, total_load, cache, position_id = block(x, total_load, cache, position_id, layer_idx)
            layer_idx += 1
        load_balance_loss = total_load / n_layers
        if targets is not None:
            logits = self.unembedding(self.rmsn3(x))
            cross_entropy_loss = F.cross_entropy(logits.view(B * T, vocab_size), targets.view(B * T))
            return logits, cross_entropy_loss, load_balance_loss
        else:
            logits = self.unembedding(self.rmsn3(x[:,[-1],:]))
            return logits, cache

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):   # this will set the weight decays of 2d tensors and 1d tensors differently
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer
        
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, use_kvcaching=True):
        was_training = self.training
        self.eval()
        if use_kvcaching:
            kv_cache = [{"kv_latent": None, "k_rope": None} for _ in range(n_layers)]
        else:
            kv_cache = None
        pos_id = None
        for _ in range(max_new_tokens):

            torch.mps.synchronize()
            t0 = time.perf_counter()
    
            if pos_id is not None and use_kvcaching:
                idx_cropped = idx[:,-1:]
            else:
                idx_cropped = idx[:,-block_size:]
                
            logits, kv_cache = self(idx_cropped, cache=kv_cache, position_id=pos_id)
            if pos_id is None and use_kvcaching:
                pos_id = idx_cropped.size(1)
            elif use_kvcaching:
                pos_id += 1
            logits = logits[:,-1,:]
            #temperature
            logits = logits / temperature
            # top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs,1)
            idx = torch.cat((idx,idx_next), dim=1)

            torch.mps.synchronize()
            t1 = time.perf_counter()
            dt = t1-t0
            print(f'dt: {dt*1000:.4f}ms')
            
        if was_training:
            self.train()
        return idx
