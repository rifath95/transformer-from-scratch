import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import matplotlib.pyplot as plt

from config import *
from data import *
from model import *


# Optional CUDA optimization
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')

# Initiate Model and Training Schedules

model = MyGPT()
model = model.to(device)
#model = torch.compile(model)  # [speedup] useless for mps and gives error too
parameter_size = sum(p.numel() for p in model.parameters())
print(f'{model_size} model with {parameter_size} parameters on device {device}')



# Learning rate scheduler
max_lr = 6e-4
min_lr = max_lr * 0.1
max_steps = n_epoch
warmup_steps = int(max_steps * 0.1)
def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it+1)/warmup_steps
    elif it > max_steps:
        return min_lr
    else:
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

weight_decay = 1e-1
#optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(0.9,0.95), eps=1e-8)
optimizer = model.configure_optimizers(weight_decay, learning_rate = max_lr, betas=(0.9,0.95), device_type=device)


total_batch_size = batch_size * block_size # 524288   # = 2**19 so nice number. This number is the total number of tokens processed in a entire batch. We will split this up and do gradient accumulation to simulate doing this.
assert total_batch_size % (micro_batch_size * block_size) == 0, 'Make sure total batch size is divisible by micro batch size times block size'
grad_accum_steps = total_batch_size // (micro_batch_size * block_size)
print(f'Desired total batch size {total_batch_size} and grad accum steps {grad_accum_steps}')


#oldtime = time.perf_counter()
losses = []
lrs = []

# Training

for epoch in range(n_epoch):
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()
    t0 = time.perf_counter()

    # zero the gradients
    optimizer.zero_grad() 
    
    # # evaluate loss once in a while
    # if epoch % (n_epoch/10) == 0:
    #     testing_loss = estimate_loss()
    #     print(f"epoch {epoch}, train loss {testing_loss['train']}, val loss {testing_loss['val']}")

    # gradient accumulation
    loss_accum = 0.0
    cross_entropy_loss_accum = 0.0
    load_balance_loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        # sample a batch
        x, y = get_batch('train')
        # forward pass
        with torch.autocast(device_type=device, dtype=torch.bfloat16):  # [speedup] useless for mps
            logits, cross_entropy_loss, load_balance_loss = model(x,y)
            loss = cross_entropy_loss + load_balance_strength * load_balance_loss
                
        loss = loss / grad_accum_steps # this division is so that the gradients obtained by this way (grad accumulation) will match that of the entire big batch in one go.
        loss_accum += loss.item()
        cross_entropy_loss_accum += cross_entropy_loss.item()/grad_accum_steps
        load_balance_loss_accum  += load_balance_loss.item()/grad_accum_steps
        # backward pass to get new gradients
        loss.backward()
        
    losses.append(loss_accum)

    # gradient clipping
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # get learning rate
    for param_group in optimizer.param_groups:
        lr = get_lr(epoch)
        param_group["lr"] = lr
    lrs.append(lr)
    
    # update parameters 
    optimizer.step()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()
    t1 = time.perf_counter()
    dt = t1-t0
    tokens_per_sec = (micro_batch_size * block_size * grad_accum_steps) / dt

    # track
    print(f'step: {epoch} | loss: {loss_accum:.4f} | lr {lr:.4f} | grad norm: {norm:.4f} | dt: {dt*1000:.4f}ms | tok/sec: {tokens_per_sec:.4f} | cse loss: {cross_entropy_loss_accum:.4f} | load_loss: {load_balance_loss_accum:.4f}')



# Estimation of final train and val losses
testing_loss = estimate_loss(model)
print(
    f"Training finished at {len(losses)} epochs with Cross Entropy train loss {testing_loss['train']} and Cross Entropy val loss {testing_loss['val']}")

# Saving the trained model
torch.save(model.state_dict(), "model.pth")
print("Saved trained weights to model.pth")

# Plotting loss and learning rate
plt.figure()
plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True)

plt.figure()
plt.plot(lrs)
plt.xlabel("Epochs")
plt.ylabel("lrs")
plt.title("Learning rate")
plt.grid(True)

plt.show()
