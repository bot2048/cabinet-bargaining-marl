"""Quick E0 probe run used while tuning (not part of the final pipeline)."""
import sys
import torch
torch.set_num_threads(4)
sys.path.insert(0, ".")
from cabinet.config import E0
from cabinet.ppo import train_selfplay
iters = int(sys.argv[1]) if len(sys.argv) > 1 else 150
net, h = train_selfplay(E0, iters, seed=0, log_every=10)
torch.save(net.state_dict(), "runs/e0_probe.pt")
