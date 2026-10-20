"""Quick E0 probe run used while tuning (not part of the final pipeline)."""
import sys
import torch
torch.set_num_threads(4)
sys.path.insert(0, ".")
from cabinet.config import E0
from cabinet.ppo import train_selfplay
from cabinet.policy import PolicyNet  # noqa: F401
iters = int(sys.argv[1]) if len(sys.argv) > 1 else 150
tag = sys.argv[2] if len(sys.argv) > 2 else "e0_probe"
hp = eval(sys.argv[3]) if len(sys.argv) > 3 else {}
net, h = train_selfplay(E0, iters, seed=0, log_every=10, hp=hp)
torch.save(net.state_dict(), f"runs/{tag}.pt")
