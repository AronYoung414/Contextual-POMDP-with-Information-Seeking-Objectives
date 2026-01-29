import torch
import pickle
from light_dark_environment import ContinuousLightDarkPOMDP
from pg_pomdp_solver import train_pg_pomdp

env = ContinuousLightDarkPOMDP()

policy, value_net, train_returns = train_pg_pomdp(
    env,
    horizon=20,
    num_episodes=10000,
    lr=3e-4,
    hidden_dim=64,
    seed=0,
    device="cpu",
    log_every=50,
)

torch.save(policy.state_dict(), "data_pomdp_pg/pg_policy.pkl")
torch.save(value_net.state_dict(), "data_pomdp_pg/pg_value.pkl")

with open("data_pomdp_pg/pg_train_returns.pkl", "wb") as f:
    pickle.dump({"train_returns": train_returns.tolist()}, f)

print("Saved pg_policy.pkl, pg_value.pkl, pg_train_returns.pkl")
