import torch
import torch.nn as nn
import torch.nn.functional as F


class ContinuousPolicyNetworkMasked(nn.Module):
    """
    Policy π(a | (z_1, m_1), ..., (z_t, m_t))
    Observations include a mask bit to distinguish "no observation".
    """

    def __init__(self, action_size, hidden_dim=64):
        super().__init__()

        # LSTM input is 2-D: [z, m]
        self.lstm = nn.LSTM(input_size=2,
                            hidden_size=hidden_dim,
                            batch_first=True)

        self.action_head = nn.Linear(hidden_dim, action_size)

    def forward(self, obs_seq):
        """
        obs_seq: (batch, T, 2) float tensor
        """
        _, (hidden, _) = self.lstm(obs_seq)
        last_hidden = hidden[-1]

        logits = self.action_head(last_hidden)
        probs = F.softmax(logits, dim=-1)
        return probs, logits
