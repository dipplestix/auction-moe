import torch


def selector(s):
    # s: (k, m)
    m = s.size(1)
    device = s.device
    patterns = torch.tensor(
        [list(map(int, format(i, f'0{m}b'))) for i in range(2**m)],
        dtype=s.dtype, device=device
    )  # shape: (2^m, m)

    eps = 1e-6
    s_clamped = s.clamp(min=eps, max=1 - eps).unsqueeze(1)  # (k, 1, m)
    log_s = torch.log(s_clamped)
    log_1_minus_s = torch.log(1 - s_clamped)

    log_probs = patterns * log_s + (1 - patterns) * log_1_minus_s  # (k, 2^m, m)
    return torch.exp(log_probs.sum(dim=-1))  # (k, 2^m)


def per_example_selector(s):
    # s: (B, k, m)
    B, k, m = s.shape
    device = s.device
    patterns = torch.tensor(
        [list(map(int, format(i, f'0{m}b'))) for i in range(2**m)],
        dtype=s.dtype, device=device
    )  # (2^m, m)

    eps = 1e-6
    s_clamped = s.clamp(min=eps, max=1 - eps).unsqueeze(2)  # (B, k, 1, m)
    patterns = patterns.view(1, 1, 2**m, m)                 # (1, 1, 2^m, m)

    log_s = torch.log(s_clamped)                           # (B, k, 1, m)
    log_1_minus_s = torch.log(1 - s_clamped)

    log_probs = patterns * log_s + (1 - patterns) * log_1_minus_s  # (B, k, 2^m, m)
    return torch.exp(log_probs.sum(dim=-1))                          # (B, k, 2^m)
