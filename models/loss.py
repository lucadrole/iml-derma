import torch
import torch.nn as nn
import torch.nn.functional as F

import math


def sum_log_prob(prob, samples):
    """
    Compute the sum of log probability of samples under the given distribution.
    Used for continuous (Gaussian) outputs and for latent distributions.
    """
    # size = [n_z_samples, batch_size, *]
    log_prob = prob.log_prob(samples)
    log_prob = torch.sum(log_prob, dim=2)
    return log_prob

def sum_log_prob_categorical(logits, targets):
    """
    Compute the sum of log probability for categorical (classification) outputs.
    
    logits:  [n_z_samples, batch_size, num_query, N_ways]
    targets: [batch_size, num_query]  (class indices)
    
    Returns: [n_z_samples, batch_size]
    """
    n_z, bs, num_q, n_ways = logits.shape

    # Expand targets to match z samples: [n_z_samples, bs, num_query]
    y_expanded = targets.unsqueeze(0).expand(n_z, -1, -1)

    # Cross-entropy gives -log p(y|x,z); negate to get log p(y|x,z)
    log_p = -F.cross_entropy(
        logits.reshape(-1, n_ways),     # [n_z * bs * num_q, N_ways]
        y_expanded.reshape(-1),         # [n_z * bs * num_q]
        reduction="none",
    ).reshape(n_z, bs, num_q)           # [n_z, bs, num_q]

    # Sum over query points
    return log_p.sum(dim=2)             # [n_z, bs]


class Loss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred_outputs, y_target):
        """
        Compute the loss
        """
        p_yCc, z_samples, q_zCc, q_zCct = pred_outputs

        loss, kl_z, negative_ll = self.get_loss(
            p_yCc, z_samples, q_zCc, q_zCct, y_target
        )

        if self.reduction is None:
            return loss, kl_z, negative_ll
        elif self.reduction == "mean":
            return torch.mean(loss), torch.mean(kl_z), torch.mean(negative_ll)
        elif self.reduction == "sum":
            return torch.sum(loss), torch.sum(kl_z), torch.sum(negative_ll)
        else:
            raise (NotImplementedError)


class ELBOLoss(Loss):
    """
    ELBO loss supporting both continuous (Gaussian) and categorical outputs.
    
    Set `categorical=True` for classification tasks where p_yCc is a logits
    tensor [n_z_samples, bs, num_query, N_ways] and y_target is class indices
    [bs, num_query].
    """

    def __init__(self, reduction="mean", beta=1, categorical=False):
        super().__init__(reduction=reduction)
        self.beta = beta
        self.categorical = categorical

    def _log_likelihood(self, p_yCc, y_target):
        """Dispatch to the appropriate log-likelihood computation."""
        if self.categorical:
            return sum_log_prob_categorical(p_yCc, y_target)
        else:
            return sum_log_prob(p_yCc, y_target)

    def get_loss(self, p_yCc, z_samples, q_zCc, q_zCct, y_target):
        """
        Compute the ELBO loss during training and NLL for evaluation.
        """
        if q_zCct is not None:
            # ---- Training: ELBO ----
            # 1st term: E_{q(z|T)}[log p(y_t | z)]
            sum_log_p_yCz = self._log_likelihood(p_yCc, y_target)  # [n_z, bs]
            E_z_sum_log_p_yCz = torch.mean(sum_log_p_yCz, dim=0)  # [bs]

            # 2nd term: KL[q(z|C,T) || q(z|C)]
            kl_z = torch.distributions.kl.kl_divergence(
                q_zCct, q_zCc
            )  # [bs, *n_lat]
            E_z_kl = torch.sum(kl_z, dim=list(range(1, kl_z.dim())))  # [bs]

            loss = -(E_z_sum_log_p_yCz - self.beta * E_z_kl)
            negative_ll = -E_z_sum_log_p_yCz

        else:
            # ---- Eval: importance-weighted NLL ----
            sum_log_p_yCz = self._log_likelihood(p_yCc, y_target)
            sum_log_w_k = sum_log_p_yCz
            log_S_z_sum_p_y_Cz = torch.logsumexp(sum_log_w_k, 0)
            log_E_z_sum_p_yCz = log_S_z_sum_p_y_Cz - math.log(
                sum_log_w_k.shape[0]
            )
            kl_z = torch.zeros_like(log_E_z_sum_p_yCz)
            negative_ll = -log_E_z_sum_p_yCz
            loss = negative_ll

        return loss, kl_z, negative_ll



class NLL(Loss):
    """
    Approximate negative log-likelihood via importance sampling.
    Supports both continuous (Gaussian) and categorical outputs.
    """

    def __init__(self, reduction="mean", categorical=False):
        super().__init__(reduction=reduction)
        self.categorical = categorical

    def _log_likelihood(self, p_yCc, y_target):
        if self.categorical:
            return sum_log_prob_categorical(p_yCc, y_target)
        else:
            return sum_log_prob(p_yCc, y_target)

    def get_loss(self, p_yCc, z_samples, q_zCc, q_zCct, y_target):
        sum_log_p_yCz = self._log_likelihood(p_yCc, y_target)

        # Importance sampling
        if q_zCct is not None:
            sum_log_q_zCc = sum_log_prob(q_zCc, z_samples)
            sum_log_q_zCct = sum_log_prob(q_zCct, z_samples)
            sum_log_w_k = sum_log_p_yCz + sum_log_q_zCc - sum_log_q_zCct
        else:
            sum_log_w_k = sum_log_p_yCz

        log_s_z_sum_p_yCz = torch.logsumexp(sum_log_w_k, 0)
        log_E_z_sum_p_yCz = log_s_z_sum_p_yCz - math.log(
            sum_log_w_k.shape[0]
        )

        return (
            -log_E_z_sum_p_yCz,
            torch.zeros_like(log_E_z_sum_p_yCz),
            -log_E_z_sum_p_yCz,
        )