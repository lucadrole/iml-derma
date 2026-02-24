import torch.nn as nn
import torch
import torch.nn.functional as F

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
from models.modules import XYEncoder, LatentEncoder, Decoder, XEncoder
from models.utils import MultivariateNormalDiag

class CLIPTextKnowledgeEncoder(nn.Module):
    """
    Frozen text encoder + trainable linear projection.

    Supports two backends via encoder_type:
      - "clip":       OpenAI CLIP ViT-B/32 text tower (77-token, custom transformer)
      - "biomedclip": BiomedCLIP PubMedBERT text tower (256-token, HF BERT-based)

    Both use average pooling over last-layer hidden states, matching the paper
    (Appendix A.6.3, Setup C):
      "The embeddings of class descriptions are obtained as the average
       of all outputs from the last layer of CLIP."
    """

    def __init__(self, clip_model, tokenizer, d, encoder_type="clip",
                 freeze_clip=True, embed_dim=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.encoder_type = encoder_type

        if freeze_clip:
            for param in clip_model.parameters():
                param.requires_grad = False

        # List wrapper prevents nn.Module registration (keeps checkpoint small)
        # but also hides it from .to() — we handle device manually in forward().
        self._clip_ref = [clip_model]

        self.proj = nn.Linear(embed_dim, d)

    def _encode_text_avg_pool_clip(self, tokens, clip):
        """Average-pool all token outputs from CLIP's custom transformer."""
        x = clip.token_embedding(tokens)        # [N, seq_len, width]
        x = x + clip.positional_embedding       # broadcast add
        x = x.permute(1, 0, 2)                  # [seq_len, N, width]
        x = clip.transformer(x)
        x = x.permute(1, 0, 2)                  # [N, seq_len, width]
        x = clip.ln_final(x)                    # [N, seq_len, 512]
        return x.mean(dim=1)                     # [N, 512]

    def _encode_text_avg_pool_biomedclip(self, tokens, clip):
        """
        Average-pool last-layer hidden states from BiomedCLIP's PubMedBERT.

        open_clip wraps BiomedCLIP's text encoder in an HFTextEncoder.
        The underlying HF BERT model is at clip.text.transformer.
        We run it directly to get the full hidden state for avg pooling.
        """
        text_encoder = clip.text  # open_clip HFTextEncoder wrapper

        # Attention mask: 1 for real tokens, 0 for padding
        attn_mask = (tokens != 0).long()

        # Forward through the HF BERT model
        outputs = text_encoder.transformer(
            input_ids=tokens,
            attention_mask=attn_mask,
        )
        last_hidden = outputs.last_hidden_state  # [N, seq_len, 768]

        # Average pool over non-padding positions
        mask = attn_mask.unsqueeze(-1).float()   # [N, seq_len, 1]
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        # pooled: [N, 768]

        # Apply text projection (768 -> 512) if present
        # In BiomedCLIP, text_encoder.proj can be nn.Sequential or nn.Parameter
        if hasattr(text_encoder, 'proj') and text_encoder.proj is not None:
            if isinstance(text_encoder.proj, nn.Module):
                pooled = text_encoder.proj(pooled)   # nn.Sequential or nn.Linear
            else:
                pooled = pooled @ text_encoder.proj   # raw weight matrix

        return pooled

    def forward(self, descriptions):
        device = self.proj.weight.device
        clip = self._clip_ref[0].to(device)

        if self.encoder_type == "biomedclip":
            tokens = self.tokenizer(descriptions, context_length=256).to(device)
            with torch.no_grad():
                text_embeds = self._encode_text_avg_pool_biomedclip(tokens, clip)
        else:
            tokens = self.tokenizer(descriptions).to(device)
            with torch.no_grad():
                text_embeds = self._encode_text_avg_pool_clip(tokens, clip)

        k = self.proj(text_embeds)  # [N, d]
        return k
class INP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.xy_encoder = XYEncoder(config)
        self.latent_encoder = LatentEncoder(config)
        self.decoder = Decoder(config)
        self.x_encoder = XEncoder(config)
        self.train_num_z_samples = config.train_num_z_samples
        self.test_num_z_samples = config.test_num_z_samples

    def forward(self, x_context, y_context, x_target, y_target, knowledge=None):
        x_context = self.x_encoder(x_context)  # [bs, num_context, x_transf_dim]
        x_target = self.x_encoder(x_target)  # [bs, num_context, x_transf_dim]

        R = self.encode_globally(x_context, y_context, x_target)

        z_samples, q_z_Cc, q_zCct = self.sample_latent(
            R, x_context, x_target, y_target, knowledge
        )
        # reshape z_samples to the shape of x_target
        R_target = self.target_dependent_representation(R, x_target, z_samples)

        p_yCc = self.decode_target(x_target, R_target)

        return p_yCc, z_samples, q_z_Cc, q_zCct

    def encode_globally(self, x_context, y_context, x_target):
        """
        Encode context set all together
        """
        R = self.xy_encoder(x_context, y_context, x_target)

        if x_context.shape[1] == 0:
            R = torch.zeros((R.shape[0], 1, R.shape[-1])).to(R.device)

        return R

    def get_knowledge_embedding(self, knowledge):
        return self.latent_encoder.get_knowledge_embedding(knowledge)

    def sample_latent(self, R, x_context, x_target, y_target, knowledge):
        """
        Sample latent variable z given the global representation
        (and during training given the target)
        """
        q_zCc = self.infer_latent_dist(R, knowledge, x_context.shape[1])

        if y_target is not None and self.training:
            R_from_target = self.encode_globally(x_target, y_target, x_target)
            q_zCct = self.infer_latent_dist(R_from_target, knowledge, x_target.shape[1])
            sampling_dist = q_zCct
        else:
            q_zCct = None
            sampling_dist = q_zCc

        if self.training:
            z_samples = sampling_dist.rsample([self.train_num_z_samples])
        else:
            z_samples = sampling_dist.rsample([self.test_num_z_samples])
        # z_samples.shape = [n_z_samples, bs, 1, z_dim]
        return z_samples, q_zCc, q_zCct

    def infer_latent_dist(self, R, knowledge, n):
        """
        Infer the latent distribution given the global representation
        """
        q_z_stats = self.latent_encoder(R, knowledge, n)
        q_z_loc, q_z_scale = q_z_stats.split(self.config.hidden_dim, dim=-1)
        q_z_scale = 0.01 + 0.99 * F.softplus(q_z_scale)
        q_zCc = MultivariateNormalDiag(q_z_loc, q_z_scale)
        return q_zCc

    def target_dependent_representation(self, R, x_target, z_samples):
        """
        Compute the target dependent representation of the context set
        """
        R_target = z_samples  # [num_z_samples, batch_size, 1, hidden_dim]

        # [num_z_samples, batch_size, num_targets, hidden_dim]

        R_target = R_target.expand(-1, -1, x_target.shape[1], -1)

        return R_target

    def decode_target(self, x_target, R_target):
        """
        Decode the target set given the target dependent representation
        """
        p_y_stats = self.decoder(x_target, R_target)

        p_y_loc, p_y_scale = p_y_stats.split(self.config.output_dim, dim=-1)

        # bound the variance (minimum 0.1)
        p_y_scale = 0.1 + 0.9 * F.softplus(p_y_scale)

        p_yCc = MultivariateNormalDiag(p_y_loc, p_y_scale)

        return p_yCc



class INP_MedClassification(nn.Module):
    """
    Informed Neural Process for N-way, k-shot image classification.

    Expects pre-computed, L2-normalised CLIP/BiomedCLIP image embeddings
    as input (not raw images). Raw-image encoding is handled offline by
    dataset/isic.py::build_embedding_cache().

    Forward inputs
    --------------
    x_context : Tensor[bs, k*N, clip_dim]   context image embeddings
    y_context : Tensor[bs, k*N]             episode-local class labels 0..N-1
                                            (zero-padded when k=0)
    x_query   : Tensor[bs, Q*N, clip_dim]   query image embeddings
    y_query   : Tensor[bs, Q*N] | None      labels (None at eval time)
    knowledge : list[str] of length N | None  class descriptions for this episode

    Forward outputs
    ---------------
    logits    : Tensor[n_z, bs, Q*N, N]
    z_samples : Tensor[n_z, bs, N, d]
    q_zCc     : MultivariateNormalDiag  posterior given context only
    q_zCct    : MultivariateNormalDiag | None  posterior given context+target (train only)
    """

    def __init__(self, config, clip_model, tokenizer):
        super().__init__()
        self.n_ways = config.n_ways
        self.d      = config.hidden_dim   # latent dimension

        # ── Image projection (only trainable image-side component) ────────────
        # clip_vision is NOT stored here — we use pre-cached embeddings.
        # image_proj maps frozen CLIP embeddings -> trainable hidden space.
        self.image_proj = nn.Linear(config.clip_dim, self.d)

        # ── Knowledge encoder (Setup C) ───────────────────────────────────────
        # Frozen CLIP/BiomedCLIP text encoder + trainable linear projection.
        encoder_type = getattr(config, "encoder_type", "clip")
        self.knowledge_encoder = CLIPTextKnowledgeEncoder(
            clip_model, tokenizer, self.d,
            encoder_type=encoder_type,
            embed_dim=config.clip_dim,
        )

        # ── Aggregator: (r + k) -> (mu_z, sigma_z) ───────────────────────────
        # Input:  [bs, N, d]   (sum of data rep and knowledge rep)
        # Output: [bs, N, 2d]  (mean and pre-softplus scale of diagonal Gaussian)
        self.aggregator = nn.Sequential(
            nn.Linear(self.d, self.d),
            nn.GELU(),
            nn.Linear(self.d, 2 * self.d),
        )

        # ── Decoder: z -> class weight vectors ───────────────────────────────
        # Input:  [n_z, bs, N, d]
        # Output: [n_z, bs, N, d]  (one weight vector per class)
        self.decoder = nn.Sequential(
            nn.Linear(self.d, self.d),
            nn.GELU(),
            nn.Linear(self.d, self.d),
        )

        self.train_num_z_samples = config.train_num_z_samples
        self.test_num_z_samples  = config.test_num_z_samples

    # ── Sub-components ────────────────────────────────────────────────────────

    def encode_context_per_class(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Mean-pool projected embeddings per class.

        x : [bs, M, d]   M = k*N (context) or Q*N (query)
        y : [bs, M]      episode-local labels 0..N-1

        Returns r : [bs, N, d]

        """
        bs = x.shape[0]
        r  = torch.zeros(bs, self.n_ways, self.d, device=x.device, dtype=x.dtype)
        for c in range(self.n_ways):
            mask = (y == c).unsqueeze(-1).float()
            r[:, c, :] = (x * mask).sum(dim=1)      # sum, matching paper
        return r

    def infer_latent_dist(
        self, r: torch.Tensor, k: torch.Tensor
    ) -> "MultivariateNormalDiag":
        """
        r : [bs, N, d]  data representation
        k : [bs, N, d]  knowledge representation (zeros if masked)

        Returns diagonal Gaussian q(z) with mean and scale [bs, N, d].
        """
        combined  = F.relu(r + k)            # relu matches original LatentEncoder
        stats     = self.aggregator(combined) # [bs, N, 2d]
        mu, scale_raw = stats.split(self.d, dim=-1)
        scale = 0.01 + 0.99 * F.softplus(scale_raw)
        return MultivariateNormalDiag(mu, scale)

    def decode(self, x_query: torch.Tensor, z_samples: torch.Tensor) -> torch.Tensor:
        """
        Compute class logits via inner product between query embeddings
        and decoded class weight vectors.

        x_query   : [bs, Q*N, d]
        z_samples : [n_z, bs, N, d]

        Returns logits : [n_z, bs, Q*N, N]
        """
        W   = self.decoder(z_samples)              # [n_z, bs, N, d]
        x_q = x_query.unsqueeze(0)                 # [1,   bs, Q*N, d]
        # positive inner product: higher similarity -> higher logit
        logits = torch.einsum('sbqd,sbnd->sbqn', x_q, W)
        return logits

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        x_context: torch.Tensor,
        y_context: torch.Tensor,
        x_query:   torch.Tensor,
        y_query:   torch.Tensor = None,
        knowledge: list         = None,
    ):
        bs = x_context.shape[0]
        # ── 1. Project CLIP embeddings into hidden space ──────────────────────
        # This is the only trainable transformation on the image side.
        # Must be applied before any aggregation or decoding.
        x_context = self.image_proj(x_context)   # [bs, k*N, d]
        x_query   = self.image_proj(x_query)     # [bs, Q*N, d]

        # ── 2. Per-class aggregation of context embeddings ────────────────────
        r_C = self.encode_context_per_class(x_context, y_context)  # [bs, N, d]

        # ── 3. Knowledge embedding ────────────────────────────────────────────
        if knowledge is not None:
            flat_descs = [desc for ep_descs in knowledge for desc in ep_descs]
            flat_k = self.knowledge_encoder(flat_descs)    # [bs*N, d]
            k = flat_k.view(bs, self.n_ways, self.d)       # [bs, N, d]
        else:
            k = torch.zeros_like(r_C)

        # ── 4. Latent posterior given context (+ knowledge) ───────────────────
        q_zCc = self.infer_latent_dist(r_C, k)

        # ── 5. During training: latent posterior given context + target ───────
        if y_query is not None and self.training:
            r_T    = self.encode_context_per_class(x_query, y_query)  # [bs, N, d]
            q_zCct = self.infer_latent_dist(r_T, k)
            sampling_dist = q_zCct
        else:
            q_zCct        = None
            sampling_dist = q_zCc

        # ── 6. Sample latent variable ─────────────────────────────────────────
        n_samples = (
            self.train_num_z_samples if self.training
            else self.test_num_z_samples
        )
        z_samples = sampling_dist.rsample([n_samples])  # [n_z, bs, N, d]

        # ── 7. Decode to class logits ─────────────────────────────────────────
        logits = self.decode(x_query, z_samples)        # [n_z, bs, Q*N, N]

        return logits, z_samples, q_zCc, q_zCct

