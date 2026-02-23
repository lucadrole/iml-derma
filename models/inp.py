import torch.nn as nn
import torch
import torch.nn.functional as F

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
from models.modules import XYEncoder, LatentEncoder, Decoder, XEncoder
from models.utils import MultivariateNormalDiag

# Setup C: raw text -> CLIP text encoder -> average pool -> project [N, d]
class CLIPTextKnowledgeEncoder(nn.Module):
    def __init__(self, clip_model, tokenizer, d, freeze_clip=True):
        super().__init__()
        self.clip_model = clip_model
        self.tokenizer = tokenizer
        
        # Freeze CLIP model parameters
        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False
        
        self.proj = nn.Linear(512, d)

    def forward(self, descriptions):
        """
        descriptions: list of N strings
        Returns: [N, d]
        """
        # OpenCLIP tokenizer: just pass the list, returns [N, context_length] tensor
        tokens = self.tokenizer(descriptions).to(next(self.parameters()).device)
        
        # encode_text returns [N, 512] directly (not a dict with .last_hidden_state)
        text_embeds = self.clip_model.encode_text(tokens)  # [N, 512]
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
    def __init__(self, config, clip_model, tokenizer):
        '''
        Here we assume the clip model is a dual-encoder
        '''
        super().__init__()
        self.config = config
        self.n_ways = config.n_ways
        self.d = config.hidden_dim  # 512
        self.tokenizer = tokenizer

        # Frozen CLIP vision encoder
        self.clip_vision = clip_model.visual
        for param in self.clip_vision.parameters():
            param.requires_grad = False
        self.image_proj = nn.Linear(config.clip_dim, self.d)

        # Knowledge encoder (C - just use clip embedding)
        self.knowledge_encoder = CLIPTextKnowledgeEncoder(clip_model, tokenizer, self.d)


        # Aggregator: sum + 2-layer MLP
        self.aggregator = nn.Sequential(
            nn.Linear(self.d, self.d),
            nn.GELU(),
            nn.Linear(self.d, 2 * self.d),  # outputs mu and sigma
        )

        # Decoder: maps z to class weight vectors
        self.decoder = nn.Sequential(
            nn.Linear(self.d, self.d),
            nn.GELU(),
            nn.Linear(self.d, self.d),
        )

        self.train_num_z_samples = config.train_num_z_samples
        self.test_num_z_samples = config.test_num_z_samples

    def encode_images(self, images):
        """images: [bs, num_images, C, H, W] or precomputed CLIP features"""
        with torch.no_grad():
            feats = self.clip_vision(images)  # [bs * num_images, clip_dim]
        return self.image_proj(feats)  # [bs * num_images, d]

    def encode_context_per_class(self, x_context, y_context):
        """
        Aggregate context image embeddings per class.
        x_context: [bs, k*N, d] image embeddings
        y_context: [bs, k*N] class labels (0..N-1)
        Returns: [bs, N, d]
        """
        bs = x_context.shape[0]
        r = torch.zeros(bs, self.n_ways, self.d, device=x_context.device)
        for c in range(self.n_ways):
            mask = (y_context == c).unsqueeze(-1)  # [bs, k*N, 1]
            count = mask.sum(dim=1).clamp(min=1)  # [bs, 1]
            r[:, c, :] = (x_context * mask).sum(dim=1) / count
        return r  # [bs, N, d]

    def infer_latent_dist(self, r, knowledge):
        """
        r: [bs, N, d] data representation
        knowledge: [bs, N, d] knowledge representation (or zeros)
        Returns: diagonal Gaussian over z of shape [bs, N, d]
        """
        combined = r + knowledge  # sum aggregation
        stats = self.aggregator(combined)  # [bs, N, 2*d]
        mu, scale_raw = stats.split(self.d, dim=-1)
        scale = 0.01 + 0.99 * F.softplus(scale_raw)
        return MultivariateNormalDiag(mu, scale)

    def decode(self, x_query, z_samples):
        """
        x_query: [bs, num_query, d] CLIP embeddings of query images
        z_samples: [n_samples, bs, N, d]
        Returns: logits [n_samples, bs, num_query, N]
        """
        # z_samples -> weight vectors per class
        W = self.decoder(z_samples)  # [n_samples, bs, N, d]
        # x_query: [bs, num_query, d] -> [1, bs, num_query, d]
        x_q = x_query.unsqueeze(0)
        # logits = -x_query @ W^T -> [n_samples, bs, num_query, N]
        logits = -torch.einsum('sbqd,sbnd->sbqn', x_q, W)
        return logits

    def forward(self, x_context, y_context, x_query, y_query=None,
                knowledge=None):
        # Encode context images per class
        r_C = self.encode_context_per_class(x_context, y_context)

        # Handle missing context (zero-shot)
        if x_context.shape[1] == 0:
            r_C = torch.zeros_like(r_C)

        # Knowledge embedding (zeros if not available)
        if knowledge is not None:
            k = self.knowledge_encoder(knowledge)  # [N, d]
            # Class descriptions are shared across the batch → expand to [bs, N, d]
            if k.dim() == 2:
                k = k.unsqueeze(0).expand(r_C.shape[0], -1, -1)
        else:
            k = torch.zeros_like(r_C)

        # Latent distribution from context + knowledge
        q_zCc = self.infer_latent_dist(r_C, k)

        # During training, also condition on targets
        if y_query is not None and self.training:
            r_T = self.encode_context_per_class(x_query, y_query)
            q_zCct = self.infer_latent_dist(r_T, k)
            sampling_dist = q_zCct
        else:
            q_zCct = None
            sampling_dist = q_zCc

        # Sample z
        n_samples = self.train_num_z_samples if self.training \
            else self.test_num_z_samples
        z_samples = sampling_dist.rsample([n_samples])

        # Decode to class logits
        logits = self.decode(x_query, z_samples)

        return logits, z_samples, q_zCc, q_zCct


if __name__ == "__main__":
    from argparse import Namespace
    from loss import ELBOLoss
    from dataset.utils import get_dataloader
    #from dataset.datasets import SetKnowledgeTrendingSinusoids
    import numpy as np
    import random
    import json

    # ========== Test CLIPTextKnowledgeEncoder ==========
    print("=" * 60)
    print("Testing CLIPTextKnowledgeEncoder with BiomedCLIP")
    print("=" * 60)
    
    from open_clip import create_model_and_transforms, get_tokenizer
    from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS
    
    # Load BiomedCLIP from local files (same as lab notebook)
    LOCAL_DIR = "/home/ldrole/my_space/work/cam_phd/checkpoints/biomedclip"
    MODEL_NAME = "biomedclip_local"
    
    with open(f"{LOCAL_DIR}/open_clip_config.json", "r") as f:
        clip_config = json.load(f)
        model_cfg = clip_config["model_cfg"]
        preprocess_cfg = clip_config["preprocess_cfg"]
    
    # Register the model config
    if (not MODEL_NAME.startswith(HF_HUB_PREFIX)
        and MODEL_NAME not in _MODEL_CONFIGS
        and clip_config is not None):
        _MODEL_CONFIGS[MODEL_NAME] = model_cfg
    
    tokenizer = get_tokenizer(MODEL_NAME)
    clip_model, _, preprocess = create_model_and_transforms(
        model_name=MODEL_NAME,
        pretrained=f"{LOCAL_DIR}/open_clip_pytorch_model.bin",
        **{f"image_{k}": v for k, v in preprocess_cfg.items()},
    )
    
    clip_model.eval()
    
    # Create knowledge encoder (project 512 -> 128)
    knowledge_encoder = CLIPTextKnowledgeEncoder(clip_model, tokenizer, d=128)
    knowledge_encoder.eval()
    
    # Verify CLIP model is frozen
    print("\nVerifying frozen parameters:")
    clip_params_frozen = 0
    clip_params_total = 0
    trainable_params = []
    
    for name, param in knowledge_encoder.named_parameters():
        if 'clip_model' in name:
            clip_params_total += 1
            if not param.requires_grad:
                clip_params_frozen += 1
        elif param.requires_grad:
            trainable_params.append(name)
    
    print(f"  CLIP params frozen: {clip_params_frozen}/{clip_params_total}")
    print(f"  Trainable params: {trainable_params}")
    assert clip_params_frozen == clip_params_total, "CLIP model not fully frozen!"
    print("✓ CLIP model successfully frozen\n")
    
    # Test with sample medical descriptions
    descriptions = [
        "A dermatoscopy image of melanoma",
        "A dermatoscopy image of basal cell carcinoma",
        "A dermatoscopy image of benign nevus",
    ]
    
    print(f"\nInput: {len(descriptions)} descriptions")
    with torch.no_grad():
        knowledge_embeds = knowledge_encoder(descriptions)
    
    print(f"Output shape: {knowledge_embeds.shape}")
    print(f"Expected: [{len(descriptions)}, 128]")
    assert knowledge_embeds.shape == (len(descriptions), 128), "Shape mismatch!"
    print("✓ CLIPTextKnowledgeEncoder test passed!\n")
    
    # ========== Test INP_MedClassification forward pass ==========
    print("=" * 60)
    print("Testing INP_MedClassification forward pass")
    print("=" * 60)
    
    med_config = Namespace(
        n_ways=3,
        hidden_dim=128,
        clip_dim=512,
        train_num_z_samples=1,
        test_num_z_samples=4,
    )
    
    med_model = INP_MedClassification(med_config, clip_model, tokenizer)
    med_model.train()
    
    bs, k_shot, n_ways, d = 2, 5, 3, 128
    x_context = torch.randn(bs, k_shot * n_ways, d)
    y_context = torch.cat([torch.full((bs, k_shot), c) for c in range(n_ways)], dim=1).long()
    x_query = torch.randn(bs, 10, d) # 10 encoded query images
    y_query = torch.randint(0, n_ways, (bs, 10))
    
    knowledge = ["melanoma", "basal cell carcinoma", "benign nevus"]
    
    logits, z_samples, q_zCc, q_zCct = med_model(x_context, y_context, x_query, y_query, knowledge)
    
    
    print(f"  logits     : {logits.shape}    (n_z_samples, bs, num_query, n_ways)") # numquery: number of images
    print(f"  z_samples  : {z_samples.shape}")
    print(f"  q(z|C) loc : {q_zCc.mean.shape}")
    print(f"  q(z|C,T)   : {q_zCct.mean.shape}")
    assert logits.shape == (1, bs, 10, n_ways), f"Expected (1, {bs}, 10, {n_ways}), got {logits.shape}"
    print("✓ INP_MedClassification forward pass test passed!\n")
    
    # ========== Test INP (original) ==========
    # print("=" * 60)
    # print("Testing INP model")
    # print("=" * 60)
    
    # config = Namespace(
    #     # model
    #     input_dim=1,
    #     output_dim=1,
    #     xy_encoder_num_hidden=2,
    #     xy_encoder_hidden_dim=128,
    #     data_agg_func="mean",
    #     latent_encoder_num_hidden=2,
    #     decoder_hidden_dim=64,
    #     decoder_num_hidden=2,
    #     decoder_activation="gelu",
    #     hidden_dim=128,
    #     x_transf_dim=128,
    #     x_encoder_num_hidden=1,
    #     test_num_z_samples=32,
    #     train_num_z_samples=1,
    #     knowledge_extractor_num_hidden=0,
    #     knowledge_dropout=0,
    #     knowledge_dim=128,
    #     knowledge_merge="sum",
    #     text_encoder="set",
    #     use_knowledge=True,
    #     freeze_llm=True,
    #     tune_llm_layer_norms=False,
    #     # dataset
    #     batch_size=64,
    #     min_num_context=1,
    #     max_num_context=30,
    #     x_sampler="uniform",
    #     noise=0,
    #     # reproducibility
    #     seed=44,
    #     dataset="set-trending-sinusoids",
    #     num_targets=50,
    # )
    # config.device = "cpu"

    # dataset = SetKnowledgeTrendingSinusoids(split="train", knowledge_type="abc2")
    # train_dataloader = get_dataloader(dataset, config)
    # config.knowledge_input_dim = dataset.knowledge_input_dim

    # model = INP(config)
    # loss_func = ELBOLoss()

    # torch.manual_seed(config.seed)
    # np.random.seed(config.seed)
    # random.seed(config.seed)

    # for i, batch in enumerate(train_dataloader):
    #     print(i)
    #     context, target, knowledge, _ = batch
    #     x_context, y_context = context
    #     x_target, y_target = target

    #     if config.use_knowledge:
    #         outputs = model(x_context, y_context, x_target, y_target, knowledge)
    #     else:
    #         outputs = model(x_context, y_context, x_target, y_target, None)

    #     print(y_target.shape)
    #     p_yCc = outputs[0]
    #     print(p_yCc.mean.shape)

    #     loss = loss_func(outputs, y_target)

    #     print(loss)
