import torch
import torch.nn as nn
from functools import partial
from torch.nn.init import trunc_normal_
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence


class DiVT(nn.Module):
    def __init__(
            self,
            embed_dim=1024,
            num_heads=1024//128,
            clip_dim=1024,
            hidden_size=4096,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            threshold=0.65,
            num_patch=576
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.threshold = threshold
        self.num_patch = num_patch

        self.positional_embedding = nn.Parameter(torch.randn(num_patch, embed_dim) / embed_dim ** 0.5)

        self.q_proj_1 = nn.Linear(clip_dim, embed_dim, bias=False)

        k_modules = [nn.Linear(clip_dim, embed_dim)]
        for _ in range(1,2):
            k_modules.append(nn.GELU())
            k_modules.append(nn.Linear(embed_dim, embed_dim))
        self.k_proj_1 = nn.Sequential(*k_modules)

        v_modules = [nn.Linear(clip_dim, embed_dim)]
        for _ in range(1,2):
            v_modules.append(nn.GELU())
            v_modules.append(nn.Linear(embed_dim, embed_dim))
        self.v_proj_1 = nn.Sequential(*v_modules)

        self.ln_q_1 = norm_layer(embed_dim)
        self.ln_k_1 = norm_layer(embed_dim)
        self.ln_v_1 = norm_layer(embed_dim)

        self.clip_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        modules = [nn.Linear(embed_dim, hidden_size)]
        for _ in range(1, 2):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_size, hidden_size))
        self.mlp = nn.Sequential(*modules)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def clustering(self, query_feature, clustering_feature, threshold):
        """
        Cluster visual tokens based on cosine similarity and select representative query tokens.

        Args:
            query_feature: (B, N, D) features used to extract query tokens
            clustering_feature: (B, N, D) features used for clustering
            threshold: cosine similarity threshold

        Returns:
            query_tokens: (B, Q, D)
            num_queries: (B,)
            attn_mask: attention mask for query-token attention
        """

        # cosine similarity between tokens
        emb_norm = F.normalize(clustering_feature, p=2, dim=-1)
        sim_map = torch.bmm(emb_norm, emb_norm.transpose(1,2))
        sim_map_original = sim_map.clone()

        # number of neighbors above the threshold
        threshold_score = torch.sum((sim_map > threshold), dim=2)

        image_num, token_num, _ = sim_map.shape

        cluster_index = torch.zeros(image_num, token_num, dtype=torch.long, device=sim_map.device)
        cluster_num = torch.ones(image_num, dtype=torch.long, device=sim_map.device)

        query_index = [[] for _ in range(image_num)]
        attn_mask = [[] for _ in range(image_num)]

        min_score = -1e4

        # Initial Patch Clustering
        while True:
            mask_all_done = torch.all(cluster_index != 0, dim=1)
            if mask_all_done.all():
                break

            active_batches = (~mask_all_done).nonzero(as_tuple=True)[0]
            if len(active_batches) == 0:
                break

            query_patch_index = torch.argmax(threshold_score[active_batches], dim=1)

            b_idx = active_batches
            q_idx = query_patch_index

            sim_selected = sim_map[b_idx, q_idx, :]
            query_cluster = sim_selected > threshold

            cluster_index[b_idx] += cluster_num[b_idx].unsqueeze(1) * query_cluster

            sim_map[b_idx] = sim_map[b_idx].masked_fill(query_cluster.unsqueeze(1), min_score)
            threshold_score[b_idx] = threshold_score[b_idx].masked_fill(query_cluster, min_score)

            cluster_num[b_idx] += 1

            for i, b in enumerate(b_idx):
                query_index[b].append(q_idx[i].item())

        # Cluster Refinement
        for b in range(image_num):
            qindex = torch.tensor(query_index[b], device=query_cluster.device)
            qindex = torch.sort(qindex).values

            qsim = sim_map_original[b][qindex]

            cluster_index[b] = torch.argmax(qsim, dim=0)

            attn_mask[b] = torch.ones_like(qsim, dtype=torch.bool)
            attn_mask[b][cluster_index[b], torch.arange(qsim.shape[1])] = False

            query_index[b] = qindex

        query_index_tensor = pad_sequence(query_index, batch_first=True, padding_value=0)
        query_tensor = torch.gather(query_feature, 1,query_index_tensor.unsqueeze(-1).expand(-1, -1, query_feature.shape[-1]))
        attn_mask_tensor = pad_sequence(attn_mask, batch_first=True, padding_value=False)

        return query_tensor, cluster_num-1, attn_mask_tensor

    def forward(self, visual_feature, threshold=None):
        if threshold is None:
            threshold = self.threshold

        query_tokens, num_queries, attn_mask = self.clustering(
            visual_feature,
            visual_feature,
            threshold
        )

        attn_mask = attn_mask.repeat_interleave(self.num_heads, dim=0)
        
        query_features = self.ln_q_1(self.q_proj_1(query_tokens))
        key_features = self.ln_k_1(self.k_proj_1(visual_feature))
        value_features = self.ln_v_1(
            self.v_proj_1(visual_feature + self.positional_embedding)
        )
        
        out = self.clip_attn(
            query_features,
            key_features,
            value_features,
            attn_mask=attn_mask
        )[0]

        x = self.mlp(out)

        return x, num_queries

    def select_valid_visual_tokens(self, visual_embeddings, num_valid_tokens):
        return visual_embeddings[:num_valid_tokens]

def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'clusterer')
    if projector_type == 'clusterer':
        return DiVT(
            hidden_size=config.hidden_size,
            threshold=config.threshold
            )
    raise ValueError(f'Unknown projector type: {projector_type}')
