"""
Category Token Conditioning for Garment Mesh Generation.
Prepends a learned category embedding to the BPT token sequence,
enabling category-aware generation (e.g., 'Dress' vs 'Tops').
"""
import json
import torch
import torch.nn as nn
from pathlib import Path


class CategoryMapper:
    """Maps sample names to category IDs using the pre-built mapping."""

    def __init__(self, mapping_path="category_mapping.json"):
        with open(mapping_path, "r") as f:
            data = json.load(f)
        self.categories = data["categories"]
        self.num_categories = data["num_categories"]
        self.sample_to_category = data["sample_to_category"]
        self.unknown_id = self.num_categories  # "unknown" category

    def get_category_id(self, sample_name):
        return self.sample_to_category.get(sample_name, self.unknown_id)

    def get_category_name(self, cat_id):
        if cat_id < self.num_categories:
            return self.categories[cat_id]
        return "Unknown"


class CategoryEmbedding(nn.Module):
    """Learned category embedding prepended to token sequence."""

    def __init__(self, num_categories, embed_dim):
        super().__init__()
        # +1 for "unknown" category
        self.embedding = nn.Embedding(num_categories + 1, embed_dim)

    def forward(self, category_ids):
        """
        Args:
            category_ids: [B] tensor of category indices
        Returns:
            [B, 1, embed_dim] category embeddings
        """
        return self.embedding(category_ids).unsqueeze(1)


def load_category_conditioner(embed_dim, mapping_path="category_mapping.json"):
    """
    Factory function: loads category mapping and creates embedding.
    Returns (CategoryMapper, CategoryEmbedding).
    """
    mapper = CategoryMapper(mapping_path)
    num_categories = mapper.num_categories  # 11 known + 1 unknown = 12
    embedding = CategoryEmbedding(num_categories, embed_dim)
    return mapper, embedding
