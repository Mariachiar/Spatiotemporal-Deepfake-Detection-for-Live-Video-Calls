import torch
import torch.nn.functional as F
from typing import Tuple

# Tip: Batch deve avere .labels (tensor [B])
#      self.training e self.config.slerp_feature_augmentation / _range devono esistere

def slerp(A: torch.Tensor, B: torch.Tensor, t: torch.Tensor | float) -> torch.Tensor:
    """
    Spherical linear interpolation between two batched points A and B on a unit hypersphere.

    Parameters:
    - A: First set of points, shape (batch_size, d).
    - B: Second set of points, shape (batch_size, d).
    - t: Interpolation parameter in range [0, 1], shape (batch_size, 1) or single value.

    Returns:
    - torch.Tensor: Interpolated points, shape (batch_size, d).
    """
    # Ensure inputs are unit vectors
    A = F.normalize(A, dim=-1)
    B = F.normalize(B, dim=-1)

    # Compute dot product for each pair of points
    dot = torch.sum(A * B, dim=-1, keepdim=True).clamp(-1 + 1e-7, 1 - 1e-7)  # Avoid numerical issues

    # Compute the angle for each pair
    theta = torch.acos(dot)

    # Slerp formula
    sin_theta = torch.sin(theta)
    t_theta = t * theta
    coeff_a = torch.sin(theta - t_theta) / sin_theta
    coeff_b = torch.sin(t_theta) / sin_theta

    # Compute the interpolated points
    interpolated = coeff_a * A + coeff_b * B

    return interpolated


def slerp_feature_augmentation(self, batch, features: torch.Tensor):
    # Perform slerp on features, each class independently, vectorized

    if self.training and self.config.slerp_feature_augmentation:
        labels = batch.labels

        # Iterate over each unique class label
        for class_label in torch.unique(labels):
            class_mask = labels == class_label

            # If there are fewer than 2 features for the class, skip slerp
            if class_mask.sum() < 2:
                continue

            # Get the features for the current class
            class_features = features[class_mask]

            # Sample pairs of embeddings from the current class
            num_embeddings = len(class_features)
            indices2 = torch.randperm(num_embeddings)
            A = class_features
            B = class_features[indices2]

            # Generate a random interpolation parameter t for each embedding in the batch
            t = torch.rand((num_embeddings, 1), device=features.device, dtype=features.dtype)

            # Extend range from [0, 1] to [t0, t1]
            t0, t1 = self.config.slerp_feature_augmentation_range
            t = t * (t1 - t0) + t0

            # autocast
            augmented_embeddings = slerp(A, B, t)  # Perform slerp

            # Update the features for the current class
            features[class_mask] = augmented_embeddings.to(features.dtype)

    return features
