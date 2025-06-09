"""Set a fixed random seed for reproducibility.

* @File    :   set_seed.py
* @Time    :   2025/03/26 09:21:12
* @Author  :   Marc Ballestero Ribó
* @Version :   0
* @Contact :   marcballesteroribo@gmail.com
* @License :   MIT
* @Desc    :   None
"""

import random
import numpy as np
import torch

SEED = 42

def set_seed(
    seed: int = SEED,
) -> np.random.Generator:
    """Set a fixed random seed for reproducibility.

    Args:
        seed: int -- Random seed (default: {SEED}).

    """
    random.seed(seed)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return rng
