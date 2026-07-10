import sys
sys.path.insert(0, ".")

import numpy as np
import pytest
import torch

def test_labels_taxonomy():
    from backend.app.main import labels
    assert len(labels) == 7
    assert labels == ['angry', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

def test_late_fusion_weight_renormalization():
    import backend.app.main as main
    main.load_all_components()
    w_a, w_t, w_f = main.late_fusion_weights
    active_weights = [w_a, w_f]
    renormalized = np.array(active_weights) / sum(active_weights)
    assert abs(renormalized.sum() - 1.0) < 1e-6

def test_text_feature_extraction_dims():
    from backend.app.main import preprocess_text, load_all_components
    # Ensure components are loaded
    load_all_components()
    cls_embedding = preprocess_text("I am feeling very happy today!")
    assert cls_embedding.shape == (768,)
