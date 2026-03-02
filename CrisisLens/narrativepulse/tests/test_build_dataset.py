from __future__ import annotations

import pandas as pd

from narrativepulse.models.weak_supervision import fit_label_model


def test_weak_supervision_outputs_shapes() -> None:
    frame = pd.DataFrame(
        {
            "gdelt_event_count_total": [10, 12, 11, 25, 13],
            "gdelt_goldstein_min": [-1.0, -2.0, -1.2, -5.0, -1.1],
            "attention_shock_index": [0.1, 0.2, 0.1, 4.5, 0.0],
            "macro_VIXCLS": [20.0, 20.5, 21.0, 30.0, 22.0],
        }
    )
    result = fit_label_model(frame)
    assert len(result.probabilities) == len(frame)
    assert len(result.pseudo_labels) == len(frame)
    assert set(result.coverage.keys()) == {
        "event_spike",
        "goldstein_tail",
        "attention_spike",
        "macro_regime",
    }
