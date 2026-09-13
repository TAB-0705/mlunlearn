"""
Aggregate Membership-Inference signal  —  Layer 5.  Owner: Person B.

The cheap, no-shadow-model baseline auditor. It asks a POPULATION-level question:
across the whole forget set, does the model still behave as if it trained on
those records? It compares the model's logit-scaled confidence on the forget set
(candidate members) against the held-out test set (known non-members).

  - On the ORIGINAL model:            forget scores >> test scores  ->  AUC -> 1
  - On the GOLD model (never saw forget): forget scores ~ test scores -> AUC ~ 0.5
  - On a genuinely unlearned model:   AUC drifts back toward 0.5.

Its blind spot — and the reason LiRA exists — is that it is an AVERAGE. It can
report "forgotten on average" (AUC near 0.5) while a tail of individual records
stays fully exposed. Measuring that gap between MIA and LiRA is thesis claim #1.

Works unchanged on tabular OR image data: it only needs model + labels.
audit() never reads ground-truth membership — it only uses the forget set it was
handed and the test set as a non-member reference.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from core.interfaces import VerificationSignal
from core.verify._common import membership_scores
from core.verify._predict import logits_of
from core.verify.scoring import verdict_from_scores


class AggregateMIA(VerificationSignal):
    name = "aggregate_mia"

    def audit(self, model: Any,
              data: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict:
        Xf, yf = data["forget"]
        Xt, yt = data["test"]
        forget_scores = membership_scores(logits_of(model, Xf), yf)
        test_scores = membership_scores(logits_of(model, Xt), yt)
        return verdict_from_scores(self.name, forget_scores, test_scores)
