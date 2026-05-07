# D-023 decision-row parity audit

Sessions aggregated by main sweep: 20260506_175802, 20260506_181244

This audit re-runs each prompt's decision row off-spec vs on-spec at `temperature=0` with **full text + sha256** capture (n=2 reps per prompt). The sha256 verdict is the load-bearing parity claim.

| prompt | block | rep | off sha256 (head) | on sha256 (head) | off len | on len | parity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `factorial` | 3 | 0 | `a60631d7527dd89b...` | `a60631d7527dd89b...` | 716 | 716 | **PASS** |
| `factorial` | 3 | 1 | `a60631d7527dd89b...` | `a60631d7527dd89b...` | 716 | 716 | **PASS** |
| `bst` | 3 | 0 | `3b8a7557e1fe1f3b...` | `ec6a9829c99fa435...` | 977 | 980 | **FAIL** |
| `bst` | 3 | 1 | `3b8a7557e1fe1f3b...` | `ec6a9829c99fa435...` | 977 | 980 | **FAIL** |
| `creative_scene` | 2 | 0 | `0f8e0ff986b46147...` | `c9349e72ee11635a...` | 964 | 922 | **FAIL** |
| `creative_scene` | 2 | 1 | `0f8e0ff986b46147...` | `c9349e72ee11635a...` | 964 | 922 | **FAIL** |
| `factual_explain` | 3 | 0 | `113a834d0a39c2ce...` | `39bb22bf23b63ef0...` | 922 | 936 | **FAIL** |
| `factual_explain` | 3 | 1 | `113a834d0a39c2ce...` | `39bb22bf23b63ef0...` | 922 | 936 | **FAIL** |

**Verdict: FAIL** (2/8 decision-row pairs match bytewise across reps).
