# D-023 decision-row parity audit

Sessions aggregated by main sweep: 20260506_175802, 20260506_181244

This audit re-runs each prompt's decision row off-spec vs on-spec at `temperature=0` with **full text + sha256** capture (n=1 reps per prompt). The sha256 verdict is the load-bearing parity claim.

| prompt | block | rep | off sha256 (head) | on sha256 (head) | off len | on len | parity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `factorial` | 3 | 0 | `aa3b17600d88d316...` | `aa3b17600d88d316...` | 4 | 4 | **PASS** |
| `bst` | 3 | 0 | `559aead08264d579...` | `559aead08264d579...` | 1 | 1 | **PASS** |
| `creative_scene` | 2 | 0 | `b344d80e24a36799...` | `b344d80e24a36799...` | 3 | 3 | **PASS** |
| `factual_explain` | 3 | 0 | `7108efeda67ae946...` | `7108efeda67ae946...` | 7 | 7 | **PASS** |

**Verdict: PASS** (4/4 decision-row pairs match bytewise across reps).
