# silica-mlx bench report

Generated: 2026-05-04T12:12:49

Scenarios: total=1 Runs: total=1 ok=1 skipped=0 failed=0

## Results

| id | codec | runs | ok | skipped | failed | ttft_ms | decode_tok_s | resident_mb | peak_mb | wall_s | tokens | vqbench_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3.5-27b-warm-decode-b52 |  | 1 | 1 | 0 | 0 | 16714.3 | 204.0 |  | 35518.3 | 135.791 | 19968 |  |

## Scenario details

### `qwen3.5-27b-warm-decode-b52` (seed=0)

- repo: `mlx-community/Qwen3.5-27B-4bit`
- oracle: `warm_decode`
- gate: `SILICA_REAL_QWEN3_5_27B`
- workload: `max_batch_size=52`, `max_tokens=384`, `prompts=52`
- status: **ok**

**Cycle 10 (2026-05-03) — dense 27B B=52 warm-decode.** Same workload shape as warm-decode-b4 (128-token prompt, 384-token generation, max_tokens=384) at higher batch. Cycle-10 probe found B=52 sits comfortably within the 36 GB envelope and exceeds the 60 tok/s aggregate gate. Dual-gated on SILICA_REAL_QWEN3_5_27B.

Metadata:

```
{
  "aggregate_overlap_decodes": 18202,
  "aggregate_overlap_window_ms": 89203.71470798273,
  "codec_id": null,
  "decode_tok_s_warm_aggregate": 204.04979836978836,
  "decode_tok_s_warm_per_row_mean": 3.934813123253636,
  "measurement_steps_min": 64,
  "rows": [
    {
      "cold_ttft_ms": 16714.251708996017,
      "decode_interval_ms_mean": 254.1417007349787,
      "decode_interval_ms_std": 12.134848107389931,
      "decode_interval_rel_std": 0.04774835484415154,
      "decode_tok_s_warm": 3.934812732849416,
      "measurement_steps": 351.0,
      "row": 0,
      "row_first_meas_ms": 45112.45658400003,
      "row_last_ms": 134316.19354197755,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16714.264084002934,
      "decode_interval_ms_mean": 254.14167509690867,
      "decode_interval_ms_std": 12.135306135896135,
      "decode_interval_rel_std": 0.04775016191763405,
      "decode_tok_s_warm": 3.9348131297973166,
      "measurement_steps": 351.0,
      "row": 1,
      "row_first_meas_ms": 45112.46774997562,
      "row_last_ms": 134316.19570899056,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16714.26508401055,
      "decode_interval_ms_mean": 254.1416774699902,
      "decode_interval_ms_std": 12.136081329444059,
      "decode_interval_rel_std": 0.04775321171348262,
      "decode_tok_s_warm": 3.9348130930554786,
      "measurement_steps": 351.0,
      "row": 2,
      "row_first_meas_ms": 45112.46804200346,
      "row_last_ms": 134316.19683397003,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16714.26541695837,
      "decode_interval_ms_mean": 254.1416780626802,
      "decode_interval_ms_std": 12.136569128300003,
      "decode_interval_rel_std": 0.047755130999436866,
      "decode_tok_s_warm": 3.934813083879005,
      "measurement_steps": 351.0,
      "row": 3,
      "row_first_meas_ms": 45112.468209001236,
      "row_last_ms": 134316.19720900198,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16714.26558395615,
      "decode_interval_ms_mean": 254.14167853563816,
      "decode_interval_ms_std": 12.136979010536978,
      "decode_interval_rel_std": 0.04775674372055041,
      "decode_tok_s_warm": 3.9348130765563134,
      "measurement_steps": 351.0,
      "row": 4,
      "row_first_meas_ms": 45112.468333973084,
      "row_last_ms": 134316.19749998208,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16714.265708986204,
      "decode_interval_ms_mean": 254.14167936746048,
      "decode_interval_ms_std": 12.137860926947493,
      "decode_interval_rel_std": 0.04776021374045263,
      "decode_tok_s_warm": 3.9348130636774132,
      "measurement_steps": 351.0,
      "row": 5,
      "row_first_meas_ms": 45112.46845900314,
      "row_last_ms": 134316.19791698176,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16714.26579199033,
      "decode_interval_ms_mean": 254.14167948719248,
      "decode_interval_ms_std": 12.138931273111094,
      "decode_interval_rel_std": 0.0477644253300169,
      "decode_tok_s_warm": 3.934813061823632,
      "measurement_steps": 351.0,
      "row": 6,
      "row_first_meas_ms": 45112.46858397499,
      "row_last_ms": 134316.19808397954,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16714.265916962177,
      "decode_interval_ms_mean": 254.14167960393945,
      "decode_interval_ms_std": 12.140339865517689,
      "decode_interval_rel_std": 0.04776996785587271,
      "decode_tok_s_warm": 3.9348130600160673,
      "measurement_steps": 351.0,
      "row": 7,
      "row_first_meas_ms": 45112.46870900504,
      "row_last_ms": 134316.1982499878,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16714.266083959956,
      "decode_interval_ms_mean": 254.14167972350563,
      "decode_interval_ms_std": 12.141874498436128,
      "decode_interval_rel_std": 0.04777600632704531,
      "decode_tok_s_warm": 3.934813058164854,
      "measurement_steps": 351.0,
      "row": 8,
      "row_first_meas_ms": 45112.46879200917,
      "row_last_ms": 134316.19837495964,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16714.26616696408,
      "decode_interval_ms_mean": 254.14167972367144,
      "decode_interval_ms_std": 12.141876796875284,
      "decode_interval_rel_std": 0.047776015370942546,
      "decode_tok_s_warm": 3.9348130581622867,
      "measurement_steps": 351.0,
      "row": 9,
      "row_first_meas_ms": 45112.468916981015,
      "row_last_ms": 134316.1984999897,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16714.266291994136,
      "decode_interval_ms_mean": 254.14167972350563,
      "decode_interval_ms_std": 12.14187830945646,
      "decode_interval_rel_std": 0.04777602132269788,
      "decode_tok_s_warm": 3.934813058164854,
      "measurement_steps": 351.0,
      "row": 10,
      "row_first_meas_ms": 45112.46904201107,
      "row_last_ms": 134316.19862496154,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16714.266416965984,
      "decode_interval_ms_mean": 254.14167972367144,
      "decode_interval_ms_std": 12.14187624706667,
      "decode_interval_rel_std": 0.04777601320754843,
      "decode_tok_s_warm": 3.9348130581622867,
      "measurement_steps": 351.0,
      "row": 11,
      "row_first_meas_ms": 45112.46916698292,
      "row_last_ms": 134316.1987499916,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16714.26649997011,
      "decode_interval_ms_mean": 254.14167972367144,
      "decode_interval_ms_std": 12.141874630028283,
      "decode_interval_rel_std": 0.04777600684480467,
      "decode_tok_s_warm": 3.9348130581622867,
      "measurement_steps": 351.0,
      "row": 12,
      "row_first_meas_ms": 45112.46929195477,
      "row_last_ms": 134316.19887496345,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16714.26670899382,
      "decode_interval_ms_mean": 254.14167984340347,
      "decode_interval_ms_std": 12.141876620694827,
      "decode_interval_rel_std": 0.047776014655197,
      "decode_tok_s_warm": 3.934813056308505,
      "measurement_steps": 351.0,
      "row": 13,
      "row_first_meas_ms": 45112.46937495889,
      "row_last_ms": 134316.1989999935,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16714.269166986924,
      "decode_interval_ms_mean": 254.14167984323763,
      "decode_interval_ms_std": 12.141878660529136,
      "decode_interval_rel_std": 0.047776022681594844,
      "decode_tok_s_warm": 3.934813056311073,
      "measurement_steps": 351.0,
      "row": 14,
      "row_first_meas_ms": 45112.46949998895,
      "row_last_ms": 134316.19912496535,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16714.269333984703,
      "decode_interval_ms_mean": 254.14167996296962,
      "decode_interval_ms_std": 12.141880321303308,
      "decode_interval_rel_std": 0.047776029193922355,
      "decode_tok_s_warm": 3.9348130544572917,
      "measurement_steps": 351.0,
      "row": 15,
      "row_first_meas_ms": 45112.469666986726,
      "row_last_ms": 134316.19933398906,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16714.269541960675,
      "decode_interval_ms_mean": 254.1416800827016,
      "decode_interval_ms_std": 12.141879349277945,
      "decode_interval_rel_std": 0.04777602534667588,
      "decode_tok_s_warm": 3.934813052603511,
      "measurement_steps": 351.0,
      "row": 16,
      "row_first_meas_ms": 45112.4698749627,
      "row_last_ms": 134316.19958399097,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16714.273583958857,
      "decode_interval_ms_mean": 254.1416800797166,
      "decode_interval_ms_std": 12.141881419229767,
      "decode_interval_rel_std": 0.04777603349211048,
      "decode_tok_s_warm": 3.9348130526497274,
      "measurement_steps": 351.0,
      "row": 17,
      "row_first_meas_ms": 45112.47008398641,
      "row_last_ms": 134316.19979196694,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16714.273708988912,
      "decode_interval_ms_mean": 254.14168019944861,
      "decode_interval_ms_std": 12.141880047074508,
      "decode_interval_rel_std": 0.04777602807042767,
      "decode_tok_s_warm": 3.934813050795946,
      "measurement_steps": 351.0,
      "row": 18,
      "row_first_meas_ms": 45112.47020895826,
      "row_last_ms": 134316.19995896472,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16714.273958990816,
      "decode_interval_ms_mean": 254.14168162396086,
      "decode_interval_ms_std": 12.141879436868672,
      "decode_interval_rel_std": 0.04777602540158811,
      "decode_tok_s_warm": 3.9348130287405736,
      "measurement_steps": 351.0,
      "row": 19,
      "row_first_meas_ms": 45112.47033398831,
      "row_last_ms": 134316.20058399858,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16714.274124999065,
      "decode_interval_ms_mean": 254.14168209973803,
      "decode_interval_ms_std": 12.141884922627137,
      "decode_interval_rel_std": 0.04777604689758073,
      "decode_tok_s_warm": 3.9348130213742327,
      "measurement_steps": 351.0,
      "row": 20,
      "row_first_meas_ms": 45112.470541964285,
      "row_last_ms": 134316.20095897233,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16714.274291996844,
      "decode_interval_ms_mean": 254.1416822193042,
      "decode_interval_ms_std": 12.141886371325247,
      "decode_interval_rel_std": 0.04777605257545969,
      "decode_tok_s_warm": 3.9348130195230198,
      "measurement_steps": 351.0,
      "row": 21,
      "row_first_meas_ms": 45112.470749998465,
      "row_last_ms": 134316.20120897423,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16714.27437500097,
      "decode_interval_ms_mean": 254.141682336217,
      "decode_interval_ms_std": 12.141889207185766,
      "decode_interval_rel_std": 0.04777606371206216,
      "decode_tok_s_warm": 3.9348130177128873,
      "measurement_steps": 351.0,
      "row": 22,
      "row_first_meas_ms": 45112.47083399212,
      "row_last_ms": 134316.2013340043,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16714.277166989632,
      "decode_interval_ms_mean": 254.141682216485,
      "decode_interval_ms_std": 12.141889246084883,
      "decode_interval_rel_std": 0.047776063887631316,
      "decode_tok_s_warm": 3.9348130195666684,
      "measurement_steps": 351.0,
      "row": 23,
      "row_first_meas_ms": 45112.47108399402,
      "row_last_ms": 134316.20154198026,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16714.27729196148,
      "decode_interval_ms_mean": 254.141682336217,
      "decode_interval_ms_std": 12.141889975720083,
      "decode_interval_rel_std": 0.047776066736101,
      "decode_tok_s_warm": 3.9348130177128873,
      "measurement_steps": 351.0,
      "row": 24,
      "row_first_meas_ms": 45112.47116699815,
      "row_last_ms": 134316.20166701032,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16714.277416991536,
      "decode_interval_ms_mean": 254.141682336217,
      "decode_interval_ms_std": 12.141889430455842,
      "decode_interval_rel_std": 0.04777606459058816,
      "decode_tok_s_warm": 3.9348130177128873,
      "measurement_steps": 351.0,
      "row": 25,
      "row_first_meas_ms": 45112.471291969996,
      "row_last_ms": 134316.20179198217,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16714.27749999566,
      "decode_interval_ms_mean": 254.14167070363985,
      "decode_interval_ms_std": 12.141895160440404,
      "decode_interval_rel_std": 0.047776089323813935,
      "decode_tok_s_warm": 3.93481319781722,
      "measurement_steps": 351.0,
      "row": 26,
      "row_first_meas_ms": 45112.47554200236,
      "row_last_ms": 134316.20195897995,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16714.277624967508,
      "decode_interval_ms_mean": 254.14167082337184,
      "decode_interval_ms_std": 12.141892896635172,
      "decode_interval_rel_std": 0.047776080393654816,
      "decode_tok_s_warm": 3.934813195963439,
      "measurement_steps": 351.0,
      "row": 27,
      "row_first_meas_ms": 45112.47562500648,
      "row_last_ms": 134316.20208401,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16714.27783399122,
      "decode_interval_ms_mean": 254.14167082337184,
      "decode_interval_ms_std": 12.14189442845519,
      "decode_interval_rel_std": 0.04777608642108044,
      "decode_tok_s_warm": 3.934813195963439,
      "measurement_steps": 351.0,
      "row": 28,
      "row_first_meas_ms": 45112.47574997833,
      "row_last_ms": 134316.20220898185,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16714.277958963066,
      "decode_interval_ms_mean": 254.14167046716088,
      "decode_interval_ms_std": 12.141896691923192,
      "decode_interval_rel_std": 0.0477760953943683,
      "decode_tok_s_warm": 3.934813201478566,
      "measurement_steps": 351.0,
      "row": 29,
      "row_first_meas_ms": 45112.47599998023,
      "row_last_ms": 134316.2023339537,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16714.27804196719,
      "decode_interval_ms_mean": 254.1416701081307,
      "decode_interval_ms_std": 12.14189836545685,
      "decode_interval_rel_std": 0.04777610204690473,
      "decode_tok_s_warm": 3.934813207037342,
      "measurement_steps": 351.0,
      "row": 30,
      "row_first_meas_ms": 45112.476209003944,
      "row_last_ms": 134316.20241695782,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16714.278166997246,
      "decode_interval_ms_mean": 254.14166975208556,
      "decode_interval_ms_std": 12.141899677228109,
      "decode_interval_rel_std": 0.04777610727541255,
      "decode_tok_s_warm": 3.934813212549902,
      "measurement_steps": 351.0,
      "row": 31,
      "row_first_meas_ms": 45112.47645900585,
      "row_last_ms": 134316.20254198788,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16714.27825000137,
      "decode_interval_ms_mean": 254.14166987181756,
      "decode_interval_ms_std": 12.141898647286789,
      "decode_interval_rel_std": 0.04777610320027741,
      "decode_tok_s_warm": 3.9348132106961207,
      "measurement_steps": 351.0,
      "row": 32,
      "row_first_meas_ms": 45112.476583977696,
      "row_last_ms": 134316.20270898566,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16714.278333995026,
      "decode_interval_ms_mean": 254.1416693988596,
      "decode_interval_ms_std": 12.14190169719874,
      "decode_interval_rel_std": 0.04777611529002266,
      "decode_tok_s_warm": 3.934813218018813,
      "measurement_steps": 351.0,
      "row": 33,
      "row_first_meas_ms": 45112.47687495779,
      "row_last_ms": 134316.2028339575,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16714.27858399693,
      "decode_interval_ms_mean": 254.14166951560657,
      "decode_interval_ms_std": 12.141903519054155,
      "decode_interval_rel_std": 0.04777612243673615,
      "decode_tok_s_warm": 3.934813216211248,
      "measurement_steps": 351.0,
      "row": 34,
      "row_first_meas_ms": 45112.476959009655,
      "row_last_ms": 134316.20295898756,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16714.278667001054,
      "decode_interval_ms_mean": 254.1416693960404,
      "decode_interval_ms_std": 12.141905309367639,
      "decode_interval_rel_std": 0.04777612950376257,
      "decode_tok_s_warm": 3.9348132180624615,
      "measurement_steps": 351.0,
      "row": 35,
      "row_first_meas_ms": 45112.4770839815,
      "row_last_ms": 134316.20304199168,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16714.278750005178,
      "decode_interval_ms_mean": 254.14166951560657,
      "decode_interval_ms_std": 12.14191360195426,
      "decode_interval_rel_std": 0.04777616211106474,
      "decode_tok_s_warm": 3.934813216211248,
      "measurement_steps": 351.0,
      "row": 36,
      "row_first_meas_ms": 45112.47720901156,
      "row_last_ms": 134316.20320898946,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16714.278833998833,
      "decode_interval_ms_mean": 254.14166963251938,
      "decode_interval_ms_std": 12.141911840181184,
      "decode_interval_rel_std": 0.047776155156838286,
      "decode_tok_s_warm": 3.9348132144011156,
      "measurement_steps": 351.0,
      "row": 37,
      "row_first_meas_ms": 45112.47733398341,
      "row_last_ms": 134316.2033749977,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16714.27895897068,
      "decode_interval_ms_mean": 254.14166975208556,
      "decode_interval_ms_std": 12.14191195430604,
      "decode_interval_rel_std": 0.04777615558342101,
      "decode_tok_s_warm": 3.934813212549902,
      "measurement_steps": 351.0,
      "row": 38,
      "row_first_meas_ms": 45112.47741698753,
      "row_last_ms": 134316.20349996956,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16714.279041974805,
      "decode_interval_ms_mean": 254.14166987181756,
      "decode_interval_ms_std": 12.141916945329413,
      "decode_interval_rel_std": 0.04777617519965726,
      "decode_tok_s_warm": 3.9348132106961207,
      "measurement_steps": 351.0,
      "row": 39,
      "row_first_meas_ms": 45112.47754195938,
      "row_last_ms": 134316.20366696734,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16714.27916700486,
      "decode_interval_ms_mean": 254.14166987181756,
      "decode_interval_ms_std": 12.141917962467407,
      "decode_interval_rel_std": 0.04777617920190528,
      "decode_tok_s_warm": 3.9348132106961207,
      "measurement_steps": 351.0,
      "row": 40,
      "row_first_meas_ms": 45112.477666989435,
      "row_last_ms": 134316.2037919974,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16714.279250008985,
      "decode_interval_ms_mean": 254.1416701111157,
      "decode_interval_ms_std": 12.141919754945386,
      "decode_interval_rel_std": 0.04777618620998556,
      "decode_tok_s_warm": 3.934813206991126,
      "measurement_steps": 351.0,
      "row": 41,
      "row_first_meas_ms": 45112.47774999356,
      "row_last_ms": 134316.20395899517,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16714.27929197671,
      "decode_interval_ms_mean": 254.1416702278627,
      "decode_interval_ms_std": 12.141923219458024,
      "decode_interval_rel_std": 0.047776199820248326,
      "decode_tok_s_warm": 3.934813205183561,
      "measurement_steps": 351.0,
      "row": 42,
      "row_first_meas_ms": 45112.477833987214,
      "row_last_ms": 134316.20408396702,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16714.279374980833,
      "decode_interval_ms_mean": 254.14167022802854,
      "decode_interval_ms_std": 12.141925130242079,
      "decode_interval_rel_std": 0.047776207338795486,
      "decode_tok_s_warm": 3.934813205180993,
      "measurement_steps": 351.0,
      "row": 43,
      "row_first_meas_ms": 45112.47795895906,
      "row_last_ms": 134316.20420899708,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16714.279458974488,
      "decode_interval_ms_mean": 254.1416702278627,
      "decode_interval_ms_std": 12.141925413563952,
      "decode_interval_rel_std": 0.0477762084536453,
      "decode_tok_s_warm": 3.934813205183561,
      "measurement_steps": 351.0,
      "row": 44,
      "row_first_meas_ms": 45112.47808398912,
      "row_last_ms": 134316.20433396893,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16714.279584004544,
      "decode_interval_ms_mean": 254.14167010829655,
      "decode_interval_ms_std": 12.141921001344745,
      "decode_interval_rel_std": 0.04777619111486419,
      "decode_tok_s_warm": 3.9348132070347743,
      "measurement_steps": 351.0,
      "row": 45,
      "row_first_meas_ms": 45112.478208960965,
      "row_last_ms": 134316.20441697305,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16714.27966700867,
      "decode_interval_ms_mean": 254.1416705840737,
      "decode_interval_ms_std": 12.141920274235675,
      "decode_interval_rel_std": 0.0477761881643843,
      "decode_tok_s_warm": 3.9348131996684335,
      "measurement_steps": 351.0,
      "row": 46,
      "row_first_meas_ms": 45112.47829196509,
      "row_last_ms": 134316.20466697495,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16714.279749954585,
      "decode_interval_ms_mean": 254.1416707038057,
      "decode_interval_ms_std": 12.14192184476178,
      "decode_interval_rel_std": 0.04777619432160268,
      "decode_tok_s_warm": 3.9348131978146523,
      "measurement_steps": 351.0,
      "row": 47,
      "row_first_meas_ms": 45112.478374969214,
      "row_last_ms": 134316.204792005,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16714.279834006447,
      "decode_interval_ms_mean": 254.14167070363985,
      "decode_interval_ms_std": 12.141923017126045,
      "decode_interval_rel_std": 0.04777619893466824,
      "decode_tok_s_warm": 3.93481319781722,
      "measurement_steps": 351.0,
      "row": 48,
      "row_first_meas_ms": 45112.47849999927,
      "row_last_ms": 134316.20491697686,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16714.27991701057,
      "decode_interval_ms_mean": 254.14167082055266,
      "decode_interval_ms_std": 12.141926437200468,
      "decode_interval_rel_std": 0.04777621237004372,
      "decode_tok_s_warm": 3.9348131960070876,
      "measurement_steps": 351.0,
      "row": 49,
      "row_first_meas_ms": 45112.478583992925,
      "row_last_ms": 134316.2050420069,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16714.27999995649,
      "decode_interval_ms_mean": 254.14167082055266,
      "decode_interval_ms_std": 12.14192452499108,
      "decode_interval_rel_std": 0.047776204845856984,
      "decode_tok_s_warm": 3.9348131960070876,
      "measurement_steps": 351.0,
      "row": 50,
      "row_first_meas_ms": 45112.47870896477,
      "row_last_ms": 134316.20516697876,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16714.28008400835,
      "decode_interval_ms_mean": 254.14167082055266,
      "decode_interval_ms_std": 12.141928473872971,
      "decode_interval_rel_std": 0.047776220383969566,
      "decode_tok_s_warm": 3.9348131960070876,
      "measurement_steps": 351.0,
      "row": 51,
      "row_first_meas_ms": 45112.47883399483,
      "row_last_ms": 134316.20529200882,
      "warmup_steps_used": 32.0
    }
  ],
  "seed": 0,
  "warmup_min_steps": 32,
  "warmup_rel_std_threshold": 0.05,
  "warmup_rolling_window": 16
}
```
