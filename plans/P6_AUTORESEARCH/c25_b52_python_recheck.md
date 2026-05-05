# silica-mlx bench report

Generated: 2026-05-04T11:56:18

Scenarios: total=1 Runs: total=1 ok=1 skipped=0 failed=0

## Results

| id | codec | runs | ok | skipped | failed | ttft_ms | decode_tok_s | resident_mb | peak_mb | wall_s | tokens | vqbench_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3.5-27b-warm-decode-b52 |  | 1 | 1 | 0 | 0 | 20549.6 | 205.5 |  | 35518.3 | 132.654 | 19968 |  |

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
  "aggregate_overlap_window_ms": 88560.32920797588,
  "codec_id": null,
  "decode_tok_s_warm_aggregate": 205.53220796248686,
  "decode_tok_s_warm_per_row_mean": 3.963399068748955,
  "measurement_steps_min": 64,
  "rows": [
    {
      "cold_ttft_ms": 20549.59745897213,
      "decode_interval_ms_mean": 252.3086924244188,
      "decode_interval_ms_std": 12.091135493582541,
      "decode_interval_rel_std": 0.04792199340181093,
      "decode_tok_s_warm": 3.963398923719437,
      "measurement_steps": 351.0,
      "row": 0,
      "row_first_meas_ms": 41636.66033401387,
      "row_last_ms": 130197.01137498487,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 20549.601792008616,
      "decode_interval_ms_mean": 252.30868696583372,
      "decode_interval_ms_std": 12.091166483259025,
      "decode_interval_rel_std": 0.04792211726303481,
      "decode_tok_s_warm": 3.9633990094657925,
      "measurement_steps": 351.0,
      "row": 1,
      "row_first_meas_ms": 41636.67045900365,
      "row_last_ms": 130197.01958401129,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 20549.6027499903,
      "decode_interval_ms_mean": 252.30868803430084,
      "decode_interval_ms_std": 12.091164373071566,
      "decode_interval_rel_std": 0.047922108696581214,
      "decode_tok_s_warm": 3.963398992681743,
      "measurement_steps": 351.0,
      "row": 2,
      "row_first_meas_ms": 41636.67074998375,
      "row_last_ms": 130197.02025002334,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 20549.60312502226,
      "decode_interval_ms_mean": 252.308688270614,
      "decode_interval_ms_std": 12.09115972738923,
      "decode_interval_rel_std": 0.04792209023900454,
      "decode_tok_s_warm": 3.96339898896961,
      "measurement_steps": 351.0,
      "row": 3,
      "row_first_meas_ms": 41636.67095900746,
      "row_last_ms": 130197.02054199297,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 20549.603375024162,
      "decode_interval_ms_mean": 252.308688390346,
      "decode_interval_ms_std": 12.091157332995706,
      "decode_interval_rel_std": 0.047922080726326456,
      "decode_tok_s_warm": 3.9633989870887962,
      "measurement_steps": 351.0,
      "row": 4,
      "row_first_meas_ms": 41636.67108397931,
      "row_last_ms": 130197.02070899075,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 20549.60354202194,
      "decode_interval_ms_mean": 252.3086889828701,
      "decode_interval_ms_std": 12.091156946628377,
      "decode_interval_rel_std": 0.04792207908245791,
      "decode_tok_s_warm": 3.9633989777811123,
      "measurement_steps": 351.0,
      "row": 5,
      "row_first_meas_ms": 41636.671209009364,
      "row_last_ms": 130197.02104199678,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 20549.60370901972,
      "decode_interval_ms_mean": 252.30868910260213,
      "decode_interval_ms_std": 12.091149909867784,
      "decode_interval_rel_std": 0.04792205117022696,
      "decode_tok_s_warm": 3.963398975900298,
      "measurement_steps": 351.0,
      "row": 6,
      "row_first_meas_ms": 41636.67133398121,
      "row_last_ms": 130197.02120899456,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 20549.603916995693,
      "decode_interval_ms_mean": 252.30868922233412,
      "decode_interval_ms_std": 12.091148517247307,
      "decode_interval_rel_std": 0.04792204562797518,
      "decode_tok_s_warm": 3.9633989740194844,
      "measurement_steps": 351.0,
      "row": 7,
      "row_first_meas_ms": 41636.67141698534,
      "row_last_ms": 130197.02133402461,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 20549.604166997597,
      "decode_interval_ms_mean": 252.3086889828701,
      "decode_interval_ms_std": 12.091144961812667,
      "decode_interval_rel_std": 0.04792203158185157,
      "decode_tok_s_warm": 3.9633989777811123,
      "measurement_steps": 351.0,
      "row": 8,
      "row_first_meas_ms": 41636.67166698724,
      "row_last_ms": 130197.02149997465,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 20549.60425000172,
      "decode_interval_ms_mean": 252.30868922216828,
      "decode_interval_ms_std": 12.091141890095843,
      "decode_interval_rel_std": 0.047922019361961374,
      "decode_tok_s_warm": 3.9633989740220894,
      "measurement_steps": 351.0,
      "row": 9,
      "row_first_meas_ms": 41636.671749991365,
      "row_last_ms": 130197.02166697243,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 20549.60437497357,
      "decode_interval_ms_mean": 252.3086891024363,
      "decode_interval_ms_std": 12.091140267716087,
      "decode_interval_rel_std": 0.047922012954564296,
      "decode_tok_s_warm": 3.963398975902903,
      "measurement_steps": 351.0,
      "row": 10,
      "row_first_meas_ms": 41636.67187502142,
      "row_last_ms": 130197.02174997656,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 20549.604667001404,
      "decode_interval_ms_mean": 252.30868922216828,
      "decode_interval_ms_std": 12.091145030376062,
      "decode_interval_rel_std": 0.04792203180814477,
      "decode_tok_s_warm": 3.9633989740220894,
      "measurement_steps": 351.0,
      "row": 11,
      "row_first_meas_ms": 41636.67199999327,
      "row_last_ms": 130197.02191697434,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 20549.604791973252,
      "decode_interval_ms_mean": 252.3086893390811,
      "decode_interval_ms_std": 12.091143917833993,
      "decode_interval_rel_std": 0.04792202737649094,
      "decode_tok_s_warm": 3.9633989721855607,
      "measurement_steps": 351.0,
      "row": 12,
      "row_first_meas_ms": 41636.67208398692,
      "row_last_ms": 130197.02204200439,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 20549.605708976742,
      "decode_interval_ms_mean": 252.30868874639114,
      "decode_interval_ms_std": 12.091138425874284,
      "decode_interval_rel_std": 0.04792200572223547,
      "decode_tok_s_warm": 3.9633989814958497,
      "measurement_steps": 351.0,
      "row": 13,
      "row_first_meas_ms": 41636.67241699295,
      "row_last_ms": 130197.02216697624,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 20549.605958978646,
      "decode_interval_ms_mean": 252.30868851007799,
      "decode_interval_ms_std": 12.091140271466339,
      "decode_interval_rel_std": 0.047922013081937054,
      "decode_tok_s_warm": 3.9633989852079825,
      "measurement_steps": 351.0,
      "row": 14,
      "row_first_meas_ms": 41636.67262496892,
      "row_last_ms": 130197.0222920063,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 20549.606917018536,
      "decode_interval_ms_mean": 252.308688390346,
      "decode_interval_ms_std": 12.091133587319645,
      "decode_interval_rel_std": 0.047921986612738,
      "decode_tok_s_warm": 3.9633989870887962,
      "measurement_steps": 351.0,
      "row": 15,
      "row_first_meas_ms": 41636.67287497083,
      "row_last_ms": 130197.02249998227,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 20549.60712499451,
      "decode_interval_ms_mean": 252.308688273599,
      "decode_interval_ms_std": 12.091131388958122,
      "decode_interval_rel_std": 0.047921977921928385,
      "decode_tok_s_warm": 3.96339898892272,
      "measurement_steps": 351.0,
      "row": 16,
      "row_first_meas_ms": 41636.67312497273,
      "row_last_ms": 130197.02270900598,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 20549.607250024565,
      "decode_interval_ms_mean": 252.308688390346,
      "decode_interval_ms_std": 12.091130852446796,
      "decode_interval_rel_std": 0.04792197577334572,
      "decode_tok_s_warm": 3.9633989870887962,
      "measurement_steps": 351.0,
      "row": 17,
      "row_first_meas_ms": 41636.67325000279,
      "row_last_ms": 130197.02287501423,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 20549.607374996413,
      "decode_interval_ms_mean": 252.308688390346,
      "decode_interval_ms_std": 12.091128118530317,
      "decode_interval_rel_std": 0.04792196493774392,
      "decode_tok_s_warm": 3.9633989870887962,
      "measurement_steps": 351.0,
      "row": 18,
      "row_first_meas_ms": 41636.67333399644,
      "row_last_ms": 130197.02295900788,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 20549.607667024247,
      "decode_interval_ms_mean": 252.3086895755601,
      "decode_interval_ms_std": 12.091118236689253,
      "decode_interval_rel_std": 0.047921925546952945,
      "decode_tok_s_warm": 3.9633989684708233,
      "measurement_steps": 351.0,
      "row": 19,
      "row_first_meas_ms": 41636.67345896829,
      "row_last_ms": 130197.02349998988,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 20549.60787500022,
      "decode_interval_ms_mean": 252.3086895755601,
      "decode_interval_ms_std": 12.091120610067321,
      "decode_interval_rel_std": 0.04792193495359713,
      "decode_tok_s_warm": 3.9633989684708233,
      "measurement_steps": 351.0,
      "row": 20,
      "row_first_meas_ms": 41636.67370897019,
      "row_last_ms": 130197.02374999179,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 20549.608041998,
      "decode_interval_ms_mean": 252.3086894588131,
      "decode_interval_ms_std": 12.091119186865795,
      "decode_interval_rel_std": 0.04792192933505586,
      "decode_tok_s_warm": 3.963398970304747,
      "measurement_steps": 351.0,
      "row": 21,
      "row_first_meas_ms": 41636.6739589721,
      "row_last_ms": 130197.0239590155,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 20549.608334025834,
      "decode_interval_ms_mean": 252.30868957837927,
      "decode_interval_ms_std": 12.091116076681248,
      "decode_interval_rel_std": 0.047921916985443985,
      "decode_tok_s_warm": 3.963398968426538,
      "measurement_steps": 351.0,
      "row": 22,
      "row_first_meas_ms": 41636.67416700628,
      "row_last_ms": 130197.0242090174,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 20549.60841697175,
      "decode_interval_ms_mean": 252.30868945864728,
      "decode_interval_ms_std": 12.091111944244544,
      "decode_interval_rel_std": 0.04792190062968975,
      "decode_tok_s_warm": 3.9633989703073516,
      "measurement_steps": 351.0,
      "row": 23,
      "row_first_meas_ms": 41636.674334004056,
      "row_last_ms": 130197.02433398925,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 20549.60858396953,
      "decode_interval_ms_mean": 252.3086895755601,
      "decode_interval_ms_std": 12.091116766236937,
      "decode_interval_rel_std": 0.04792191971896375,
      "decode_tok_s_warm": 3.9633989684708233,
      "measurement_steps": 351.0,
      "row": 24,
      "row_first_meas_ms": 41636.674458975904,
      "row_last_ms": 130197.0244999975,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 20549.609625013545,
      "decode_interval_ms_mean": 252.30868969512625,
      "decode_interval_ms_std": 12.091112476318893,
      "decode_interval_rel_std": 0.04792190269359737,
      "decode_tok_s_warm": 3.9633989665926146,
      "measurement_steps": 351.0,
      "row": 25,
      "row_first_meas_ms": 41636.67454198003,
      "row_last_ms": 130197.02462496934,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 20549.609749985393,
      "decode_interval_ms_mean": 252.30867782341676,
      "decode_interval_ms_std": 12.091112251476954,
      "decode_interval_rel_std": 0.04792190405729588,
      "decode_tok_s_warm": 3.9633991530797443,
      "measurement_steps": 351.0,
      "row": 26,
      "row_first_meas_ms": 41636.67883398011,
      "row_last_ms": 130197.0247499994,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 20549.609959009103,
      "decode_interval_ms_mean": 252.3086774672058,
      "decode_interval_ms_std": 12.091101782835706,
      "decode_interval_rel_std": 0.047921862633548404,
      "decode_tok_s_warm": 3.963399158675296,
      "measurement_steps": 351.0,
      "row": 27,
      "row_first_meas_ms": 41636.67908398202,
      "row_last_ms": 130197.02487497125,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 20549.61008398095,
      "decode_interval_ms_mean": 252.3086774701908,
      "decode_interval_ms_std": 12.091095523944205,
      "decode_interval_rel_std": 0.047921837826496144,
      "decode_tok_s_warm": 3.963399158628406,
      "measurement_steps": 351.0,
      "row": 28,
      "row_first_meas_ms": 41636.67916698614,
      "row_last_ms": 130197.02495902311,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 20549.61029201513,
      "decode_interval_ms_mean": 252.3086773504588,
      "decode_interval_ms_std": 12.091097647864755,
      "decode_interval_rel_std": 0.047921846267182176,
      "decode_tok_s_warm": 3.96339916050922,
      "measurement_steps": 351.0,
      "row": 29,
      "row_first_meas_ms": 41636.67933398392,
      "row_last_ms": 130197.02508399496,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 20549.61041698698,
      "decode_interval_ms_mean": 252.3086773504588,
      "decode_interval_ms_std": 12.091096297589253,
      "decode_interval_rel_std": 0.04792184091550138,
      "decode_tok_s_warm": 3.96339916050922,
      "measurement_steps": 351.0,
      "row": 30,
      "row_first_meas_ms": 41636.679459013976,
      "row_last_ms": 130197.02520902501,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 20549.61045901291,
      "decode_interval_ms_mean": 252.3086772307268,
      "decode_interval_ms_std": 12.091092690776382,
      "decode_interval_rel_std": 0.04792182664300337,
      "decode_tok_s_warm": 3.9633991623900338,
      "measurement_steps": 351.0,
      "row": 31,
      "row_first_meas_ms": 41636.67966698995,
      "row_last_ms": 130197.02537497506,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 20549.61058398476,
      "decode_interval_ms_mean": 252.3086772307268,
      "decode_interval_ms_std": 12.09109163258941,
      "decode_interval_rel_std": 0.04792182244898602,
      "decode_tok_s_warm": 3.9633991623900338,
      "measurement_steps": 351.0,
      "row": 32,
      "row_first_meas_ms": 41636.679792020004,
      "row_last_ms": 130197.02550000511,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 20549.610666988883,
      "decode_interval_ms_mean": 252.3086772307268,
      "decode_interval_ms_std": 12.091087609040844,
      "decode_interval_rel_std": 0.047921806502057,
      "decode_tok_s_warm": 3.9633991623900338,
      "measurement_steps": 351.0,
      "row": 33,
      "row_first_meas_ms": 41636.67991699185,
      "row_last_ms": 130197.02562497696,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 20549.61079201894,
      "decode_interval_ms_mean": 252.3086772307268,
      "decode_interval_ms_std": 12.091083455939426,
      "decode_interval_rel_std": 0.047921790041658315,
      "decode_tok_s_warm": 3.9633991623900338,
      "measurement_steps": 351.0,
      "row": 34,
      "row_first_meas_ms": 41636.68004202191,
      "row_last_ms": 130197.02575000701,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 20549.610875023063,
      "decode_interval_ms_mean": 252.3086771139798,
      "decode_interval_ms_std": 12.09108195617563,
      "decode_interval_rel_std": 0.04792178411966987,
      "decode_tok_s_warm": 3.963399164223958,
      "measurement_steps": 351.0,
      "row": 35,
      "row_first_meas_ms": 41636.680166993756,
      "row_last_ms": 130197.02583400067,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 20549.610959016718,
      "decode_interval_ms_mean": 252.3086772307268,
      "decode_interval_ms_std": 12.091079282222077,
      "decode_interval_rel_std": 0.04792177349955047,
      "decode_tok_s_warm": 3.9633991623900338,
      "measurement_steps": 351.0,
      "row": 36,
      "row_first_meas_ms": 41636.68029202381,
      "row_last_ms": 130197.02600000892,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 20549.611042020842,
      "decode_interval_ms_mean": 252.3086769942478,
      "decode_interval_ms_std": 12.091079728732309,
      "decode_interval_rel_std": 0.047921775314163945,
      "decode_tok_s_warm": 3.963399166104772,
      "measurement_steps": 351.0,
      "row": 37,
      "row_first_meas_ms": 41636.680499999784,
      "row_last_ms": 130197.02612498077,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 20549.61116699269,
      "decode_interval_ms_mean": 252.30867711116065,
      "decode_interval_ms_std": 12.091076707321381,
      "decode_interval_rel_std": 0.047921763316900776,
      "decode_tok_s_warm": 3.9633991642682425,
      "measurement_steps": 351.0,
      "row": 38,
      "row_first_meas_ms": 41636.68058399344,
      "row_last_ms": 130197.02625001082,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 20549.611249996815,
      "decode_interval_ms_mean": 252.3086769942478,
      "decode_interval_ms_std": 12.091076892262942,
      "decode_interval_rel_std": 0.04792176407210362,
      "decode_tok_s_warm": 3.963399166104772,
      "measurement_steps": 351.0,
      "row": 39,
      "row_first_meas_ms": 41636.68075000169,
      "row_last_ms": 130197.02637498267,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 20549.611416994594,
      "decode_interval_ms_mean": 252.30867699441364,
      "decode_interval_ms_std": 12.09107286152143,
      "decode_interval_rel_std": 0.047921748096634575,
      "decode_tok_s_warm": 3.963399166102167,
      "measurement_steps": 351.0,
      "row": 40,
      "row_first_meas_ms": 41636.680874973536,
      "row_last_ms": 130197.02650001273,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 20549.61149999872,
      "decode_interval_ms_mean": 252.3086769942478,
      "decode_interval_ms_std": 12.091070938313566,
      "decode_interval_rel_std": 0.04792174047422563,
      "decode_tok_s_warm": 3.963399166104772,
      "measurement_steps": 351.0,
      "row": 41,
      "row_first_meas_ms": 41636.68100000359,
      "row_last_ms": 130197.02662498457,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 20549.611583992373,
      "decode_interval_ms_mean": 252.30867699441364,
      "decode_interval_ms_std": 12.091069151409348,
      "decode_interval_rel_std": 0.04792173339197944,
      "decode_tok_s_warm": 3.963399166102167,
      "measurement_steps": 351.0,
      "row": 42,
      "row_first_meas_ms": 41636.68112497544,
      "row_last_ms": 130197.02675001463,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 20549.61170902243,
      "decode_interval_ms_mean": 252.3086774673716,
      "decode_interval_ms_std": 12.091068235720499,
      "decode_interval_rel_std": 0.047921729672908725,
      "decode_tok_s_warm": 3.963399158672691,
      "measurement_steps": 351.0,
      "row": 43,
      "row_first_meas_ms": 41636.681208969094,
      "row_last_ms": 130197.02700001653,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 20549.611791968346,
      "decode_interval_ms_mean": 252.3086774672058,
      "decode_interval_ms_std": 12.091064735751148,
      "decode_interval_rel_std": 0.04792171580116464,
      "decode_tok_s_warm": 3.963399158675296,
      "measurement_steps": 351.0,
      "row": 44,
      "row_first_meas_ms": 41636.68133399915,
      "row_last_ms": 130197.02712498838,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 20549.61187497247,
      "decode_interval_ms_mean": 252.3086773504588,
      "decode_interval_ms_std": 12.09106102791889,
      "decode_interval_rel_std": 0.04792170112771947,
      "decode_tok_s_warm": 3.96339916050922,
      "measurement_steps": 351.0,
      "row": 45,
      "row_first_meas_ms": 41636.681458971,
      "row_last_ms": 130197.02720898204,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 20549.611959024332,
      "decode_interval_ms_mean": 252.3086774672058,
      "decode_interval_ms_std": 12.091059620352487,
      "decode_interval_rel_std": 0.04792169552679789,
      "decode_tok_s_warm": 3.963399158675296,
      "measurement_steps": 351.0,
      "row": 46,
      "row_first_meas_ms": 41636.68158400105,
      "row_last_ms": 130197.02737499028,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 20549.61204197025,
      "decode_interval_ms_mean": 252.3086774673716,
      "decode_interval_ms_std": 12.09105805571654,
      "decode_interval_rel_std": 0.0479216893254896,
      "decode_tok_s_warm": 3.963399158672691,
      "measurement_steps": 351.0,
      "row": 47,
      "row_first_meas_ms": 41636.6817089729,
      "row_last_ms": 130197.02750002034,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 20549.612124974374,
      "decode_interval_ms_mean": 252.3086774701908,
      "decode_interval_ms_std": 12.091045100843326,
      "decode_interval_rel_std": 0.04792163797962055,
      "decode_tok_s_warm": 3.963399158628406,
      "measurement_steps": 351.0,
      "row": 48,
      "row_first_meas_ms": 41636.681791977026,
      "row_last_ms": 130197.027584014,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 20549.612209026236,
      "decode_interval_ms_mean": 252.30867794298294,
      "decode_interval_ms_std": 12.091042208592842,
      "decode_interval_rel_std": 0.047921626426678805,
      "decode_tok_s_warm": 3.9633991512015347,
      "measurement_steps": 351.0,
      "row": 49,
      "row_first_meas_ms": 41636.68191700708,
      "row_last_ms": 130197.02787499409,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 20549.612291972153,
      "decode_interval_ms_mean": 252.30867782623594,
      "decode_interval_ms_std": 12.091038265788953,
      "decode_interval_rel_std": 0.047921610821947264,
      "decode_tok_s_warm": 3.963399153035459,
      "measurement_steps": 351.0,
      "row": 50,
      "row_first_meas_ms": 41636.68204197893,
      "row_last_ms": 130197.02795898775,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 20549.612374976277,
      "decode_interval_ms_mean": 252.30867782623594,
      "decode_interval_ms_std": 12.091034656035276,
      "decode_interval_rel_std": 0.047921596515052595,
      "decode_tok_s_warm": 3.963399153035459,
      "measurement_steps": 351.0,
      "row": 51,
      "row_first_meas_ms": 41636.682167008985,
      "row_last_ms": 130197.0280840178,
      "warmup_steps_used": 32.0
    }
  ],
  "seed": 0,
  "warmup_min_steps": 32,
  "warmup_rel_std_threshold": 0.05,
  "warmup_rolling_window": 16
}
```
