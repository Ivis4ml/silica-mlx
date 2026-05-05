# silica-mlx bench report

Generated: 2026-05-04T12:15:24

Scenarios: total=1 Runs: total=1 ok=1 skipped=0 failed=0

## Results

| id | codec | runs | ok | skipped | failed | ttft_ms | decode_tok_s | resident_mb | peak_mb | wall_s | tokens | vqbench_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3.5-27b-warm-decode-b52 |  | 1 | 1 | 0 | 0 | 18799.4 | 190.0 |  | 35518.3 | 128.519 | 19968 |  |

## Scenario details

### `qwen3.5-27b-warm-decode-b52` (seed=1)

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
  "aggregate_overlap_window_ms": 95777.50795800239,
  "codec_id": null,
  "decode_tok_s_warm_aggregate": 190.04461890970708,
  "decode_tok_s_warm_per_row_mean": 3.66474264885696,
  "measurement_steps_min": 64,
  "rows": [
    {
      "cold_ttft_ms": 18799.363374942914,
      "decode_interval_ms_mean": 272.8704580968509,
      "decode_interval_ms_std": 12.417477396851904,
      "decode_interval_rel_std": 0.04550685876169316,
      "decode_tok_s_warm": 3.664742629064911,
      "measurement_steps": 351.0,
      "row": 0,
      "row_first_meas_ms": 30241.153582988773,
      "row_last_ms": 126018.68437498342,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 18799.368166946806,
      "decode_interval_ms_mean": 272.8704563162934,
      "decode_interval_ms_std": 12.417532724411954,
      "decode_interval_rel_std": 0.045507061819907574,
      "decode_tok_s_warm": 3.6647426529784006,
      "measurement_steps": 351.0,
      "row": 1,
      "row_first_meas_ms": 30241.166582971346,
      "row_last_ms": 126018.69674999034,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 18799.369082960766,
      "decode_interval_ms_mean": 272.8704576210737,
      "decode_interval_ms_std": 12.41752512122015,
      "decode_interval_rel_std": 0.04550703373856602,
      "decode_tok_s_warm": 3.6647426354547603,
      "measurement_steps": 351.0,
      "row": 2,
      "row_first_meas_ms": 30241.16683297325,
      "row_last_ms": 126018.69745797012,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 18799.369416956324,
      "decode_interval_ms_mean": 272.87045809386586,
      "decode_interval_ms_std": 12.417525030078483,
      "decode_interval_rel_std": 0.04550703332570698,
      "decode_tok_s_warm": 3.664742629105001,
      "measurement_steps": 351.0,
      "row": 3,
      "row_first_meas_ms": 30241.16704199696,
      "row_last_ms": 126018.69783294387,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 18799.369624990504,
      "decode_interval_ms_mean": 272.8704580940317,
      "decode_interval_ms_std": 12.417534748064668,
      "decode_interval_rel_std": 0.045507068939597564,
      "decode_tok_s_warm": 3.6647426291027743,
      "measurement_steps": 351.0,
      "row": 4,
      "row_first_meas_ms": 30241.167166968808,
      "row_last_ms": 126018.69795797393,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 18799.369749962352,
      "decode_interval_ms_mean": 272.87045845007685,
      "decode_interval_ms_std": 12.417537713699032,
      "decode_interval_rel_std": 0.04550707974850597,
      "decode_tok_s_warm": 3.664742624320967,
      "measurement_steps": 351.0,
      "row": 5,
      "row_first_meas_ms": 30241.167291998863,
      "row_last_ms": 126018.69820797583,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 18799.369832966477,
      "decode_interval_ms_mean": 272.8704585698088,
      "decode_interval_ms_std": 12.417549441011648,
      "decode_interval_rel_std": 0.04550712270612046,
      "decode_tok_s_warm": 3.664742622712926,
      "measurement_steps": 351.0,
      "row": 6,
      "row_first_meas_ms": 30241.16737494478,
      "row_last_ms": 126018.69833294768,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 18799.369957996532,
      "decode_interval_ms_mean": 272.8704585698088,
      "decode_interval_ms_std": 12.417551886418853,
      "decode_interval_rel_std": 0.04550713166790847,
      "decode_tok_s_warm": 3.664742622712926,
      "measurement_steps": 351.0,
      "row": 7,
      "row_first_meas_ms": 30241.167499974836,
      "row_last_ms": 126018.69845797773,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 18799.370041990187,
      "decode_interval_ms_mean": 272.8704589260198,
      "decode_interval_ms_std": 12.417546642426506,
      "decode_interval_rel_std": 0.04550711239062024,
      "decode_tok_s_warm": 3.664742617928892,
      "measurement_steps": 351.0,
      "row": 8,
      "row_first_meas_ms": 30241.167624946684,
      "row_last_ms": 126018.69870797964,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 18799.37012499431,
      "decode_interval_ms_mean": 272.870458925854,
      "decode_interval_ms_std": 12.417546439757924,
      "decode_interval_rel_std": 0.0455071116479197,
      "decode_tok_s_warm": 3.6647426179311187,
      "measurement_steps": 351.0,
      "row": 9,
      "row_first_meas_ms": 30241.16774997674,
      "row_last_ms": 126018.69883295149,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 18799.37024996616,
      "decode_interval_ms_mean": 272.8704589260198,
      "decode_interval_ms_std": 12.41754839068754,
      "decode_interval_rel_std": 0.04550711879754695,
      "decode_tok_s_warm": 3.664742617928892,
      "measurement_steps": 351.0,
      "row": 10,
      "row_first_meas_ms": 30241.167874948587,
      "row_last_ms": 126018.69895798154,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 18799.370332970284,
      "decode_interval_ms_mean": 272.87045904558596,
      "decode_interval_ms_std": 12.417548185581387,
      "decode_interval_rel_std": 0.04550711802594542,
      "decode_tok_s_warm": 3.664742616323078,
      "measurement_steps": 351.0,
      "row": 11,
      "row_first_meas_ms": 30241.16795795271,
      "row_last_ms": 126018.69908295339,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 18799.370541993994,
      "decode_interval_ms_mean": 272.8704591624988,
      "decode_interval_ms_std": 12.417554632576355,
      "decode_interval_rel_std": 0.04550714163302485,
      "decode_tok_s_warm": 3.664742614752899,
      "measurement_steps": 351.0,
      "row": 12,
      "row_first_meas_ms": 30241.168041946366,
      "row_last_ms": 126018.69920798345,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 18799.37170795165,
      "decode_interval_ms_mean": 272.8704591624988,
      "decode_interval_ms_std": 12.417545244991729,
      "decode_interval_rel_std": 0.045507107229943417,
      "decode_tok_s_warm": 3.664742614752899,
      "measurement_steps": 351.0,
      "row": 13,
      "row_first_meas_ms": 30241.16829194827,
      "row_last_ms": 126018.69945798535,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 18799.372707959265,
      "decode_interval_ms_mean": 272.8704595213631,
      "decode_interval_ms_std": 12.417544761189541,
      "decode_interval_rel_std": 0.04550710539708447,
      "decode_tok_s_warm": 3.66474260993323,
      "measurement_steps": 351.0,
      "row": 14,
      "row_first_meas_ms": 30241.168624954298,
      "row_last_ms": 126018.69991695276,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 18799.372916982975,
      "decode_interval_ms_mean": 272.8704596409293,
      "decode_interval_ms_std": 12.417544538366599,
      "decode_interval_rel_std": 0.04550710456055546,
      "decode_tok_s_warm": 3.6647426083274155,
      "measurement_steps": 351.0,
      "row": 15,
      "row_first_meas_ms": 30241.16883298848,
      "row_last_ms": 126018.70016695466,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 18799.373124958947,
      "decode_interval_ms_mean": 272.870459638276,
      "decode_interval_ms_std": 12.417544612778386,
      "decode_interval_rel_std": 0.045507104833698006,
      "decode_tok_s_warm": 3.6647426083630505,
      "measurement_steps": 351.0,
      "row": 16,
      "row_first_meas_ms": 30241.16904195398,
      "row_last_ms": 126018.70037498884,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 18799.373249989003,
      "decode_interval_ms_mean": 272.8704595213631,
      "decode_interval_ms_std": 12.41754158047132,
      "decode_interval_rel_std": 0.045507093740570874,
      "decode_tok_s_warm": 3.66474260993323,
      "measurement_steps": 351.0,
      "row": 17,
      "row_first_meas_ms": 30241.16920796223,
      "row_last_ms": 126018.70049996069,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 18799.373332993127,
      "decode_interval_ms_mean": 272.8704595213631,
      "decode_interval_ms_std": 12.417544564280588,
      "decode_interval_rel_std": 0.04550710467546384,
      "decode_tok_s_warm": 3.66474260993323,
      "measurement_steps": 351.0,
      "row": 18,
      "row_first_meas_ms": 30241.169332992285,
      "row_last_ms": 126018.70062499074,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 18799.373499990907,
      "decode_interval_ms_mean": 272.8704608261434,
      "decode_interval_ms_std": 12.417547841457925,
      "decode_interval_rel_std": 0.04550711646787461,
      "decode_tok_s_warm": 3.664742592409589,
      "measurement_steps": 351.0,
      "row": 19,
      "row_first_meas_ms": 30241.169499990065,
      "row_last_ms": 126018.7012499664,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 18799.373666988686,
      "decode_interval_ms_mean": 272.8704611823544,
      "decode_interval_ms_std": 12.41755209994871,
      "decode_interval_rel_std": 0.04550713201474118,
      "decode_tok_s_warm": 3.664742587625555,
      "measurement_steps": 351.0,
      "row": 20,
      "row_first_meas_ms": 30241.169707966037,
      "row_last_ms": 126018.70158297243,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 18799.373957968783,
      "decode_interval_ms_mean": 272.8704614188334,
      "decode_interval_ms_std": 12.417558087168551,
      "decode_interval_rel_std": 0.04550715391692264,
      "decode_tok_s_warm": 3.6647425844495625,
      "measurement_steps": 351.0,
      "row": 21,
      "row_first_meas_ms": 30241.169916989747,
      "row_last_ms": 126018.70187500026,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 18799.37412496656,
      "decode_interval_ms_mean": 272.8704615383995,
      "decode_interval_ms_std": 12.417555607072376,
      "decode_interval_rel_std": 0.04550714480806829,
      "decode_tok_s_warm": 3.6647425828437483,
      "measurement_steps": 351.0,
      "row": 22,
      "row_first_meas_ms": 30241.17012496572,
      "row_last_ms": 126018.70212494396,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 18799.375249946024,
      "decode_interval_ms_mean": 272.87046165813155,
      "decode_interval_ms_std": 12.417559328036013,
      "decode_interval_rel_std": 0.04550715842447423,
      "decode_tok_s_warm": 3.664742581235707,
      "measurement_steps": 351.0,
      "row": 23,
      "row_first_meas_ms": 30241.170207969844,
      "row_last_ms": 126018.70224997401,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 18799.37537497608,
      "decode_interval_ms_mean": 272.8704616579657,
      "decode_interval_ms_std": 12.417561369215493,
      "decode_interval_rel_std": 0.04550716590489925,
      "decode_tok_s_warm": 3.6647425812379346,
      "measurement_steps": 351.0,
      "row": 24,
      "row_first_meas_ms": 30241.1703329999,
      "row_last_ms": 126018.70237494586,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 18799.375457980204,
      "decode_interval_ms_mean": 272.8704620113575,
      "decode_interval_ms_std": 12.417561360181518,
      "decode_interval_rel_std": 0.04550716581285617,
      "decode_tok_s_warm": 3.6647425764917627,
      "measurement_steps": 351.0,
      "row": 25,
      "row_first_meas_ms": 30241.170416993555,
      "row_last_ms": 126018.70258298004,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 18799.375582952052,
      "decode_interval_ms_mean": 272.8704544158382,
      "decode_interval_ms_std": 12.41756664862074,
      "decode_interval_rel_std": 0.04550718646034544,
      "decode_tok_s_warm": 3.6647426785021584,
      "measurement_steps": 351.0,
      "row": 26,
      "row_first_meas_ms": 30241.173207992688,
      "row_last_ms": 126018.70270795189,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 18799.375666945707,
      "decode_interval_ms_mean": 272.87045382049484,
      "decode_interval_ms_std": 12.417572191663725,
      "decode_interval_rel_std": 0.045507206873458356,
      "decode_tok_s_warm": 3.6647426864978216,
      "measurement_steps": 351.0,
      "row": 27,
      "row_first_meas_ms": 30241.173541988246,
      "row_last_ms": 126018.70283298194,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 18799.37591694761,
      "decode_interval_ms_mean": 272.87045370358203,
      "decode_interval_ms_std": 12.417581931994587,
      "decode_interval_rel_std": 0.04550724258876247,
      "decode_tok_s_warm": 3.6647426880680003,
      "measurement_steps": 351.0,
      "row": 28,
      "row_first_meas_ms": 30241.17379199015,
      "row_last_ms": 126018.70304194745,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 18799.375999951735,
      "decode_interval_ms_mean": 272.8704537037479,
      "decode_interval_ms_std": 12.417583325578955,
      "decode_interval_rel_std": 0.04550724769586293,
      "decode_tok_s_warm": 3.6647426880657723,
      "measurement_steps": 351.0,
      "row": 29,
      "row_first_meas_ms": 30241.173916961998,
      "row_last_ms": 126018.7031669775,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 18799.37608295586,
      "decode_interval_ms_mean": 272.870453586835,
      "decode_interval_ms_std": 12.41758763147575,
      "decode_interval_rel_std": 0.04550726349536457,
      "decode_tok_s_warm": 3.664742689635952,
      "measurement_steps": 351.0,
      "row": 30,
      "row_first_meas_ms": 30241.174082970247,
      "row_last_ms": 126018.70329194935,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 18799.376166949514,
      "decode_interval_ms_mean": 272.87045346428386,
      "decode_interval_ms_std": 12.417586198070982,
      "decode_interval_rel_std": 0.04550725826274308,
      "decode_tok_s_warm": 3.6647426912818557,
      "measurement_steps": 351.0,
      "row": 31,
      "row_first_meas_ms": 30241.174291993957,
      "row_last_ms": 126018.7034579576,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 18799.37629197957,
      "decode_interval_ms_mean": 272.8704533475369,
      "decode_interval_ms_std": 12.41758812809286,
      "decode_interval_rel_std": 0.045507265355246816,
      "decode_tok_s_warm": 3.664742692849807,
      "measurement_steps": 351.0,
      "row": 32,
      "row_first_meas_ms": 30241.174416965805,
      "row_last_ms": 126018.70354195125,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 18799.376374983694,
      "decode_interval_ms_mean": 272.8704534672689,
      "decode_interval_ms_std": 12.41759304354029,
      "decode_interval_rel_std": 0.045507283349128874,
      "decode_tok_s_warm": 3.664742691241766,
      "measurement_steps": 351.0,
      "row": 33,
      "row_first_meas_ms": 30241.17449996993,
      "row_last_ms": 126018.70366698131,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 18799.376499955542,
      "decode_interval_ms_mean": 272.8704533477027,
      "decode_interval_ms_std": 12.417593860183405,
      "decode_interval_rel_std": 0.04550728636185611,
      "decode_tok_s_warm": 3.66474269284758,
      "measurement_steps": 351.0,
      "row": 34,
      "row_first_meas_ms": 30241.174707945902,
      "row_last_ms": 126018.70383298956,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 18799.376582959667,
      "decode_interval_ms_mean": 272.8704533475369,
      "decode_interval_ms_std": 12.417598804764904,
      "decode_interval_rel_std": 0.045507304482502675,
      "decode_tok_s_warm": 3.664742692849807,
      "measurement_steps": 351.0,
      "row": 35,
      "row_first_meas_ms": 30241.174832975958,
      "row_last_ms": 126018.7039579614,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 18799.376624985598,
      "decode_interval_ms_mean": 272.8704533477027,
      "decode_interval_ms_std": 12.417595970202958,
      "decode_interval_rel_std": 0.04550729409453485,
      "decode_tok_s_warm": 3.66474269284758,
      "measurement_steps": 351.0,
      "row": 36,
      "row_first_meas_ms": 30241.174957947806,
      "row_last_ms": 126018.70408299146,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 18799.376707989722,
      "decode_interval_ms_mean": 272.87045346428386,
      "decode_interval_ms_std": 12.417599365254413,
      "decode_interval_rel_std": 0.04550730651708232,
      "decode_tok_s_warm": 3.6647426912818557,
      "measurement_steps": 351.0,
      "row": 37,
      "row_first_meas_ms": 30241.175041999668,
      "row_last_ms": 126018.70420796331,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 18799.376791983377,
      "decode_interval_ms_mean": 272.8704533475369,
      "decode_interval_ms_std": 12.41760224563541,
      "decode_interval_rel_std": 0.04550731709240772,
      "decode_tok_s_warm": 3.664742692849807,
      "measurement_steps": 351.0,
      "row": 38,
      "row_first_meas_ms": 30241.175166971516,
      "row_last_ms": 126018.70429195696,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 18799.376916955225,
      "decode_interval_ms_mean": 272.8704535840159,
      "decode_interval_ms_std": 12.41760486920969,
      "decode_interval_rel_std": 0.04550732666769417,
      "decode_tok_s_warm": 3.664742689673814,
      "measurement_steps": 351.0,
      "row": 39,
      "row_first_meas_ms": 30241.17524997564,
      "row_last_ms": 126018.70445796521,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 18799.37699995935,
      "decode_interval_ms_mean": 272.8704535841817,
      "decode_interval_ms_std": 12.417609747533778,
      "decode_interval_rel_std": 0.04550734454546906,
      "decode_tok_s_warm": 3.6647426896715873,
      "measurement_steps": 351.0,
      "row": 40,
      "row_first_meas_ms": 30241.17537494749,
      "row_last_ms": 126018.70458299527,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 18799.377082963474,
      "decode_interval_ms_mean": 272.8704535870009,
      "decode_interval_ms_std": 12.417611018594016,
      "decode_interval_rel_std": 0.0455073492031076,
      "decode_tok_s_warm": 3.6647426896337243,
      "measurement_steps": 351.0,
      "row": 41,
      "row_first_meas_ms": 30241.175457951613,
      "row_last_ms": 126018.70466698892,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 18799.37716695713,
      "decode_interval_ms_mean": 272.8704538206607,
      "decode_interval_ms_std": 12.417612008292576,
      "decode_interval_rel_std": 0.04550735279113008,
      "decode_tok_s_warm": 3.6647426864955936,
      "measurement_steps": 351.0,
      "row": 42,
      "row_first_meas_ms": 30241.175541945267,
      "row_last_ms": 126018.70483299717,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 18799.377249961253,
      "decode_interval_ms_mean": 272.87045394022687,
      "decode_interval_ms_std": 12.417615907422476,
      "decode_interval_rel_std": 0.04550736706049748,
      "decode_tok_s_warm": 3.66474268488978,
      "measurement_steps": 351.0,
      "row": 43,
      "row_first_meas_ms": 30241.175624949392,
      "row_last_ms": 126018.70495796902,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 18799.377332965378,
      "decode_interval_ms_mean": 272.87045394022687,
      "decode_interval_ms_std": 12.417619050609309,
      "decode_interval_rel_std": 0.04550737857946844,
      "decode_tok_s_warm": 3.66474268488978,
      "measurement_steps": 351.0,
      "row": 44,
      "row_first_meas_ms": 30241.175749979448,
      "row_last_ms": 126018.70508299908,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 18799.377416959032,
      "decode_interval_ms_mean": 272.870454059793,
      "decode_interval_ms_std": 12.417621313141598,
      "decode_interval_rel_std": 0.04550738685112671,
      "decode_tok_s_warm": 3.664742683283966,
      "measurement_steps": 351.0,
      "row": 45,
      "row_first_meas_ms": 30241.175832983572,
      "row_last_ms": 126018.70520797092,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 18799.377499963157,
      "decode_interval_ms_mean": 272.87045417670583,
      "decode_interval_ms_std": 12.417622628199323,
      "decode_interval_rel_std": 0.045507391650976996,
      "decode_tok_s_warm": 3.664742681713787,
      "measurement_steps": 351.0,
      "row": 46,
      "row_first_meas_ms": 30241.175916977227,
      "row_last_ms": 126018.70533300098,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 18799.37758296728,
      "decode_interval_ms_mean": 272.8704540599589,
      "decode_interval_ms_std": 12.417624074836633,
      "decode_interval_rel_std": 0.045507396972000715,
      "decode_tok_s_warm": 3.6647426832817382,
      "measurement_steps": 351.0,
      "row": 47,
      "row_first_meas_ms": 30241.176041949075,
      "row_last_ms": 126018.70541699464,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 18799.377666960936,
      "decode_interval_ms_mean": 272.870454296272,
      "decode_interval_ms_std": 12.417610825828978,
      "decode_interval_rel_std": 0.04550734837838627,
      "decode_tok_s_warm": 3.664742680107973,
      "measurement_steps": 351.0,
      "row": 48,
      "row_first_meas_ms": 30241.1761249532,
      "row_last_ms": 126018.70558294468,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 18799.37774996506,
      "decode_interval_ms_mean": 272.87045417952504,
      "decode_interval_ms_std": 12.417611658325377,
      "decode_interval_rel_std": 0.045507351448741566,
      "decode_tok_s_warm": 3.664742681675924,
      "measurement_steps": 351.0,
      "row": 49,
      "row_first_meas_ms": 30241.176249983255,
      "row_last_ms": 126018.70566699654,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 18799.377832969185,
      "decode_interval_ms_mean": 272.8704542990912,
      "decode_interval_ms_std": 12.417611594210506,
      "decode_interval_rel_std": 0.045507351193836684,
      "decode_tok_s_warm": 3.6647426800701104,
      "measurement_steps": 351.0,
      "row": 50,
      "row_first_meas_ms": 30241.17633298738,
      "row_last_ms": 126018.70579196839,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 18799.37791696284,
      "decode_interval_ms_mean": 272.870454416004,
      "decode_interval_ms_std": 12.41761390821719,
      "decode_interval_rel_std": 0.04550735965457787,
      "decode_tok_s_warm": 3.6647426784999317,
      "measurement_steps": 351.0,
      "row": 51,
      "row_first_meas_ms": 30241.176416981034,
      "row_last_ms": 126018.70591699844,
      "warmup_steps_used": 32.0
    }
  ],
  "seed": 1,
  "warmup_min_steps": 32,
  "warmup_rel_std_threshold": 0.05,
  "warmup_rolling_window": 16
}
```
