# silica-mlx bench report

Generated: 2026-05-04T12:01:43

Scenarios: total=1 Runs: total=1 ok=1 skipped=0 failed=0

## Results

| id | codec | runs | ok | skipped | failed | ttft_ms | decode_tok_s | resident_mb | peak_mb | wall_s | tokens | vqbench_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3.5-27b-warm-decode-b52 |  | 1 | 1 | 0 | 0 | 15818.5 | 193.6 |  | 35518.3 | 133.249 | 19968 |  |

## Scenario details

### `qwen3.5-27b-warm-decode-b52` (seed=2)

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
  "aggregate_overlap_window_ms": 94007.26320798276,
  "codec_id": null,
  "decode_tok_s_warm_aggregate": 193.62333695141922,
  "decode_tok_s_warm_per_row_mean": 3.733752989661305,
  "measurement_steps_min": 64,
  "rows": [
    {
      "cold_ttft_ms": 15818.510792043526,
      "decode_interval_ms_mean": 267.8271066011036,
      "decode_interval_ms_std": 17.449378427944975,
      "decode_interval_rel_std": 0.06515165193467043,
      "decode_tok_s_warm": 3.733752018944745,
      "measurement_steps": 351.0,
      "row": 0,
      "row_first_meas_ms": 36922.24350001197,
      "row_last_ms": 130929.55791699933,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 15818.517959036399,
      "decode_interval_ms_mean": 267.827028606818,
      "decode_interval_ms_std": 17.449506605611386,
      "decode_interval_rel_std": 0.06515214949133473,
      "decode_tok_s_warm": 3.733753106255921,
      "measurement_steps": 351.0,
      "row": 1,
      "row_first_meas_ms": 36922.28395899292,
      "row_last_ms": 130929.57099998603,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 15818.518750020303,
      "decode_interval_ms_mean": 267.8270319316197,
      "decode_interval_ms_std": 17.449511175280175,
      "decode_interval_rel_std": 0.06515216574455152,
      "decode_tok_s_warm": 3.733753059905153,
      "measurement_steps": 351.0,
      "row": 2,
      "row_first_meas_ms": 36922.28445899673,
      "row_last_ms": 130929.57266699523,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 15818.51904198993,
      "decode_interval_ms_mean": 267.8270326438758,
      "decode_interval_ms_std": 17.449513365019694,
      "decode_interval_rel_std": 0.0651521737472332,
      "decode_tok_s_warm": 3.733753049975653,
      "measurement_steps": 351.0,
      "row": 3,
      "row_first_meas_ms": 36922.28466703091,
      "row_last_ms": 130929.57312503131,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 15818.519167019986,
      "decode_interval_ms_mean": 267.8270326438758,
      "decode_interval_ms_std": 17.449507410224037,
      "decode_interval_rel_std": 0.06515215151349675,
      "decode_tok_s_warm": 3.733753049975653,
      "measurement_steps": 351.0,
      "row": 4,
      "row_first_meas_ms": 36922.28491703281,
      "row_last_ms": 130929.57337503321,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 15818.519291991834,
      "decode_interval_ms_mean": 267.8270341879542,
      "decode_interval_ms_std": 17.449509165161675,
      "decode_interval_rel_std": 0.06515215769038481,
      "decode_tok_s_warm": 3.733753028449792,
      "measurement_steps": 351.0,
      "row": 5,
      "row_first_meas_ms": 36922.285000036936,
      "row_last_ms": 130929.57400000887,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 15818.519374995958,
      "decode_interval_ms_mean": 267.82703442443324,
      "decode_interval_ms_std": 17.449627702497555,
      "decode_interval_rel_std": 0.06515260022199486,
      "decode_tok_s_warm": 3.733753025153059,
      "measurement_steps": 351.0,
      "row": 6,
      "row_first_meas_ms": 36922.28508403059,
      "row_last_ms": 130929.57416700665,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 15818.519458989613,
      "decode_interval_ms_mean": 267.82703442459905,
      "decode_interval_ms_std": 17.449618896783715,
      "decode_interval_rel_std": 0.06515256734359384,
      "decode_tok_s_warm": 3.7337530251507474,
      "measurement_steps": 351.0,
      "row": 7,
      "row_first_meas_ms": 36922.28520900244,
      "row_last_ms": 130929.5742920367,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 15818.519584019668,
      "decode_interval_ms_mean": 267.8270345441652,
      "decode_interval_ms_std": 17.449616037451708,
      "decode_interval_rel_std": 0.06515255663846822,
      "decode_tok_s_warm": 3.733753023483886,
      "measurement_steps": 351.0,
      "row": 8,
      "row_first_meas_ms": 36922.285334032495,
      "row_last_ms": 130929.57445903448,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 15818.519667023793,
      "decode_interval_ms_mean": 267.8270347806442,
      "decode_interval_ms_std": 17.44961320620254,
      "decode_interval_rel_std": 0.06515254600975635,
      "decode_tok_s_warm": 3.733753020187153,
      "measurement_steps": 351.0,
      "row": 9,
      "row_first_meas_ms": 36922.28541703662,
      "row_last_ms": 130929.57462504273,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 15818.51979199564,
      "decode_interval_ms_mean": 267.8270349003762,
      "decode_interval_ms_std": 17.449614816951915,
      "decode_interval_rel_std": 0.06515255199477028,
      "decode_tok_s_warm": 3.73375301851798,
      "measurement_steps": 351.0,
      "row": 10,
      "row_first_meas_ms": 36922.28554200847,
      "row_last_ms": 130929.57479204051,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 15818.519874999765,
      "decode_interval_ms_mean": 267.82703501994234,
      "decode_interval_ms_std": 17.449612472144203,
      "decode_interval_rel_std": 0.06515254321075133,
      "decode_tok_s_warm": 3.733753016851119,
      "measurement_steps": 351.0,
      "row": 11,
      "row_first_meas_ms": 36922.28566703852,
      "row_last_ms": 130929.57495903829,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 15818.51995899342,
      "decode_interval_ms_mean": 267.82703501994234,
      "decode_interval_ms_std": 17.449611443729612,
      "decode_interval_rel_std": 0.06515253937090525,
      "decode_tok_s_warm": 3.733753016851119,
      "measurement_steps": 351.0,
      "row": 12,
      "row_first_meas_ms": 36922.28579201037,
      "row_last_ms": 130929.57508401014,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 15818.520041997544,
      "decode_interval_ms_mean": 267.8270358519305,
      "decode_interval_ms_std": 17.44961141220258,
      "decode_interval_rel_std": 0.06515253905079875,
      "decode_tok_s_warm": 3.733753005252446,
      "measurement_steps": 351.0,
      "row": 13,
      "row_first_meas_ms": 36922.285875014495,
      "row_last_ms": 130929.5754590421,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 15818.520125001669,
      "decode_interval_ms_mean": 267.8270359686775,
      "decode_interval_ms_std": 17.44961000726811,
      "decode_interval_rel_std": 0.06515253377672017,
      "decode_tok_s_warm": 3.7337530036248863,
      "measurement_steps": 351.0,
      "row": 14,
      "row_first_meas_ms": 36922.28595900815,
      "row_last_ms": 130929.57558401395,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 15818.52033402538,
      "decode_interval_ms_mean": 267.8270360854245,
      "decode_interval_ms_std": 17.449612492600263,
      "decode_interval_rel_std": 0.06515254302793629,
      "decode_tok_s_warm": 3.733753001997327,
      "measurement_steps": 351.0,
      "row": 15,
      "row_first_meas_ms": 36922.28620901005,
      "row_last_ms": 130929.57587499404,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 15818.520417029504,
      "decode_interval_ms_mean": 267.8270360854245,
      "decode_interval_ms_std": 17.44961191438681,
      "decode_interval_rel_std": 0.06515254086903008,
      "decode_tok_s_warm": 3.733753001997327,
      "measurement_steps": 351.0,
      "row": 16,
      "row_first_meas_ms": 36922.28633404011,
      "row_last_ms": 130929.5760000241,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 15818.520500033628,
      "decode_interval_ms_mean": 267.8270369174126,
      "decode_interval_ms_std": 17.44961015307994,
      "decode_interval_rel_std": 0.06515253409035295,
      "decode_tok_s_warm": 3.7337529903986537,
      "measurement_steps": 351.0,
      "row": 17,
      "row_first_meas_ms": 36922.28645901196,
      "row_last_ms": 130929.57641702378,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 15818.520584027283,
      "decode_interval_ms_mean": 267.82703703697877,
      "decode_interval_ms_std": 17.44973740506354,
      "decode_interval_rel_std": 0.06515300918874095,
      "decode_tok_s_warm": 3.733752988731793,
      "measurement_steps": 351.0,
      "row": 18,
      "row_first_meas_ms": 36922.28654201608,
      "row_last_ms": 130929.57654199563,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 15818.520667031407,
      "decode_interval_ms_mean": 267.82704119094944,
      "decode_interval_ms_std": 17.449739245269424,
      "decode_interval_rel_std": 0.06515301504909839,
      "decode_tok_s_warm": 3.7337529308216566,
      "measurement_steps": 351.0,
      "row": 19,
      "row_first_meas_ms": 36922.28666698793,
      "row_last_ms": 130929.57812501118,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 15818.520834029187,
      "decode_interval_ms_mean": 267.82704143024756,
      "decode_interval_ms_std": 17.449739529887022,
      "decode_interval_rel_std": 0.06515301605357726,
      "decode_tok_s_warm": 3.733752927485623,
      "measurement_steps": 351.0,
      "row": 20,
      "row_first_meas_ms": 36922.28687502211,
      "row_last_ms": 130929.57841703901,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 15818.521042005159,
      "decode_interval_ms_mean": 267.8270416667266,
      "decode_interval_ms_std": 17.449735679565883,
      "decode_interval_rel_std": 0.06515300161990231,
      "decode_tok_s_warm": 3.73375292418889,
      "measurement_steps": 351.0,
      "row": 21,
      "row_first_meas_ms": 36922.28708398761,
      "row_last_ms": 130929.57870900864,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 15818.521250039339,
      "decode_interval_ms_mean": 267.82704261546166,
      "decode_interval_ms_std": 17.449739038283916,
      "decode_interval_rel_std": 0.06515301392973132,
      "decode_tok_s_warm": 3.7337529109626586,
      "measurement_steps": 351.0,
      "row": 22,
      "row_first_meas_ms": 36922.287333989516,
      "row_last_ms": 130929.57929201657,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 15818.521459004842,
      "decode_interval_ms_mean": 267.8270428517748,
      "decode_interval_ms_std": 17.449740208087395,
      "decode_interval_rel_std": 0.06515301824000168,
      "decode_tok_s_warm": 3.733752907668238,
      "measurement_steps": 351.0,
      "row": 23,
      "row_first_meas_ms": 36922.28745901957,
      "row_last_ms": 130929.57949999254,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 15818.52162501309,
      "decode_interval_ms_mean": 267.82704297150684,
      "decode_interval_ms_std": 17.449739906126915,
      "decode_interval_rel_std": 0.06515301708342922,
      "decode_tok_s_warm": 3.7337529059990646,
      "measurement_steps": 351.0,
      "row": 24,
      "row_first_meas_ms": 36922.287542023696,
      "row_last_ms": 130929.5796250226,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 15818.521709006745,
      "decode_interval_ms_mean": 267.8270428517748,
      "decode_interval_ms_std": 17.449738870928734,
      "decode_interval_rel_std": 0.06515301324738164,
      "decode_tok_s_warm": 3.733752907668238,
      "measurement_steps": 351.0,
      "row": 25,
      "row_first_meas_ms": 36922.287709021475,
      "row_last_ms": 130929.57974999445,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 15818.521834036801,
      "decode_interval_ms_mean": 267.8270328803548,
      "decode_interval_ms_std": 17.449748278702668,
      "decode_interval_rel_std": 0.06515305079938633,
      "decode_tok_s_warm": 3.73375304667892,
      "measurement_steps": 351.0,
      "row": 26,
      "row_first_meas_ms": 36922.291334019974,
      "row_last_ms": 130929.5798750245,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 15818.521959008649,
      "decode_interval_ms_mean": 267.8270327636078,
      "decode_interval_ms_std": 17.449749673404618,
      "decode_interval_rel_std": 0.06515305603525949,
      "decode_tok_s_warm": 3.7337530483064794,
      "measurement_steps": 351.0,
      "row": 27,
      "row_first_meas_ms": 36922.291541995946,
      "row_last_ms": 130929.58004202228,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 15818.522042012773,
      "decode_interval_ms_mean": 267.8270327636078,
      "decode_interval_ms_std": 17.449760082760285,
      "decode_interval_rel_std": 0.06515309490122294,
      "decode_tok_s_warm": 3.7337530483064794,
      "measurement_steps": 351.0,
      "row": 28,
      "row_first_meas_ms": 36922.29179199785,
      "row_last_ms": 130929.58029202418,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 15818.522334040608,
      "decode_interval_ms_mean": 267.827032646695,
      "decode_interval_ms_std": 17.449758622807487,
      "decode_interval_rel_std": 0.06515308947856059,
      "decode_tok_s_warm": 3.7337530499363507,
      "measurement_steps": 351.0,
      "row": 29,
      "row_first_meas_ms": 36922.29200003203,
      "row_last_ms": 130929.58045902196,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 15818.522416986525,
      "decode_interval_ms_mean": 267.82703252696297,
      "decode_interval_ms_std": 17.449756524691548,
      "decode_interval_rel_std": 0.06515308167384047,
      "decode_tok_s_warm": 3.7337530516055244,
      "measurement_steps": 351.0,
      "row": 30,
      "row_first_meas_ms": 36922.29216702981,
      "row_last_ms": 130929.58058399381,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 15818.52254201658,
      "decode_interval_ms_mean": 267.8270324073968,
      "decode_interval_ms_std": 17.449755857587306,
      "decode_interval_rel_std": 0.06515307921212429,
      "decode_tok_s_warm": 3.7337530532723857,
      "measurement_steps": 351.0,
      "row": 31,
      "row_first_meas_ms": 36922.29237500578,
      "row_last_ms": 130929.58075000206,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 15818.522625020705,
      "decode_interval_ms_mean": 267.82703205135164,
      "decode_interval_ms_std": 17.449754802096614,
      "decode_interval_rel_std": 0.06515307535779621,
      "decode_tok_s_warm": 3.73375305823598,
      "measurement_steps": 351.0,
      "row": 32,
      "row_first_meas_ms": 36922.292625007685,
      "row_last_ms": 130929.58087503212,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 15818.52270901436,
      "decode_interval_ms_mean": 267.8270320511858,
      "decode_interval_ms_std": 17.449753706850725,
      "decode_interval_rel_std": 0.06515307126845886,
      "decode_tok_s_warm": 3.7337530582382916,
      "measurement_steps": 351.0,
      "row": 33,
      "row_first_meas_ms": 36922.29275003774,
      "row_last_ms": 130929.58100000396,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 15818.522792018484,
      "decode_interval_ms_mean": 267.82703216809864,
      "decode_interval_ms_std": 17.44975229588136,
      "decode_interval_rel_std": 0.06515306597180683,
      "decode_tok_s_warm": 3.7337530566084203,
      "measurement_steps": 351.0,
      "row": 34,
      "row_first_meas_ms": 36922.292834031396,
      "row_last_ms": 130929.58112503402,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 15818.522875022609,
      "decode_interval_ms_mean": 267.82703216809864,
      "decode_interval_ms_std": 17.449760657297745,
      "decode_interval_rel_std": 0.06515309719127081,
      "decode_tok_s_warm": 3.7337530566084203,
      "measurement_steps": 351.0,
      "row": 35,
      "row_first_meas_ms": 36922.29295900324,
      "row_last_ms": 130929.58125000587,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 15818.522959016263,
      "decode_interval_ms_mean": 267.82703299992096,
      "decode_interval_ms_std": 17.449759676288497,
      "decode_interval_rel_std": 0.06515309332607082,
      "decode_tok_s_warm": 3.7337530450120586,
      "measurement_steps": 351.0,
      "row": 36,
      "row_first_meas_ms": 36922.2930840333,
      "row_last_ms": 130929.58166700555,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 15818.523042020388,
      "decode_interval_ms_mean": 267.8270341879542,
      "decode_interval_ms_std": 17.44983417675376,
      "decode_interval_rel_std": 0.06515337120340103,
      "decode_tok_s_warm": 3.733753028449792,
      "measurement_steps": 351.0,
      "row": 37,
      "row_first_meas_ms": 36922.293167037424,
      "row_last_ms": 130929.58216700936,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 15818.523125024512,
      "decode_interval_ms_mean": 267.8270341881201,
      "decode_interval_ms_std": 17.449834747635816,
      "decode_interval_rel_std": 0.06515337333489328,
      "decode_tok_s_warm": 3.7337530284474796,
      "measurement_steps": 351.0,
      "row": 38,
      "row_first_meas_ms": 36922.29329200927,
      "row_last_ms": 130929.58229203941,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 15818.523209018167,
      "decode_interval_ms_mean": 267.82703430768623,
      "decode_interval_ms_std": 17.449833493318295,
      "decode_interval_rel_std": 0.06515336862249499,
      "decode_tok_s_warm": 3.7337530267806183,
      "measurement_steps": 351.0,
      "row": 39,
      "row_first_meas_ms": 36922.293375013396,
      "row_last_ms": 130929.58241701126,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 15818.523292022292,
      "decode_interval_ms_mean": 267.8270347806442,
      "decode_interval_ms_std": 17.44983417771665,
      "decode_interval_rel_std": 0.06515337106281455,
      "decode_tok_s_warm": 3.733753020187153,
      "measurement_steps": 351.0,
      "row": 40,
      "row_first_meas_ms": 36922.29345900705,
      "row_last_ms": 130929.58266701316,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 15818.523375026416,
      "decode_interval_ms_mean": 267.82703513668935,
      "decode_interval_ms_std": 17.449833696808767,
      "decode_interval_rel_std": 0.0651533691806094,
      "decode_tok_s_warm": 3.7337530152235594,
      "measurement_steps": 351.0,
      "row": 41,
      "row_first_meas_ms": 36922.29358403711,
      "row_last_ms": 130929.58291701507,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 15818.52345902007,
      "decode_interval_ms_mean": 267.82703632472266,
      "decode_interval_ms_std": 17.449833374029332,
      "decode_interval_rel_std": 0.06515336768642191,
      "decode_tok_s_warm": 3.733752998661292,
      "measurement_steps": 351.0,
      "row": 42,
      "row_first_meas_ms": 36922.293709008954,
      "row_last_ms": 130929.5834589866,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 15818.523542024195,
      "decode_interval_ms_mean": 267.8270366807678,
      "decode_interval_ms_std": 17.44983062779746,
      "decode_interval_rel_std": 0.06515335734605655,
      "decode_tok_s_warm": 3.7337529936976988,
      "measurement_steps": 351.0,
      "row": 43,
      "row_first_meas_ms": 36922.29383403901,
      "row_last_ms": 130929.5837089885,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 15818.52362502832,
      "decode_interval_ms_mean": 267.8270367976806,
      "decode_interval_ms_std": 17.449829572744097,
      "decode_interval_rel_std": 0.06515335337830692,
      "decode_tok_s_warm": 3.7337529920678274,
      "measurement_steps": 351.0,
      "row": 44,
      "row_first_meas_ms": 36922.29395901086,
      "row_last_ms": 130929.58387499675,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 15818.523709021974,
      "decode_interval_ms_mean": 267.8270369174126,
      "decode_interval_ms_std": 17.449829927352873,
      "decode_interval_rel_std": 0.06515335467320171,
      "decode_tok_s_warm": 3.7337529903986537,
      "measurement_steps": 351.0,
      "row": 45,
      "row_first_meas_ms": 36922.29404201498,
      "row_last_ms": 130929.5840000268,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 15818.523792026099,
      "decode_interval_ms_mean": 267.8270369174126,
      "decode_interval_ms_std": 17.44982915703914,
      "decode_interval_rel_std": 0.06515335179704052,
      "decode_tok_s_warm": 3.7337529903986537,
      "measurement_steps": 351.0,
      "row": 46,
      "row_first_meas_ms": 36922.29416698683,
      "row_last_ms": 130929.58412499866,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 15818.523875030223,
      "decode_interval_ms_mean": 267.8270370371446,
      "decode_interval_ms_std": 17.44982771977537,
      "decode_interval_rel_std": 0.06515334640152584,
      "decode_tok_s_warm": 3.733752988729481,
      "measurement_steps": 351.0,
      "row": 47,
      "row_first_meas_ms": 36922.294249990955,
      "row_last_ms": 130929.58425002871,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 15818.523959023878,
      "decode_interval_ms_mean": 267.82703703697877,
      "decode_interval_ms_std": 17.449827520332985,
      "decode_interval_rel_std": 0.06515334565689757,
      "decode_tok_s_warm": 3.733752988731793,
      "measurement_steps": 351.0,
      "row": 48,
      "row_first_meas_ms": 36922.29437502101,
      "row_last_ms": 130929.58437500056,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 15818.524042028002,
      "decode_interval_ms_mean": 267.8270371538916,
      "decode_interval_ms_std": 17.449826330084264,
      "decode_interval_rel_std": 0.06515334118436188,
      "decode_tok_s_warm": 3.7337529871019215,
      "measurement_steps": 351.0,
      "row": 49,
      "row_first_meas_ms": 36922.294459014665,
      "row_last_ms": 130929.58450003061,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 15818.524125032127,
      "decode_interval_ms_mean": 267.8270371538916,
      "decode_interval_ms_std": 17.449826624057426,
      "decode_interval_rel_std": 0.06515334228198505,
      "decode_tok_s_warm": 3.7337529871019215,
      "measurement_steps": 351.0,
      "row": 50,
      "row_first_meas_ms": 36922.29458398651,
      "row_last_ms": 130929.58462500246,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 15818.524209025782,
      "decode_interval_ms_mean": 267.8270371538916,
      "decode_interval_ms_std": 17.449825754275537,
      "decode_interval_rel_std": 0.06515333903443432,
      "decode_tok_s_warm": 3.7337529871019215,
      "measurement_steps": 351.0,
      "row": 51,
      "row_first_meas_ms": 36922.29470901657,
      "row_last_ms": 130929.58475003252,
      "warmup_steps_used": 32.0
    }
  ],
  "seed": 2,
  "warmup_min_steps": 32,
  "warmup_rel_std_threshold": 0.05,
  "warmup_rolling_window": 16
}
```
