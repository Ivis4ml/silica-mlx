# silica-mlx bench report

Generated: 2026-05-04T11:59:22

Scenarios: total=1 Runs: total=1 ok=1 skipped=0 failed=0

## Results

| id | codec | runs | ok | skipped | failed | ttft_ms | decode_tok_s | resident_mb | peak_mb | wall_s | tokens | vqbench_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3.5-27b-warm-decode-b52 |  | 1 | 1 | 0 | 0 | 16917.2 | 201.3 |  | 35518.3 | 129.524 | 19968 |  |

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
  "aggregate_overlap_window_ms": 90437.883333012,
  "codec_id": null,
  "decode_tok_s_warm_aggregate": 201.26521463329996,
  "decode_tok_s_warm_per_row_mean": 3.881115961799031,
  "measurement_steps_min": 64,
  "rows": [
    {
      "cold_ttft_ms": 16917.181917000562,
      "decode_interval_ms_mean": 257.6578574301874,
      "decode_interval_ms_std": 11.180126546487005,
      "decode_interval_rel_std": 0.04339136658976631,
      "decode_tok_s_warm": 3.8811158719308643,
      "measurement_steps": 351.0,
      "row": 0,
      "row_first_meas_ms": 36745.6901249825,
      "row_last_ms": 127183.59808297828,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16917.419333010912,
      "decode_interval_ms_mean": 257.6578574330066,
      "decode_interval_ms_std": 11.180148860141326,
      "decode_interval_rel_std": 0.043391453191169486,
      "decode_tok_s_warm": 3.8811158718883982,
      "measurement_steps": 351.0,
      "row": 1,
      "row_first_meas_ms": 36745.700958010275,
      "row_last_ms": 127183.60891699558,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16917.420333018526,
      "decode_interval_ms_mean": 257.6578584986545,
      "decode_interval_ms_std": 11.180156585112,
      "decode_interval_rel_std": 0.04339148299321281,
      "decode_tok_s_warm": 3.8811158558364798,
      "measurement_steps": 351.0,
      "row": 2,
      "row_first_meas_ms": 36745.70116697578,
      "row_last_ms": 127183.60950000351,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16917.420624988154,
      "decode_interval_ms_mean": 257.6578586182207,
      "decode_interval_ms_std": 11.180159289249316,
      "decode_interval_rel_std": 0.04339149346814719,
      "decode_tok_s_warm": 3.8811158540354467,
      "measurement_steps": 351.0,
      "row": 3,
      "row_first_meas_ms": 36745.70141697768,
      "row_last_ms": 127183.60979197314,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16917.420791985933,
      "decode_interval_ms_mean": 257.65785873795267,
      "decode_interval_ms_std": 11.18016134003647,
      "decode_interval_rel_std": 0.04339150140732598,
      "decode_tok_s_warm": 3.881115852231917,
      "measurement_steps": 351.0,
      "row": 4,
      "row_first_meas_ms": 36745.70158298593,
      "row_last_ms": 127183.61000000732,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16917.42091701599,
      "decode_interval_ms_mean": 257.6578589744317,
      "decode_interval_ms_std": 11.180181023774864,
      "decode_interval_rel_std": 0.04339157776237019,
      "decode_tok_s_warm": 3.881115848669819,
      "measurement_steps": 351.0,
      "row": 5,
      "row_first_meas_ms": 36745.70174998371,
      "row_last_ms": 127183.61025000922,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16917.421041987836,
      "decode_interval_ms_mean": 257.6578589744317,
      "decode_interval_ms_std": 11.180194896446606,
      "decode_interval_rel_std": 0.04339163160381635,
      "decode_tok_s_warm": 3.881115848669819,
      "measurement_steps": 351.0,
      "row": 6,
      "row_first_meas_ms": 36745.70191698149,
      "row_last_ms": 127183.610417007,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16917.421167017892,
      "decode_interval_ms_mean": 257.6578589744317,
      "decode_interval_ms_std": 11.180203371409297,
      "decode_interval_rel_std": 0.043391664496128364,
      "decode_tok_s_warm": 3.881115848669819,
      "measurement_steps": 351.0,
      "row": 7,
      "row_first_meas_ms": 36745.70208298974,
      "row_last_ms": 127183.61058301525,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16917.42124996381,
      "decode_interval_ms_mean": 257.65785885469967,
      "decode_interval_ms_std": 11.18020495574978,
      "decode_interval_rel_std": 0.04339167066530116,
      "decode_tok_s_warm": 3.88111585047335,
      "measurement_steps": 351.0,
      "row": 8,
      "row_first_meas_ms": 36745.70224998752,
      "row_last_ms": 127183.6107079871,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16917.421374993864,
      "decode_interval_ms_mean": 257.6578589742658,
      "decode_interval_ms_std": 11.180206341799261,
      "decode_interval_rel_std": 0.04339167602458387,
      "decode_tok_s_warm": 3.8811158486723176,
      "measurement_steps": 351.0,
      "row": 9,
      "row_first_meas_ms": 36745.70237501757,
      "row_last_ms": 127183.61087498488,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16917.42145799799,
      "decode_interval_ms_mean": 257.65785885469967,
      "decode_interval_ms_std": 11.180204518686734,
      "decode_interval_rel_std": 0.043391668969008854,
      "decode_tok_s_warm": 3.88111585047335,
      "measurement_steps": 351.0,
      "row": 10,
      "row_first_meas_ms": 36745.70249998942,
      "row_last_ms": 127183.610957989,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16917.421582969837,
      "decode_interval_ms_mean": 257.6578589742658,
      "decode_interval_ms_std": 11.180208290078234,
      "decode_interval_rel_std": 0.04339168358608026,
      "decode_tok_s_warm": 3.8811158486723176,
      "measurement_steps": 351.0,
      "row": 11,
      "row_first_meas_ms": 36745.702625019476,
      "row_last_ms": 127183.61112498678,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16917.42166696349,
      "decode_interval_ms_mean": 257.65785885469967,
      "decode_interval_ms_std": 11.180212826927969,
      "decode_interval_rel_std": 0.0433917012142556,
      "decode_tok_s_warm": 3.88111585047335,
      "measurement_steps": 351.0,
      "row": 12,
      "row_first_meas_ms": 36745.702749991324,
      "row_last_ms": 127183.6112079909,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16917.422708007507,
      "decode_interval_ms_mean": 257.65785885469967,
      "decode_interval_ms_std": 11.180216308398748,
      "decode_interval_rel_std": 0.04339171472624702,
      "decode_tok_s_warm": 3.88111585047335,
      "measurement_steps": 351.0,
      "row": 13,
      "row_first_meas_ms": 36745.70287496317,
      "row_last_ms": 127183.61133296276,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16917.423582985066,
      "decode_interval_ms_mean": 257.65785885469967,
      "decode_interval_ms_std": 11.180221426554324,
      "decode_interval_rel_std": 0.04339173459040175,
      "decode_tok_s_warm": 3.88111585047335,
      "measurement_steps": 351.0,
      "row": 14,
      "row_first_meas_ms": 36745.70299999323,
      "row_last_ms": 127183.61145799281,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16917.63791698031,
      "decode_interval_ms_mean": 257.6578589742658,
      "decode_interval_ms_std": 11.180225061567945,
      "decode_interval_rel_std": 0.04339174867817479,
      "decode_tok_s_warm": 3.8811158486723176,
      "measurement_steps": 351.0,
      "row": 15,
      "row_first_meas_ms": 36745.70324999513,
      "row_last_ms": 127183.61174996244,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16917.823916999623,
      "decode_interval_ms_mean": 257.65785885469967,
      "decode_interval_ms_std": 11.18040397391666,
      "decode_interval_rel_std": 0.04339244307786318,
      "decode_tok_s_warm": 3.88111585047335,
      "measurement_steps": 351.0,
      "row": 16,
      "row_first_meas_ms": 36745.703499997035,
      "row_last_ms": 127183.61195799662,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16917.82441700343,
      "decode_interval_ms_mean": 257.65785885469967,
      "decode_interval_ms_std": 11.180409974169544,
      "decode_interval_rel_std": 0.04339246636553975,
      "decode_tok_s_warm": 3.88111585047335,
      "measurement_steps": 351.0,
      "row": 17,
      "row_first_meas_ms": 36745.70374999894,
      "row_last_ms": 127183.61220799852,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16917.824624979403,
      "decode_interval_ms_mean": 257.65785945020883,
      "decode_interval_ms_std": 11.180406397336377,
      "decode_interval_rel_std": 0.04339245238314548,
      "decode_tok_s_warm": 3.8811158415031595,
      "measurement_steps": 351.0,
      "row": 18,
      "row_first_meas_ms": 36745.703874970786,
      "row_last_ms": 127183.61254199408,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16917.824833013583,
      "decode_interval_ms_mean": 257.65786111103426,
      "decode_interval_ms_std": 11.180414669039086,
      "decode_interval_rel_std": 0.0433924842068802,
      "decode_tok_s_warm": 3.881115816486046,
      "measurement_steps": 351.0,
      "row": 19,
      "row_first_meas_ms": 36745.70408300497,
      "row_last_ms": 127183.61333297798,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16917.825041979086,
      "decode_interval_ms_mean": 257.6578611112001,
      "decode_interval_ms_std": 11.18070878981842,
      "decode_interval_rel_std": 0.0433936257236609,
      "decode_tok_s_warm": 3.881115816483548,
      "measurement_steps": 351.0,
      "row": 20,
      "row_first_meas_ms": 36745.70429197047,
      "row_last_ms": 127183.6135420017,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16917.825250013266,
      "decode_interval_ms_mean": 257.6578611112001,
      "decode_interval_ms_std": 11.180713486238762,
      "decode_interval_rel_std": 0.043393643951012174,
      "decode_tok_s_warm": 3.881115816483548,
      "measurement_steps": 351.0,
      "row": 21,
      "row_first_meas_ms": 36745.70454197237,
      "row_last_ms": 127183.6137920036,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16917.8260419867,
      "decode_interval_ms_mean": 257.65786123076623,
      "decode_interval_ms_std": 11.18071984295548,
      "decode_interval_rel_std": 0.04339366860202913,
      "decode_tok_s_warm": 3.8811158146825164,
      "measurement_steps": 351.0,
      "row": 22,
      "row_first_meas_ms": 36745.70470798062,
      "row_last_ms": 127183.61399997957,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16917.826291988604,
      "decode_interval_ms_mean": 257.65786111103426,
      "decode_interval_ms_std": 11.180723509254976,
      "decode_interval_rel_std": 0.04339368285152686,
      "decode_tok_s_warm": 3.881115816486046,
      "measurement_steps": 351.0,
      "row": 23,
      "row_first_meas_ms": 36745.70491700433,
      "row_last_ms": 127183.61416697735,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16917.82637499273,
      "decode_interval_ms_mean": 257.65786099428726,
      "decode_interval_ms_std": 11.180998977086631,
      "decode_interval_rel_std": 0.04339475199374776,
      "decode_tok_s_warm": 3.8811158182446133,
      "measurement_steps": 351.0,
      "row": 24,
      "row_first_meas_ms": 36745.70508301258,
      "row_last_ms": 127183.6142920074,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16917.826499964576,
      "decode_interval_ms_mean": 257.65786135049825,
      "decode_interval_ms_std": 11.181612370902979,
      "decode_interval_rel_std": 0.043397132586194834,
      "decode_tok_s_warm": 3.8811158128789858,
      "measurement_steps": 351.0,
      "row": 25,
      "row_first_meas_ms": 36745.70520798443,
      "row_last_ms": 127183.61454200931,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16917.82674996648,
      "decode_interval_ms_mean": 257.65784342452866,
      "decode_interval_ms_std": 11.182551145573212,
      "decode_interval_rel_std": 0.043400779098924376,
      "decode_tok_s_warm": 3.881116082898959,
      "measurement_steps": 351.0,
      "row": 26,
      "row_first_meas_ms": 36745.71162497159,
      "row_last_ms": 127183.61466698116,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16917.826874996535,
      "decode_interval_ms_mean": 257.6578433049625,
      "decode_interval_ms_std": 11.18290355916617,
      "decode_interval_rel_std": 0.04340214687712861,
      "decode_tok_s_warm": 3.8811160846999915,
      "measurement_steps": 351.0,
      "row": 27,
      "row_first_meas_ms": 36745.71179196937,
      "row_last_ms": 127183.61479201121,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16917.826999968383,
      "decode_interval_ms_mean": 257.6578433047967,
      "decode_interval_ms_std": 11.182915481113648,
      "decode_interval_rel_std": 0.04340219314761865,
      "decode_tok_s_warm": 3.881116084702489,
      "measurement_steps": 351.0,
      "row": 28,
      "row_first_meas_ms": 36745.711916999426,
      "row_last_ms": 127183.61491698306,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16917.827082972508,
      "decode_interval_ms_mean": 257.6578428320045,
      "decode_interval_ms_std": 11.182923771062075,
      "decode_interval_rel_std": 0.04340222540151224,
      "decode_tok_s_warm": 3.881116091824187,
      "measurement_steps": 351.0,
      "row": 29,
      "row_first_meas_ms": 36745.71220797952,
      "row_last_ms": 127183.61504201312,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16917.827166966163,
      "decode_interval_ms_mean": 257.6578428318387,
      "decode_interval_ms_std": 11.18292374790764,
      "decode_interval_rel_std": 0.043402225311675126,
      "decode_tok_s_warm": 3.8811160918266845,
      "measurement_steps": 351.0,
      "row": 30,
      "row_first_meas_ms": 36745.71233300958,
      "row_last_ms": 127183.61516698496,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16917.82729199622,
      "decode_interval_ms_mean": 257.6578428320045,
      "decode_interval_ms_std": 11.182922669235849,
      "decode_interval_rel_std": 0.04340222112519674,
      "decode_tok_s_warm": 3.881116091824187,
      "measurement_steps": 351.0,
      "row": 31,
      "row_first_meas_ms": 36745.712457981426,
      "row_last_ms": 127183.61529201502,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16917.827375000343,
      "decode_interval_ms_mean": 257.6578429487515,
      "decode_interval_ms_std": 11.182924727445684,
      "decode_interval_rel_std": 0.04340222909368213,
      "decode_tok_s_warm": 3.88111609006562,
      "measurement_steps": 351.0,
      "row": 32,
      "row_first_meas_ms": 36745.712624979205,
      "row_last_ms": 127183.61549999099,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16917.827458004467,
      "decode_interval_ms_mean": 257.65784342452866,
      "decode_interval_ms_std": 11.182926813977176,
      "decode_interval_rel_std": 0.04340223711160883,
      "decode_tok_s_warm": 3.881116082898959,
      "measurement_steps": 351.0,
      "row": 33,
      "row_first_meas_ms": 36745.71275000926,
      "row_last_ms": 127183.61579201883,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16917.827541998122,
      "decode_interval_ms_mean": 257.65784342452866,
      "decode_interval_ms_std": 11.182930917360869,
      "decode_interval_rel_std": 0.043402253037317276,
      "decode_tok_s_warm": 3.881116082898959,
      "measurement_steps": 351.0,
      "row": 34,
      "row_first_meas_ms": 36745.71287498111,
      "row_last_ms": 127183.61591699068,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16917.827625002246,
      "decode_interval_ms_mean": 257.6578433047967,
      "decode_interval_ms_std": 11.182932129883579,
      "decode_interval_rel_std": 0.04340225776342742,
      "decode_tok_s_warm": 3.881116084702489,
      "measurement_steps": 351.0,
      "row": 35,
      "row_first_meas_ms": 36745.713000011165,
      "row_last_ms": 127183.6159999948,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16917.82770800637,
      "decode_interval_ms_mean": 257.6578435440948,
      "decode_interval_ms_std": 11.182933458964138,
      "decode_interval_rel_std": 0.04340226288143378,
      "decode_tok_s_warm": 3.8811160810979266,
      "measurement_steps": 351.0,
      "row": 36,
      "row_first_meas_ms": 36745.71308301529,
      "row_last_ms": 127183.61616699258,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16917.827792000026,
      "decode_interval_ms_mean": 257.6578435440948,
      "decode_interval_ms_std": 11.182934775380645,
      "decode_interval_rel_std": 0.04340226799059905,
      "decode_tok_s_warm": 3.8811160810979266,
      "measurement_steps": 351.0,
      "row": 37,
      "row_first_meas_ms": 36745.71320798714,
      "row_last_ms": 127183.61629196443,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16917.827916971873,
      "decode_interval_ms_mean": 257.6578436610077,
      "decode_interval_ms_std": 11.18293629555592,
      "decode_interval_rel_std": 0.04340227387088187,
      "decode_tok_s_warm": 3.8811160793368606,
      "measurement_steps": 351.0,
      "row": 38,
      "row_first_meas_ms": 36745.71329198079,
      "row_last_ms": 127183.61641699448,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16917.827999975998,
      "decode_interval_ms_mean": 257.6578436608418,
      "decode_interval_ms_std": 11.182937065654475,
      "decode_interval_rel_std": 0.0434022768597517,
      "decode_tok_s_warm": 3.881116079339359,
      "measurement_steps": 351.0,
      "row": 39,
      "row_first_meas_ms": 36745.71341701085,
      "row_last_ms": 127183.61654196633,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16917.828082980122,
      "decode_interval_ms_mean": 257.6578436610077,
      "decode_interval_ms_std": 11.182939573565529,
      "decode_interval_rel_std": 0.04340228659321767,
      "decode_tok_s_warm": 3.8811160793368606,
      "measurement_steps": 351.0,
      "row": 40,
      "row_first_meas_ms": 36745.713541982695,
      "row_last_ms": 127183.61666699639,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16917.828166973777,
      "decode_interval_ms_mean": 257.6578436608418,
      "decode_interval_ms_std": 11.182941518432957,
      "decode_interval_rel_std": 0.04340229414150186,
      "decode_tok_s_warm": 3.881116079339359,
      "measurement_steps": 351.0,
      "row": 41,
      "row_first_meas_ms": 36745.71366701275,
      "row_last_ms": 127183.61679196823,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16917.8282499779,
      "decode_interval_ms_mean": 257.6578436610077,
      "decode_interval_ms_std": 11.182933638686452,
      "decode_interval_rel_std": 0.043402263559263056,
      "decode_tok_s_warm": 3.8811160793368606,
      "measurement_steps": 351.0,
      "row": 42,
      "row_first_meas_ms": 36745.7137919846,
      "row_last_ms": 127183.61691699829,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16917.828332982026,
      "decode_interval_ms_mean": 257.6578436610077,
      "decode_interval_ms_std": 11.1833076184719,
      "decode_interval_rel_std": 0.04340371501822171,
      "decode_tok_s_warm": 3.8811160793368606,
      "measurement_steps": 351.0,
      "row": 43,
      "row_first_meas_ms": 36745.71387498872,
      "row_last_ms": 127183.61700000241,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16917.82841697568,
      "decode_interval_ms_mean": 257.65784378057384,
      "decode_interval_ms_std": 11.183320542349904,
      "decode_interval_rel_std": 0.043403765157150914,
      "decode_tok_s_warm": 3.8811160775358284,
      "measurement_steps": 351.0,
      "row": 44,
      "row_first_meas_ms": 36745.71400001878,
      "row_last_ms": 127183.6171670002,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16917.828499979805,
      "decode_interval_ms_mean": 257.6578436610077,
      "decode_interval_ms_std": 11.18332619339312,
      "decode_interval_rel_std": 0.043403787109647124,
      "decode_tok_s_warm": 3.8811160793368606,
      "measurement_steps": 351.0,
      "row": 45,
      "row_first_meas_ms": 36745.71412499063,
      "row_last_ms": 127183.61725000432,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16917.82858298393,
      "decode_interval_ms_mean": 257.6578439003058,
      "decode_interval_ms_std": 11.183339439547796,
      "decode_interval_rel_std": 0.043403838479199985,
      "decode_tok_s_warm": 3.881116075732298,
      "measurement_steps": 351.0,
      "row": 46,
      "row_first_meas_ms": 36745.71420799475,
      "row_last_ms": 127183.6174170021,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16917.828666977584,
      "decode_interval_ms_mean": 257.6578440170528,
      "decode_interval_ms_std": 11.183412061991994,
      "decode_interval_rel_std": 0.043404120315668836,
      "decode_tok_s_warm": 3.881116073973731,
      "measurement_steps": 351.0,
      "row": 47,
      "row_first_meas_ms": 36745.714291988406,
      "row_last_ms": 127183.61754197394,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16917.82879200764,
      "decode_interval_ms_mean": 257.65784389732084,
      "decode_interval_ms_std": 11.183309987606723,
      "decode_interval_rel_std": 0.04340372417330086,
      "decode_tok_s_warm": 3.881116075777261,
      "measurement_steps": 351.0,
      "row": 48,
      "row_first_meas_ms": 36745.71441701846,
      "row_last_ms": 127183.61762497807,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16917.828875011764,
      "decode_interval_ms_mean": 257.6578440170528,
      "decode_interval_ms_std": 11.18332311274611,
      "decode_interval_rel_std": 0.04340377509332087,
      "decode_tok_s_warm": 3.881116073973731,
      "measurement_steps": 351.0,
      "row": 49,
      "row_first_meas_ms": 36745.71454199031,
      "row_last_ms": 127183.61779197585,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16917.82895801589,
      "decode_interval_ms_mean": 257.6578440170528,
      "decode_interval_ms_std": 11.183334997954097,
      "decode_interval_rel_std": 0.04340382122119263,
      "decode_tok_s_warm": 3.881116073973731,
      "measurement_steps": 351.0,
      "row": 50,
      "row_first_meas_ms": 36745.714624994434,
      "row_last_ms": 127183.61787497997,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 16917.829042009544,
      "decode_interval_ms_mean": 257.65784413678483,
      "decode_interval_ms_std": 11.183345670873225,
      "decode_interval_rel_std": 0.0434038626238611,
      "decode_tok_s_warm": 3.8811160721702,
      "measurement_steps": 351.0,
      "row": 51,
      "row_first_meas_ms": 36745.71474996628,
      "row_last_ms": 127183.61804197775,
      "warmup_steps_used": 32.0
    }
  ],
  "seed": 1,
  "warmup_min_steps": 32,
  "warmup_rel_std_threshold": 0.05,
  "warmup_rolling_window": 16
}
```
