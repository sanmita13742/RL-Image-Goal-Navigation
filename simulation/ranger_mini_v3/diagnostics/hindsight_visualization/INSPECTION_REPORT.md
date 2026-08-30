# Hindsight Relabeling Visual Inspection Report

- Total inspected: 65
- Geometric examples: 45
- Uniform examples: 20
- Reward=1 examples: 49
- Reward=0 examples: 16
- Boundary examples: 10

## Sample Table

| type      | category          |   current_step |   goal_step |      k |   similarity |   reward |   done |
|:----------|:------------------|---------------:|------------:|-------:|-------------:|---------:|-------:|
| geometric | high_similarity   |          34252 |       34301 |     49 |     0.992123 |        1 |      1 |
| geometric | high_similarity   |          52365 |       52461 |     96 |     0.999064 |        1 |      1 |
| geometric | high_similarity   |          61293 |       61528 |    235 |     0.992273 |        1 |      1 |
| geometric | high_similarity   |          35494 |       35495 |      1 |     0.975141 |        1 |      1 |
| geometric | high_similarity   |          33082 |       33142 |     60 |     0.96393  |        1 |      1 |
| geometric | medium_similarity |          22732 |       22764 |     32 |     0.900766 |        1 |      1 |
| geometric | medium_similarity |          14222 |       14242 |     20 |     0.920863 |        1 |      1 |
| geometric | medium_similarity |          10582 |       10641 |     59 |     0.879929 |        1 |      1 |
| geometric | medium_similarity |          17886 |       17928 |     42 |     0.948636 |        1 |      1 |
| geometric | medium_similarity |          25470 |       25538 |     68 |     0.922227 |        1 |      1 |
| geometric | boundary          |          43700 |       43820 |    120 |     0.823467 |        1 |      1 |
| geometric | boundary          |          43156 |       43290 |    134 |     0.810277 |        1 |      1 |
| geometric | boundary          |          58166 |       58251 |     85 |     0.785412 |        0 |      0 |
| geometric | boundary          |            753 |         828 |     75 |     0.816163 |        1 |      1 |
| geometric | boundary          |          47495 |       47716 |    221 |     0.814215 |        1 |      1 |
| geometric | negative          |          48372 |       48535 |    163 |     0.695397 |        0 |      0 |
| geometric | negative          |           3481 |        3564 |     83 |     0.715447 |        0 |      0 |
| geometric | negative          |          16534 |       16826 |    292 |     0.689973 |        0 |      0 |
| geometric | negative          |          57572 |       57924 |    352 |     0.713944 |        0 |      0 |
| geometric | negative          |          47752 |       47874 |    122 |     0.715808 |        0 |      0 |
| geometric | dist_k_lt_25      |          48009 |       48025 |     16 |     0.993742 |        1 |      1 |
| geometric | dist_k_lt_25      |          66048 |       66055 |      7 |     0.94599  |        1 |      1 |
| geometric | dist_k_lt_25      |          22573 |       22578 |      5 |     0.998303 |        1 |      1 |
| geometric | dist_k_lt_25      |          56987 |       56991 |      4 |     0.998954 |        1 |      1 |
| geometric | dist_k_lt_25      |           7863 |        7879 |     16 |     0.983848 |        1 |      1 |
| geometric | dist_k_25_99      |          20821 |       20864 |     43 |     0.996876 |        1 |      1 |
| geometric | dist_k_25_99      |          39834 |       39869 |     35 |     0.634453 |        0 |      0 |
| geometric | dist_k_25_99      |          65891 |       65928 |     37 |     0.977948 |        1 |      1 |
| geometric | dist_k_25_99      |          18251 |       18287 |     36 |     0.960151 |        1 |      1 |
| geometric | dist_k_25_99      |          21254 |       21332 |     78 |     0.939261 |        1 |      1 |
| geometric | dist_k_100_249    |          23404 |       23512 |    108 |     0.900261 |        1 |      1 |
| geometric | dist_k_100_249    |          60457 |       60566 |    109 |     0.984165 |        1 |      1 |
| geometric | dist_k_100_249    |           3553 |        3722 |    169 |     0.936717 |        1 |      1 |
| geometric | dist_k_100_249    |          27893 |       28067 |    174 |     0.616916 |        0 |      0 |
| geometric | dist_k_100_249    |          26316 |       26433 |    117 |     0.983552 |        1 |      1 |
| geometric | dist_k_ge_250     |          12583 |       13306 |    723 |     0.831793 |        1 |      1 |
| geometric | dist_k_ge_250     |          32248 |       32507 |    259 |     0.720272 |        0 |      0 |
| geometric | dist_k_ge_250     |          34506 |       34782 |    276 |     0.977873 |        1 |      1 |
| geometric | dist_k_ge_250     |          62805 |       63293 |    488 |     0.853209 |        1 |      1 |
| geometric | dist_k_ge_250     |          52996 |       53257 |    261 |     0.998055 |        1 |      1 |
| geometric | cross_segment     |           2486 |        2628 |    142 |     0.995638 |        1 |      1 |
| geometric | cross_segment     |           2407 |        2551 |    144 |     0.826848 |        1 |      1 |
| geometric | cross_segment     |           2422 |        2559 |    137 |     0.930184 |        1 |      1 |
| geometric | cross_segment     |            480 |         534 |     54 |     0.780666 |        0 |      0 |
| geometric | cross_segment     |           2161 |        2643 |    482 |     0.989988 |        1 |      1 |
| uniform   | high_similarity   |          70264 |       71586 |   1322 |     0.998276 |        1 |      1 |
| uniform   | high_similarity   |          70883 |       71370 |    487 |     0.996235 |        1 |      1 |
| uniform   | high_similarity   |          70698 |       70370 |   -328 |     0.998195 |        1 |      1 |
| uniform   | high_similarity   |          39391 |       37940 |  -1451 |     0.951572 |        1 |      1 |
| uniform   | high_similarity   |          71156 |       24525 | -46631 |     0.965297 |        1 |      1 |
| uniform   | medium_similarity |           9728 |       45883 |  36155 |     0.918402 |        1 |      1 |
| uniform   | medium_similarity |          52451 |       37934 | -14517 |     0.936914 |        1 |      1 |
| uniform   | medium_similarity |          27715 |        8865 | -18850 |     0.899414 |        1 |      1 |
| uniform   | medium_similarity |          39003 |       32429 |  -6574 |     0.90904  |        1 |      1 |
| uniform   | medium_similarity |          18042 |       67412 |  49370 |     0.883883 |        1 |      1 |
| uniform   | boundary          |          15655 |       31404 |  15749 |     0.822485 |        1 |      1 |
| uniform   | boundary          |          34724 |       39579 |   4855 |     0.758254 |        0 |      0 |
| uniform   | boundary          |          51233 |       62142 |  10909 |     0.827138 |        1 |      1 |
| uniform   | boundary          |            602 |       66073 |  65471 |     0.844252 |        1 |      1 |
| uniform   | boundary          |           2173 |        3073 |    900 |     0.830768 |        1 |      1 |
| uniform   | negative          |          63901 |       66010 |   2109 |     0.703948 |        0 |      0 |
| uniform   | negative          |           7797 |       12127 |   4330 |     0.705272 |        0 |      0 |
| uniform   | negative          |          10537 |       34337 |  23800 |     0.748145 |        0 |      0 |
| uniform   | negative          |          49310 |       46157 |  -3153 |     0.469395 |        0 |      0 |
| uniform   | negative          |          32284 |       49264 |  16980 |     0.549832 |        0 |      0 |