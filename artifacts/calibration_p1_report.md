# NNUE / Search Calibration Report

- Rows: 96
- Generated: 2026-05-24 08:12:23

## Bucket Summary

| Bucket | Rows | Static Abs P50 | Static Abs P90 | Gap Abs P50 | Gap Abs P90 | Vol P90 |
|---|---:|---:|---:|---:|---:|---:|
| quiet_static | 32 | 19.0 | 53.0 | 12.5 | 332.8 | 87.2 |
| pawn_endgame_rfp | 24 | 0.0 | 118.0 | 9.0 | 1548.0 | 1435.0 |
| low_material_nmp | 16 | 142.0 | 807.0 | 626.5 | 1022.5 | 608.0 |
| tactical_search | 24 | 18.0 | 24.0 | 49.5 | 117.7 | 71.5 |

## Position Results

| ID | Bucket | Engine | Mode | Static | d8 | d10 | Gap d8 | Vol | Best | Notes |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| Q01 | quiet_static | our_engine | full_search | 16 | 158 | 159 | 142 | 60 | f3e5 | Italian-like quiet center |
| Q02 | quiet_static | our_engine | full_search | 17 | 13 | 25 | -4 | 12 | e5c6 | Sicilian developed center |
| Q03 | quiet_static | our_engine | full_search | 21 | 32 | 26 | 11 | 9 | c4e6 | Closed center, active minors |
| Q04 | quiet_static | our_engine | full_search | -2 | -10 | -16 | -8 | 38 | d4c5 | QGD/Carlsbad development |
| Q05 | quiet_static | our_engine | full_search | -16 | 85 | 92 | 101 | 57 | d4e5 | French-style locked center |
| Q06 | quiet_static | our_engine | full_search | 26 | 16 | 15 | -10 | 11 | c4d5 | English/QGD quiet tension |
| Q07 | quiet_static | our_engine | full_search | 56 | 473 | 478 | 417 | 21 | d4d5 | KID/Benoni-style closed center |
| Q08 | quiet_static | our_engine | full_search | 21 | 17 | 24 | -4 | 15 | c4d5 | Maroczy-like quiet structure |
| P01 | pawn_endgame_rfp | our_engine | full_search | 0 | 15 | 16 | 15 | 15 | d3c3 | Opposition, white to move |
| P02 | pawn_endgame_rfp | our_engine | full_search | 0 | -22 | -44 | -22 | 22 | d5d6 | Opposition, black to move |
| P03 | pawn_endgame_rfp | our_engine | full_search | 0 | 0 | 0 | 0 | 0 | c1d1 | K+P boundary, white to move |
| P04 | pawn_endgame_rfp | our_engine | full_search | 0 | 0 | 0 | 0 | 0 | c4c3 | K+P boundary, black to move |
| P05 | pawn_endgame_rfp | our_engine | full_search | 9 | 0 | 0 | -9 | 0 | f2e2 | Passed pawn race |
| P06 | pawn_endgame_rfp | our_engine | full_search | 118 | 1797 | 1310 | 1679 | 1566 | a4a5 | Outside passer race |
| N01 | low_material_nmp | our_engine | full_search | 807 | 968 | 1255 | 161 | 287 | f3e5 | Pawn opposition plus knight; NMP guard candidate |
| N02 | low_material_nmp | our_engine | full_search | 10 | 687 | 544 | 677 | 635 | f6g4 | Pawn opposition plus black knight; NMP guard candidate |
| N03 | low_material_nmp | our_engine | full_search | 263 | 1334 | 1249 | 1071 | 294 | e2d1 | Pawn opposition plus bishop; NMP guard candidate |
| N04 | low_material_nmp | our_engine | full_search | 21 | 679 | 632 | 658 | 361 | e1h4 | Pawn opposition plus black bishop; NMP guard candidate |
| T01 | tactical_search | our_engine | full_search | -17 | -7 | -4 | 10 | 27 | e2a6 | Kiwipete-like, corrected to 8 black pawns |
| T02 | tactical_search | our_engine | full_search | 14 | 52 | 55 | 38 | 25 | c1g5 | Central tension capture sequence |
| T03 | tactical_search | our_engine | full_search | -22 | -122 | -117 | -100 | 30 | f3e5 | King-side pressure, uncastled king |
| T04 | tactical_search | our_engine | full_search | 24 | 74 | 76 | 50 | 35 | c3d5 | Exposed king / attacking compensation |
| T05 | tactical_search | our_engine | full_search | 19 | 68 | 70 | 49 | 42 | c1g5 | Sacrifice-compensation candidate |
| T06 | tactical_search | our_engine | full_search | -16 | -118 | -118 | -102 | 61 | f3e5 | Pinned knight / central break candidate |
| Q01 | quiet_static | our_engine | no_nmp_rfp_razor | 16 | 112 | 141 | 96 | 29 | f3e5 | Italian-like quiet center |
| Q02 | quiet_static | our_engine | no_nmp_rfp_razor | 17 | 13 | 15 | -4 | 12 | e5c6 | Sicilian developed center |
| Q03 | quiet_static | our_engine | no_nmp_rfp_razor | 21 | 32 | 120 | 11 | 88 | d4d5 | Closed center, active minors |
| Q04 | quiet_static | our_engine | no_nmp_rfp_razor | -2 | -11 | -22 | -9 | 38 | d4c5 | QGD/Carlsbad development |
| Q05 | quiet_static | our_engine | no_nmp_rfp_razor | -16 | 90 | 92 | 106 | 57 | d4e5 | French-style locked center |
| Q06 | quiet_static | our_engine | no_nmp_rfp_razor | 26 | 16 | 25 | -10 | 11 | c4d5 | English/QGD quiet tension |
| Q07 | quiet_static | our_engine | no_nmp_rfp_razor | 56 | 519 | 512 | 463 | 121 | d4d5 | KID/Benoni-style closed center |
| Q08 | quiet_static | our_engine | no_nmp_rfp_razor | 21 | 23 | 31 | 2 | 15 | d4c5 | Maroczy-like quiet structure |
| P01 | pawn_endgame_rfp | our_engine | no_nmp_rfp_razor | 0 | 15 | 16 | 15 | 15 | d3c3 | Opposition, white to move |
| P02 | pawn_endgame_rfp | our_engine | no_nmp_rfp_razor | 0 | -49 | -22 | -49 | 49 | d5d6 | Opposition, black to move |
| P03 | pawn_endgame_rfp | our_engine | no_nmp_rfp_razor | 0 | 0 | 0 | 0 | 0 | c1d1 | K+P boundary, white to move |
| P04 | pawn_endgame_rfp | our_engine | no_nmp_rfp_razor | 0 | 0 | 0 | 0 | 0 | c4c3 | K+P boundary, black to move |
| P05 | pawn_endgame_rfp | our_engine | no_nmp_rfp_razor | 9 | 0 | 0 | -9 | 0 | f2e2 | Passed pawn race |
| P06 | pawn_endgame_rfp | our_engine | no_nmp_rfp_razor | 118 | 2103 | 2096 | 1985 | 1872 | a4a5 | Outside passer race |
| N01 | low_material_nmp | our_engine | no_nmp_rfp_razor | 807 | 1055 | 1165 | 248 | 110 | f3e1 | Pawn opposition plus knight; NMP guard candidate |
| N02 | low_material_nmp | our_engine | no_nmp_rfp_razor | 10 | 488 | 484 | 478 | 436 | f6g4 | Pawn opposition plus black knight; NMP guard candidate |
| N03 | low_material_nmp | our_engine | no_nmp_rfp_razor | 263 | 1213 | 1344 | 950 | 131 | e2f3 | Pawn opposition plus bishop; NMP guard candidate |
| N04 | low_material_nmp | our_engine | no_nmp_rfp_razor | 21 | 88 | 669 | 67 | 581 | e1f2 | Pawn opposition plus black bishop; NMP guard candidate |
| T01 | tactical_search | our_engine | no_nmp_rfp_razor | -17 | -7 | -23 | 10 | 27 | e2a6 | Kiwipete-like, corrected to 8 black pawns |
| T02 | tactical_search | our_engine | no_nmp_rfp_razor | 14 | 54 | 55 | 40 | 27 | c1g5 | Central tension capture sequence |
| T03 | tactical_search | our_engine | no_nmp_rfp_razor | -22 | -140 | -131 | -118 | 48 | f3e5 | King-side pressure, uncastled king |
| T04 | tactical_search | our_engine | no_nmp_rfp_razor | 24 | 74 | 98 | 50 | 35 | c3d5 | Exposed king / attacking compensation |
| T05 | tactical_search | our_engine | no_nmp_rfp_razor | 19 | 62 | 69 | 43 | 42 | d4e5 | Sacrifice-compensation candidate |
| T06 | tactical_search | our_engine | no_nmp_rfp_razor | -16 | -138 | -109 | -122 | 81 | f3e5 | Pinned knight / central break candidate |
| Q01 | quiet_static | our_engine | no_lmr | 16 | 158 | 174 | 142 | 60 | f3e5 | Italian-like quiet center |
| Q02 | quiet_static | our_engine | no_lmr | 17 | 34 | 22 | 17 | 12 | e5c6 | Sicilian developed center |
| Q03 | quiet_static | our_engine | no_lmr | 21 | 120 | 174 | 99 | 79 | d4d5 | Closed center, active minors |
| Q04 | quiet_static | our_engine | no_lmr | -2 | -11 | -1 | -9 | 38 | f3g5 | QGD/Carlsbad development |
| Q05 | quiet_static | our_engine | no_lmr | -16 | 99 | 99 | 115 | 42 | d4e5 | French-style locked center |
| Q06 | quiet_static | our_engine | no_lmr | 26 | 16 | 17 | -10 | 11 | c4d5 | English/QGD quiet tension |
| Q07 | quiet_static | our_engine | no_lmr | 56 | 410 | 472 | 354 | 80 | d4d5 | KID/Benoni-style closed center |
| Q08 | quiet_static | our_engine | no_lmr | 21 | 17 | 32 | -4 | 15 | d4c5 | Maroczy-like quiet structure |
| P01 | pawn_endgame_rfp | our_engine | no_lmr | 0 | 15 | 27 | 15 | 13 | d3c3 | Opposition, white to move |
| P02 | pawn_endgame_rfp | our_engine | no_lmr | 0 | -41 | -44 | -41 | 41 | d5d6 | Opposition, black to move |
| P03 | pawn_endgame_rfp | our_engine | no_lmr | 0 | 0 | 16 | 0 | 16 | c1b2 | K+P boundary, white to move |
| P04 | pawn_endgame_rfp | our_engine | no_lmr | 0 | 0 | 0 | 0 | 0 | c4c3 | K+P boundary, black to move |
| P05 | pawn_endgame_rfp | our_engine | no_lmr | 9 | 0 | 0 | -9 | 0 | f2e2 | Passed pawn race |
| P06 | pawn_endgame_rfp | our_engine | no_lmr | 118 | 1610 | 1743 | 1492 | 1379 | a4a5 | Outside passer race |
| N01 | low_material_nmp | our_engine | no_lmr | 807 | 1402 | 1327 | 595 | 416 | f3e5 | Pawn opposition plus knight; NMP guard candidate |
| N02 | low_material_nmp | our_engine | no_lmr | 10 | 526 | 867 | 516 | 474 | f6g4 | Pawn opposition plus black knight; NMP guard candidate |
| N03 | low_material_nmp | our_engine | no_lmr | 263 | 1318 | 1439 | 1055 | 298 | e2d1 | Pawn opposition plus bishop; NMP guard candidate |
| N04 | low_material_nmp | our_engine | no_lmr | 21 | 727 | 663 | 706 | 393 | e1f2 | Pawn opposition plus black bishop; NMP guard candidate |
| T01 | tactical_search | our_engine | no_lmr | -17 | -7 | -5 | 10 | 27 | e2a6 | Kiwipete-like, corrected to 8 black pawns |
| T02 | tactical_search | our_engine | no_lmr | 14 | 112 | 15 | 98 | 97 | c1g5 | Central tension capture sequence |
| T03 | tactical_search | our_engine | no_lmr | -22 | -151 | -153 | -129 | 45 | f3e5 | King-side pressure, uncastled king |
| T04 | tactical_search | our_engine | no_lmr | 24 | 74 | 79 | 50 | 35 | c3d5 | Exposed king / attacking compensation |
| T05 | tactical_search | our_engine | no_lmr | 19 | 60 | 84 | 41 | 41 | d4e5 | Sacrifice-compensation candidate |
| T06 | tactical_search | our_engine | no_lmr | -16 | -133 | -131 | -117 | 76 | f3e5 | Pinned knight / central break candidate |
| Q01 | quiet_static | our_engine | no_see_pruning | 16 | 158 | 159 | 142 | 60 | f3e5 | Italian-like quiet center |
| Q02 | quiet_static | our_engine | no_see_pruning | 17 | 17 | 23 | 0 | 8 | e5c6 | Sicilian developed center |
| Q03 | quiet_static | our_engine | no_see_pruning | 21 | 35 | 152 | 14 | 117 | d4d5 | Closed center, active minors |
| Q04 | quiet_static | our_engine | no_see_pruning | -2 | -17 | -23 | -15 | 38 | d4c5 | QGD/Carlsbad development |
| Q05 | quiet_static | our_engine | no_see_pruning | -16 | 85 | 99 | 101 | 57 | d4e5 | French-style locked center |
| Q06 | quiet_static | our_engine | no_see_pruning | 26 | 16 | 25 | -10 | 11 | c4d5 | English/QGD quiet tension |
| Q07 | quiet_static | our_engine | no_see_pruning | 56 | 465 | 429 | 409 | 164 | d4d5 | KID/Benoni-style closed center |
| Q08 | quiet_static | our_engine | no_see_pruning | 21 | 22 | 24 | 1 | 15 | c4d5 | Maroczy-like quiet structure |
| P01 | pawn_endgame_rfp | our_engine | no_see_pruning | 0 | 0 | 21 | 0 | 21 | d3c3 | Opposition, white to move |
| P02 | pawn_endgame_rfp | our_engine | no_see_pruning | 0 | -22 | -44 | -22 | 22 | d5d6 | Opposition, black to move |
| P03 | pawn_endgame_rfp | our_engine | no_see_pruning | 0 | 0 | 0 | 0 | 0 | c1d1 | K+P boundary, white to move |
| P04 | pawn_endgame_rfp | our_engine | no_see_pruning | 0 | 0 | 0 | 0 | 0 | c4c3 | K+P boundary, black to move |
| P05 | pawn_endgame_rfp | our_engine | no_see_pruning | 9 | 0 | 0 | -9 | 0 | f2e2 | Passed pawn race |
| P06 | pawn_endgame_rfp | our_engine | no_see_pruning | 118 | 1690 | 1384 | 1572 | 1459 | a4a5 | Outside passer race |
| N01 | low_material_nmp | our_engine | no_see_pruning | 807 | 1203 | 1287 | 396 | 289 | f3d2 | Pawn opposition plus knight; NMP guard candidate |
| N02 | low_material_nmp | our_engine | no_see_pruning | 10 | 416 | 483 | 406 | 357 | f6g4 | Pawn opposition plus black knight; NMP guard candidate |
| N03 | low_material_nmp | our_engine | no_see_pruning | 263 | 1253 | 1319 | 990 | 233 | e2f3 | Pawn opposition plus bishop; NMP guard candidate |
| N04 | low_material_nmp | our_engine | no_see_pruning | 21 | 804 | 799 | 783 | 722 | e1f2 | Pawn opposition plus black bishop; NMP guard candidate |
| T01 | tactical_search | our_engine | no_see_pruning | -17 | -7 | -4 | 10 | 27 | e2a6 | Kiwipete-like, corrected to 8 black pawns |
| T02 | tactical_search | our_engine | no_see_pruning | 14 | 53 | 55 | 39 | 26 | c1g5 | Central tension capture sequence |
| T03 | tactical_search | our_engine | no_see_pruning | -22 | -113 | -160 | -91 | 47 | c1g5 | King-side pressure, uncastled king |
| T04 | tactical_search | our_engine | no_see_pruning | 24 | 63 | 76 | 39 | 48 | c3d5 | Exposed king / attacking compensation |
| T05 | tactical_search | our_engine | no_see_pruning | 19 | 52 | 62 | 33 | 41 | d4e5 | Sacrifice-compensation candidate |
| T06 | tactical_search | our_engine | no_see_pruning | -16 | -108 | -140 | -92 | 47 | a2a3 | Pinned knight / central break candidate |
