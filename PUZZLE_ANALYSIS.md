# Puzzle Analysis Report
Total Failed Puzzles: 9

## Lichess Puzzle 006eO
**FEN:** `8/8/2p5/1p1p1k2/3P4/1PP1pK2/8/8 w - - 4 65`
**Correct Move:** `b3b4`
**Engine Move:** `f3e3`

### Static Evaluation (Root)
```
FEN: 8/8/2p5/1p1p1k2/3P4/1PP1pK2/8/8 w - - 4 65
Side to Move: White
Phase: 0 / 24 (MG Weight: 0.00, EG Weight: 1.00)
------------------------------------------------------------
Term                 | White MG | White EG | Black MG | Black EG |   Net MG |   Net EG
------------------------------------------------------------
Material             |      300 |      360 |      400 |      480 |     -100 |     -120
PST                  |      -15 |       60 |       10 |      130 |      -25 |      -70
------------------------------------------------------------
Term (Net Only)      |   Net MG |   Net EG
----------------------------------------
King Safety          |        0 |        0
Pawn Structure       |      -48 |      -95
Coordination         |        0 |        0
Mobility             |        0 |        0
Threats              |       35 |       20
------------------------------------------------------------
Total                |     -138 |     -265
Final Interpolated Score (White Perspective): -265
Final Score (Side to Move): -265

```

### Search Comparison
Comparing Engine Move vs Correct Move:
```
FEN: 8/8/2p5/1p1p1k2/3P4/1PP1pK2/8/8 w - - 4 65
Depth: 12
Time Limit: 2000ms
------------------------------------------------------------

Analyzing move: f3e3
info depth 1 score cp -25 nodes 24 nps 0 time 71082 pv f5e6
  Score (for me): 25 (cp)
  Best Reply: f5e6
  Nodes: 24

Analyzing move: b3b4
info depth 1 score cp 50 nodes 21 nps 80733 time 0 pv e3e2
info depth 2 score cp 50 nodes 43 nps 95174 time 0 pv e3e2 f3e2
info depth 3 score cp 90 nodes 120 nps 176974 time 0 pv e3e2 f3e2 f5e4
info depth 4 score cp 60 nodes 273 nps 220795 time 1 pv e3e2 f3e2 f5e4 e2d2
info depth 5 score cp 20 nodes 768 nps 324687 time 2 pv e3e2 f3e2 f5e4 e2d2 e4f4
info depth 6 score cp 0 nodes 1971 nps 387411 time 5 pv e3e2 f3e2 f5e6 e2d2 e6d6 d2d3
info depth 7 score cp 0 nodes 2896 nps 405173 time 7 pv e3e2 f3e2 f5e6 e2d3 e6d6 d3e3 d6e6
info depth 8 score cp 0 nodes 3524 nps 420420 time 8 pv e3e2 f3e2 f5e6 e2d3 e6d6 d3e3 d6e6 e3d3
info depth 9 score cp 0 nodes 5821 nps 120230 time 48 pv e3e2 f3e2 f5e6 e2d3 e6d6 d3e3 d6e6 e3d2 e6d7
info depth 10 score cp 0 nodes 9503 nps 139365 time 68 pv e3e2 f3e2 f5e6 e2d3 e6d6 d3e3 d6e6 e3d2 e6d6 d2d3
info depth 11 score cp 0 nodes 10805 nps 150326 time 71 pv e3e2 f3e2 f5e6 e2d3 e6d6 d3e3 d6e6 e3d2 e6d7 d2e3 d7e6
info depth 12 score cp 0 nodes 19075 nps 208454 time 91 pv e3e2 f3e2 f5e6 e2d3 e6d6 d3e3 d6e6 e3d2 e6d7 d2e3 d7e6 e3d3
  Score (for me): 0 (cp)
  Best Reply: e3e2
  Nodes: 19075

============================================================
Summary Comparison
------------------------------------------------------------
Move: f3e3   | Score:     25 | Best Reply: f5e6   | Nodes: 24
Move: b3b4   | Score:      0 | Best Reply: e3e2   | Nodes: 19075

```

---

## Lichess Puzzle 006of
**FEN:** `r2qr2k/1pp2Qp1/1b4np/pP2P3/P4n2/B1N2N1P/5PP1/R3R1K1 b - - 0 20`
**Correct Move:** `d8d3`
**Engine Move:** `f4d3`

### Static Evaluation (Root)
```
FEN: r2qr2k/1pp2Qp1/1b4np/pP2P3/P4n2/B1N2N1P/5PP1/R3R1K1 b - - 0 20
Side to Move: Black
Phase: 22 / 24 (MG Weight: 0.92, EG Weight: 0.08)
------------------------------------------------------------
Term                 | White MG | White EG | Black MG | Black EG |   Net MG |   Net EG
------------------------------------------------------------
Material             |     3470 |     3690 |     3370 |     3570 |      100 |      120
PST                  |      100 |       95 |       85 |       35 |       15 |       60
------------------------------------------------------------
Term (Net Only)      |   Net MG |   Net EG
----------------------------------------
King Safety          |       96 |       48
Pawn Structure       |       80 |      195
Coordination         |      -15 |      -10
Mobility             |       21 |        9
Threats              |        0 |        0
------------------------------------------------------------
Total                |      297 |      422
Final Interpolated Score (White Perspective): 307
Final Score (Side to Move): -307

```

### Search Comparison
Comparing Engine Move vs Correct Move:
```
FEN: r2qr2k/1pp2Qp1/1b4np/pP2P3/P4n2/B1N2N1P/5PP1/R3R1K1 b - - 0 20
Depth: 12
Time Limit: 2000ms
------------------------------------------------------------

Analyzing move: f4d3
info depth 1 score cp 153 nodes 761 nps 236084 time 3 pv c3e4
info depth 2 score cp 680 nodes 896 nps 196892 time 4 pv f7g6 d3e1
info depth 3 score cp 130 nodes 13150 nps 188399 time 69 pv f7g6 b6f2 g1f1 d3e1
info depth 4 score cp 130 nodes 17170 nps 193570 time 88 pv f7g6 b6f2 g1f1 d3e1 a1e1
info depth 5 score cp 130 nodes 30962 nps 203822 time 151 pv f7g6 b6f2 g1f1 d3e1 a1e1 f2e1
info depth 6 score cp 39 nodes 113169 nps 219082 time 516 pv f7g6 b6f2 g1f1 f2e1 a1e1 d3f4 g6f7
info depth 7 score cp 12 nodes 325434 nps 228294 time 1425 pv e1e2 d3e5 f3e5 g6e5 f7f5 d8d4 a1c1
  Score (for me): -12 (cp)
  Best Reply: e1e2
  Nodes: 325434

Analyzing move: d8d3
info depth 1 score cp 42 nodes 864 nps 203474 time 4 pv c3e4
info depth 2 score cp 178 nodes 1833 nps 187251 time 9 pv c3e4 f4h3 g2h3
info depth 3 score cp 177 nodes 6645 nps 204885 time 32 pv c3e4 a8d8 g1h1
info depth 4 score cp 235 nodes 25816 nps 216890 time 119 pv a1c1 e8e6 e1e4 f4h3 g2h3
info depth 5 score cp 162 nodes 58343 nps 164295 time 355 pv a1c1 a8d8 e1d1 g6e5 f3e5
info depth 6 score cp 112 nodes 227175 nps 197312 time 1151 pv c3d1 e8f8 a3f8 a8f8 d1b2 f8f7
  Score (for me): -112 (cp)
  Best Reply: c3d1
  Nodes: 227175

============================================================
Summary Comparison
------------------------------------------------------------
Move: f4d3   | Score:    -12 | Best Reply: e1e2   | Nodes: 325434
Move: d8d3   | Score:   -112 | Best Reply: c3d1   | Nodes: 227175

```

---

## Lichess Puzzle 0078T
**FEN:** `rk5r/1b3R2/pp2p2q/4P2p/B6B/4p2P/PP4P1/5Q1K w - - 0 28`
**Correct Move:** `f7b7`
**Engine Move:** `f1d3`

### Static Evaluation (Root)
```
FEN: rk5r/1b3R2/pp2p2q/4P2p/B6B/4p2P/PP4P1/5Q1K w - - 0 28
Side to Move: White
Phase: 17 / 24 (MG Weight: 0.71, EG Weight: 0.29)
------------------------------------------------------------
Term                 | White MG | White EG | Black MG | Black EG |   Net MG |   Net EG
------------------------------------------------------------
Material             |     2560 |     2760 |     2730 |     2950 |     -170 |     -190
PST                  |       55 |        0 |       55 |       65 |        0 |      -65
------------------------------------------------------------
Term (Net Only)      |   Net MG |   Net EG
----------------------------------------
King Safety          |      518 |      259
Pawn Structure       |      -45 |     -195
Coordination         |       55 |       85
Mobility             |       49 |       20
Threats              |        0 |        0
------------------------------------------------------------
Total                |      407 |      -86
Final Interpolated Score (White Perspective): 263
Final Score (Side to Move): 263

```

### Search Comparison
Comparing Engine Move vs Correct Move:
```
FEN: rk5r/1b3R2/pp2p2q/4P2p/B6B/4p2P/PP4P1/5Q1K w - - 0 28
Depth: 12
Time Limit: 2000ms
------------------------------------------------------------

Analyzing move: f1d3
info depth 1 score cp -213 nodes 103 nps 196280 time 0 pv b6b5
info depth 2 score cp -7 nodes 257 nps 133144 time 1 pv h8c8 d3d6 b8a7
info depth 3 score cp -28 nodes 1824 nps 248672 time 7 pv a8a7 d3d6 b8a8 h1h2
info depth 4 score cp 46 nodes 3439 nps 95013 time 36 pv h8f8 f7f8 h6f8 a4d7 f8f4
info depth 5 score cp 4 nodes 26979 nps 163618 time 164 pv h8c8 d3d7 a8a7 h4d8 h6g6
info depth 6 score cp 13 nodes 58597 nps 153174 time 382 pv h8c8 d3d7 a8a7 d7d6 b8a8 d6b6 c8c1 h1h2
info depth 7 score cp 12 nodes 164458 nps 159450 time 1031 pv h8c8 d3d7 a8a7 f7f6 b7g2 h1g2 a7d7 f6h6
  Score (for me): -12 (cp)
  Best Reply: h8c8
  Nodes: 164458

Analyzing move: f7b7
info depth 1 score cp 141 nodes 7 nps 20000 time 0 pv b8b7
info depth 2 score cp 107 nodes 56 nps 84066 time 0 pv b8b7 f1f7 b7b8
info depth 3 score cp 164 nodes 150 nps 73549 time 2 pv b8b7 f1f7 b7b8 f7f6
info depth 4 score cp 200 nodes 2873 nps 186069 time 15 pv b8b7 f1d3 b6b5 d3d7 b7b8
info depth 5 score cp 107 nodes 15253 nps 197810 time 77 pv b8b7 f1f7 b7b8 a4c6 a8a7 c6d7
info depth 6 score cp 168 nodes 16123 nps 102897 time 156 pv b8b7 f1f7 b7b8 a4c6 a8a7 f7f3 a7c7
info depth 7 score cp 181 nodes 32404 nps 125288 time 258 pv b8b7 f1f7 b7b8 a4c6 a8a7 h4e7 a7c7 e7d6
info depth 8 score cp 146 nodes 51744 nps 87418 time 591 pv b8b7 f1f7 b7b8 h4e7 h6h7 e7d6 b8c8 a4d7 c8d8 f7h7 h8h7
info depth 9 score cp -17 nodes 114546 nps 113810 time 1006 pv b8b7 f1f7 b7b8 h4e7 h6h7 f7e6 h7e4 e6b6 b8c8 b6c6 e4c6 a4c6
  Score (for me): 17 (cp)
  Best Reply: b8b7
  Nodes: 114546

============================================================
Summary Comparison
------------------------------------------------------------
Move: f7b7   | Score:     17 | Best Reply: b8b7   | Nodes: 114546
Move: f1d3   | Score:    -12 | Best Reply: h8c8   | Nodes: 164458

```

---

## Lichess Puzzle 004d8
**FEN:** `8/4kr2/R2p4/1p1Pp3/5pp1/3K1P2/PPP5/8 w - - 0 40`
**Correct Move:** `a6a7`
**Engine Move:** `f3g4`

### Static Evaluation (Root)
```
FEN: 8/4kr2/R2p4/1p1Pp3/5pp1/3K1P2/PPP5/8 w - - 0 40
Side to Move: White
Phase: 4 / 24 (MG Weight: 0.17, EG Weight: 0.83)
------------------------------------------------------------
Term                 | White MG | White EG | Black MG | Black EG |   Net MG |   Net EG
------------------------------------------------------------
Material             |     1000 |     1130 |     1000 |     1130 |        0 |        0
PST                  |       15 |       95 |       35 |      110 |      -20 |      -15
------------------------------------------------------------
Term (Net Only)      |   Net MG |   Net EG
----------------------------------------
King Safety          |      -66 |      -33
Pawn Structure       |      -10 |      -25
Coordination         |        0 |        0
Mobility             |        6 |        2
Threats              |        0 |        0
------------------------------------------------------------
Total                |      -90 |      -71
Final Interpolated Score (White Perspective): -75
Final Score (Side to Move): -75

```

### Search Comparison
Comparing Engine Move vs Correct Move:
```
FEN: 8/4kr2/R2p4/1p1Pp3/5pp1/3K1P2/PPP5/8 w - - 0 40
Depth: 12
Time Limit: 2000ms
------------------------------------------------------------

Analyzing move: f3g4
info depth 1 score cp 179 nodes 8 nps 30093 time 0 pv f4f3
info depth 2 score cp 8 nodes 101 nps 141538 time 0 pv f4f3 d3e3
info depth 3 score cp 36 nodes 689 nps 161013 time 4 pv f4f3 d3e3
info depth 4 score cp 10 nodes 2294 nps 310769 time 7 pv f4f3 a6b6 f7f4 d3e3
info depth 5 score cp 31 nodes 4988 nps 353058 time 14 pv f4f3 a6a7 e7f6 g4g5 f6g6 a7a5 g6g5
info depth 6 score cp -83 nodes 23774 nps 260125 time 91 pv f7f8 a6a7 e7f6 a7d7 f6g5 d7d6 g5g4
info depth 7 score cp 3 nodes 55383 nps 350486 time 158 pv f4f3 d3e3 f7f4 a6a7 e7f6 e3f2 b5b4 f2e3
info depth 8 score cp -86 nodes 90476 nps 383649 time 235 pv f4f3 d3e3 f7f4 a6a7 e7f8 e3f2 f8e8 g4g5 f4f5
info depth 9 score cp -38 nodes 215426 nps 439376 time 490 pv f4f3 d3e3 f7f4 e3f2 f4g4 a6a7 e7f6
  Score (for me): 38 (cp)
  Best Reply: f4f3
  Nodes: 215426

Analyzing move: a6a7
info depth 1 score cp -85 nodes 24 nps 71595 time 0 pv e7e8
info depth 2 score cp -85 nodes 100 nps 170223 time 0 pv e7e8 a7f7
info depth 3 score cp -85 nodes 297 nps 290442 time 1 pv e7e8 a7a8 e8e7 a8a7
info depth 4 score cp 101 nodes 600 nps 199428 time 3 pv e7f6 a7f7 f6f7 f3g4 f7f6
info depth 5 score cp 76 nodes 1139 nps 269189 time 4 pv e7f6 a7f7 f6f7 f3g4 f7f6 c2c3
info depth 6 score cp 83 nodes 3616 nps 399310 time 9 pv e7f6 a7f7 f6f7 f3g4 f7f6 d3e2 f6g5
info depth 7 score cp 3 nodes 6612 nps 459151 time 14 pv e7f6 a7f7 f6f7 f3g4 f7e7 d3e2 e7f6 e2f3
info depth 8 score cp -29 nodes 9675 nps 477432 time 20 pv e7f6 a7f7 f6f7 f3g4 b5b4 a2a4 b4a3 b2a3 f7e7
info depth 9 score cp -44 nodes 15871 nps 446778 time 35 pv e7f6 a7f7 f6f7 f3g4 f7e7 d3e4 e7f6 c2c3 f6g5 e4f3
info depth 10 score cp -45 nodes 44339 nps 475764 time 93 pv e7f6 a7f7 f6f7 f3g4 f7e8 d3e4 e8e7 c2c4 e7d7 e4f3 e5e4 f3e2
info depth 11 score cp -103 nodes 72302 nps 474140 time 152 pv e7f6 a7f7 f6f7 f3g4 f4f3
info depth 12 score cp -100 nodes 134402 nps 455696 time 294 pv e7f6 a7f7 f6f7 f3g4 f4f3 d3e3 e5e4 c2c4 b5b4 e3f2 f7e7 a2a4 b4a3
  Score (for me): 100 (cp)
  Best Reply: e7f6
  Nodes: 134402

============================================================
Summary Comparison
------------------------------------------------------------
Move: a6a7   | Score:    100 | Best Reply: e7f6   | Nodes: 134402
Move: f3g4   | Score:     38 | Best Reply: f4f3   | Nodes: 215426

```

---

## Lichess Puzzle 005qG
**FEN:** `8/8/1p1k1p1p/3np3/2B2p2/PP1K1PP1/7P/8 w - - 0 37`
**Correct Move:** `c4d5`
**Engine Move:** `d3e4`

### Static Evaluation (Root)
```
FEN: 8/8/1p1k1p1p/3np3/2B2p2/PP1K1PP1/7P/8 w - - 0 37
Side to Move: White
Phase: 2 / 24 (MG Weight: 0.08, EG Weight: 0.92)
------------------------------------------------------------
Term                 | White MG | White EG | Black MG | Black EG |   Net MG |   Net EG
------------------------------------------------------------
Material             |      830 |      940 |      820 |      910 |       10 |       30
PST                  |      -20 |       90 |       20 |      130 |      -40 |      -40
------------------------------------------------------------
Term (Net Only)      |   Net MG |   Net EG
----------------------------------------
King Safety          |      138 |       69
Pawn Structure       |       35 |       20
Coordination         |        0 |        0
Mobility             |       -8 |       -4
Threats              |        0 |        0
------------------------------------------------------------
Total                |      135 |       75
Final Interpolated Score (White Perspective): 80
Final Score (Side to Move): 80

```

### Search Comparison
Comparing Engine Move vs Correct Move:
```
FEN: 8/8/1p1k1p1p/3np3/2B2p2/PP1K1PP1/7P/8 w - - 0 37
Depth: 12
Time Limit: 2000ms
------------------------------------------------------------

Analyzing move: d3e4
info depth 1 score cp -118 nodes 100 nps 248920 time 0 pv d5c3 e4f5
info depth 2 score cp -94 nodes 490 nps 427367 time 1 pv d5c3 e4d3 c3d5
info depth 3 score cp -128 nodes 1506 nps 469567 time 3 pv d5c3 e4f5 b6b5 c4d3
info depth 4 score cp -111 nodes 3544 nps 488148 time 7 pv d5c3 e4d3 f4g3 h2g3 c3d5
info depth 5 score cp -128 nodes 9057 nps 479904 time 18 pv d5c3 e4f5 f4g3 h2g3 c3d5 b3b4
info depth 6 score cp -116 nodes 14907 nps 459860 time 32 pv d5c3 e4f5 f4g3 h2g3 b6b5 c4e6 e5e4
info depth 7 score cp -143 nodes 31268 nps 462521 time 67 pv d5c3 e4f5 c3b1 g3f4 b1d2 h2h4 d2f3 h4h5
info depth 8 score cp -125 nodes 48234 nps 432420 time 111 pv d5c3 e4f5 c3b1 g3f4 b1d2 f4e5 f6e5 h2h3 d2f3 f5e4
info depth 9 score cp -138 nodes 84037 nps 424751 time 197 pv d5c3 e4f5 c3b1 g3f4 b1d2 f5f6 e5f4 f6f5 d2f3 f5f4
info depth 10 score cp -120 nodes 274089 nps 422912 time 648 pv d5c3 e4d3 b6b5 d3c3 b5c4 g3f4 c4b3 f4e5 d6e5 c3b3 h6h5 b3c4
info depth 11 score cp -120 nodes 388882 nps 413470 time 940 pv d5c3 e4d3 b6b5 d3c3 b5c4 g3f4 c4b3 f4e5 f6e5 c3b3 d6d5 b3c3 h6h5
  Score (for me): 120 (cp)
  Best Reply: d5c3
  Nodes: 388882

Analyzing move: c4d5
info depth 1 score cp 0 nodes 10 nps 40603 time 0 pv d6d5
info depth 2 score cp 0 nodes 49 nps 121466 time 0 pv d6d5 g3f4
info depth 3 score cp 0 nodes 236 nps 300868 time 0 pv d6d5 g3f4 e5f4
info depth 4 score cp -11 nodes 618 nps 388442 time 1 pv d6d5 g3g4 e5e4 f3e4 d5e5 h2h4
info depth 5 score cp 10 nodes 1566 nps 424389 time 3 pv d6d5 g3f4 e5f4 b3b4 d5e5
info depth 6 score cp -15 nodes 3181 nps 447360 time 7 pv d6d5 g3f4 e5f4 b3b4 d5e5 a3a4
info depth 7 score cp -5 nodes 5147 nps 461017 time 11 pv d6d5 g3f4 e5f4 b3b4 d5e5 a3a4 f6f5
info depth 8 score cp -5 nodes 7990 nps 502541 time 15 pv d6d5 g3f4 e5f4 b3b4 f6f5 a3a4 h6h5 b4b5
info depth 9 score cp -75 nodes 37360 nps 501554 time 74 pv f4g3 h2g3 d6d5 g3g4 d5d6 b3b4 d6e6 d3e4 e6d6
info depth 10 score cp -110 nodes 58520 nps 489770 time 119 pv f4g3 h2g3 d6d5 g3g4 d5d6 d3e4 d6e6 a3a4 e6d6 b3b4
info depth 11 score cp -91 nodes 75597 nps 488862 time 154 pv f4g3 h2g3 d6d5 g3g4 d5e6 d3e4 f6f5 g4f5 e6f6 b3b4 h6h5 f3f4 e5f4
info depth 12 score cp -96 nodes 370166 nps 498384 time 742 pv f4g3 h2g3 d6d5 g3g4 d5e6 d3e4 f6f5 g4f5 e6f6 f3f4 e5f4 e4f4 h6h5 f4e4
  Score (for me): 96 (cp)
  Best Reply: f4g3
  Nodes: 370166

============================================================
Summary Comparison
------------------------------------------------------------
Move: d3e4   | Score:    120 | Best Reply: d5c3   | Nodes: 388882
Move: c4d5   | Score:     96 | Best Reply: f4g3   | Nodes: 370166

```

---

## Lichess Puzzle 005f3
**FEN:** `r5k1/2p1pp2/pp4p1/1q5r/5P2/2QP2R1/PP6/1K4R1 w - - 1 33`
**Correct Move:** `g3g6`
**Engine Move:** `c3c7`

### Static Evaluation (Root)
```
FEN: r5k1/2p1pp2/pp4p1/1q5r/5P2/2QP2R1/PP6/1K4R1 w - - 1 33
Side to Move: White
Phase: 16 / 24 (MG Weight: 0.67, EG Weight: 0.33)
------------------------------------------------------------
Term                 | White MG | White EG | Black MG | Black EG |   Net MG |   Net EG
------------------------------------------------------------
Material             |     2300 |     2490 |     2500 |     2730 |     -200 |     -240
PST                  |       50 |       25 |       20 |       25 |       30 |        0
------------------------------------------------------------
Term (Net Only)      |   Net MG |   Net EG
----------------------------------------
King Safety          |      111 |       55
Pawn Structure       |      -20 |      -10
Coordination         |      -10 |       -5
Mobility             |       -6 |       -1
Threats              |       35 |       20
------------------------------------------------------------
Total                |      -60 |     -181
Final Interpolated Score (White Perspective): -101
Final Score (Side to Move): -101

```

### Search Comparison
Comparing Engine Move vs Correct Move:
```
FEN: r5k1/2p1pp2/pp4p1/1q5r/5P2/2QP2R1/PP6/1K4R1 w - - 1 33
Depth: 12
Time Limit: 2000ms
------------------------------------------------------------

Analyzing move: c3c7
info depth 1 score cp 196 nodes 266 nps 212754 time 1 pv b5b4
info depth 2 score cp 8 nodes 973 nps 137196 time 7 pv b5b4 g3g2
info depth 3 score cp 109 nodes 2813 nps 76371 time 36 pv a8e8 c7c4 b5c4
info depth 4 score cp 57 nodes 8661 nps 129691 time 66 pv a8e8 c7c2 e7e5 c2c7
info depth 5 score cp 169 nodes 35669 nps 84049 time 424 pv h5h2 c7c3 a8d8 g1e1 g8f8
info depth 6 score cp 80 nodes 121054 nps 130289 time 929 pv h5h2 c7c3 b5c5 c3c5 b6c5 g1e1
  Score (for me): -80 (cp)
  Best Reply: h5h2
  Nodes: 121054

Analyzing move: g3g6
info depth 1 score cp 331 nodes 16 nps 71697 time 0 pv f7g6
info depth 2 score cp 411 nodes 53 nps 106362 time 0 pv f7g6 g1g6 g8f7
info depth 3 score cp 195 nodes 225 nps 144897 time 1 pv f7g6 g1g6 g8f7 g6g7 f7e6
info depth 4 score cp 288 nodes 739 nps 95486 time 7 pv f7g6 g1g6 g8f7 g6g7 f7e6
info depth 5 score cp 383 nodes 7235 nps 228186 time 31 pv f7g6 g1g6 g8f7 g6g7 f7f8 g7g1 f8f7
info depth 6 score cp -92 nodes 12179 nps 185817 time 65 pv f7g6 g1g6 g8f7 c3g7 f7e8 g7g8 e8d7 g8e6 d7d8 g6g8 b5e8
info depth 7 score cp -92 nodes 17931 nps 197123 time 90 pv f7g6 g1g6 g8f7 c3g7 f7e8 g7g8 e8d7 g8e6 d7d8 g6g8 b5e8 g8e8 d8e8
info depth 8 score cp -507 nodes 48654 nps 159705 time 304 pv f7g6 g1g6 g8f7 c3g7 f7e8 g7g8 e8d7 g8e6 d7d8 g6g8 b5e8 g8e8 d8e8 e6g6 e8d8
info depth 9 score cp -507 nodes 73122 nps 159047 time 459 pv f7g6 g1g6 g8f7 c3g7 f7e8 g7g8 e8d7 g8e6 d7d8 g6g8 b5e8 g8e8 d8e8 e6g6 e8d8 g6h5
info depth 10 score cp -498 nodes 165583 nps 197091 time 840 pv f7g6 g1g6 g8f7 c3g7 f7e8 g7g8 e8d7 g8e6 d7d8 g6g8 b5e8 e6c6 e8g8 c6a8 d8d7 a8g8
  Score (for me): 498 (cp)
  Best Reply: f7g6
  Nodes: 165583

============================================================
Summary Comparison
------------------------------------------------------------
Move: g3g6   | Score:    498 | Best Reply: f7g6   | Nodes: 165583
Move: c3c7   | Score:    -80 | Best Reply: h5h2   | Nodes: 121054

```

---

## Lichess Puzzle 005yO
**FEN:** `r1r2k2/ppq3bQ/4p2p/4n3/3p4/2P5/PBB2PPP/4R1K1 w - - 3 25`
**Correct Move:** `b2a3`
**Engine Move:** `e1e4`

### Static Evaluation (Root)
```
FEN: r1r2k2/ppq3bQ/4p2p/4n3/3p4/2P5/PBB2PPP/4R1K1 w - - 3 25
Side to Move: White
Phase: 18 / 24 (MG Weight: 0.75, EG Weight: 0.25)
------------------------------------------------------------
Term                 | White MG | White EG | Black MG | Black EG |   Net MG |   Net EG
------------------------------------------------------------
Material             |     2560 |     2760 |     3050 |     3260 |     -490 |     -500
PST                  |       50 |       20 |       85 |       70 |      -35 |      -50
------------------------------------------------------------
Term (Net Only)      |   Net MG |   Net EG
----------------------------------------
King Safety          |      490 |      245
Pawn Structure       |      -10 |       -5
Coordination         |       10 |       20
Mobility             |       20 |        5
Threats              |       35 |       20
------------------------------------------------------------
Total                |       20 |     -265
Final Interpolated Score (White Perspective): -52
Final Score (Side to Move): -52

```

### Search Comparison
Comparing Engine Move vs Correct Move:
```
FEN: r1r2k2/ppq3bQ/4p2p/4n3/3p4/2P5/PBB2PPP/4R1K1 w - - 3 25
Depth: 12
Time Limit: 2000ms
------------------------------------------------------------

Analyzing move: e1e4
info depth 1 score cp 718 nodes 62 nps 118095 time 0 pv d4c3
info depth 2 score cp 320 nodes 1065 nps 185365 time 5 pv d4c3 e4f4 e5f7
info depth 3 score cp 298 nodes 5410 nps 228368 time 23 pv c7d6 c3d4 e5f3 g2f3
info depth 4 score cp 180 nodes 16085 nps 140321 time 114 pv c7e7 e4e5 g7e5 h7h6 f8e8
info depth 5 score cp 172 nodes 43391 nps 200093 time 216 pv c7f7 b2a3 f8e8 c2a4 e5c6 a4c6 c8c6 c3d4
info depth 6 score cp 182 nodes 142261 nps 225650 time 630 pv c7e7 e4f4 e5f7 c3d4 e7d6 f4f7 f8f7 c2g6 f7f8
info depth 7 score cp 213 nodes 421951 nps 230510 time 1830 pv c7d6 c3d4 c8c2 e4f4 e5f7 b2a3 d6a3 h7c2
  Score (for me): -213 (cp)
  Best Reply: c7d6
  Nodes: 421951

Analyzing move: b2a3
info depth 1 score cp -30 nodes 49 nps 142921 time 0 pv f8f7
info depth 2 score cp 166 nodes 138 nps 122552 time 1 pv f8f7 c2b3
info depth 3 score cp 410 nodes 429 nps 147803 time 2 pv f8f7 c2b3 d4c3
info depth 4 score cp 186 nodes 3340 nps 176419 time 18 pv f8f7 c2b3 e5c4 c3d4
info depth 5 score cp 330 nodes 8604 nps 149812 time 57 pv f8f7 c2b3 d4c3 h7f5 f7g8 b3e6 g8h8
info depth 6 score cp 204 nodes 36829 nps 161534 time 227 pv f8f7 c2b3 d4d3 a3c1 c7c3 h7f5 f7g8
info depth 7 score cp 0 nodes 64819 nps 74380 time 871 pv f8f7 e1e5 c7e5 h7g6 f7g8 g6h7 g8f7 h7g6 f7g8 g6h7 g8f7
  Score (for me): 0 (cp)
  Best Reply: f8f7
  Nodes: 64819

============================================================
Summary Comparison
------------------------------------------------------------
Move: b2a3   | Score:      0 | Best Reply: f8f7   | Nodes: 64819
Move: e1e4   | Score:   -213 | Best Reply: c7d6   | Nodes: 421951

```

---

## Lichess Puzzle 006NL
**FEN:** `1r6/k2qn1b1/p1N1p1p1/2PpPpN1/2n2P1P/p4B2/1PP2Q2/1K1R3R b - - 0 32`
**Correct Move:** `e7c6`
**Engine Move:** `d7c6`

### Static Evaluation (Root)
```
FEN: 1r6/k2qn1b1/p1N1p1p1/2PpPpN1/2n2P1P/p4B2/1PP2Q2/1K1R3R b - - 0 32
Side to Move: Black
Phase: 20 / 24 (MG Weight: 0.83, EG Weight: 0.17)
------------------------------------------------------------
Term                 | White MG | White EG | Black MG | Black EG |   Net MG |   Net EG
------------------------------------------------------------
Material             |     3470 |     3690 |     2970 |     3160 |      500 |      530
PST                  |      115 |      115 |       75 |      110 |       40 |        5
------------------------------------------------------------
Term (Net Only)      |   Net MG |   Net EG
----------------------------------------
King Safety          |      344 |      172
Pawn Structure       |       90 |      180
Coordination         |        0 |        0
Mobility             |        5 |        5
Threats              |      -10 |       -5
------------------------------------------------------------
Total                |      969 |      887
Final Interpolated Score (White Perspective): 955
Final Score (Side to Move): -955

```

### Search Comparison
Comparing Engine Move vs Correct Move:
```
FEN: 1r6/k2qn1b1/p1N1p1p1/2PpPpN1/2n2P1P/p4B2/1PP2Q2/1K1R3R b - - 0 32
Depth: 12
Time Limit: 2000ms
------------------------------------------------------------

Analyzing move: d7c6
info depth 1 score cp 172 nodes 210 nps 204031 time 1 pv b2b3
info depth 2 score cp 189 nodes 769 nps 241279 time 3 pv b2b3 a3a2 b1a1
info depth 3 score cp 150 nodes 4876 nps 288165 time 16 pv g5e6 g7e5 f4e5
info depth 4 score cp 121 nodes 17435 nps 282178 time 61 pv b2b3 c4e5 f4e5 a3a2 b1a2
info depth 5 score cp 199 nodes 42266 nps 272816 time 154 pv b2b3 a7a8 h4h5 c4e5 h5h6
info depth 6 score cp 172 nodes 57875 nps 150288 time 385 pv b2b3 c4e5 f4e5 g7e5 f2d2 a7a8
info depth 7 score cp 187 nodes 160357 nps 208489 time 769 pv b2b3 c4e5 f4e5 g7e5 d1d3 f5f4 h1e1
info depth 8 score cp 125 nodes 255473 nps 227572 time 1122 pv b2b3 c4e5 f4e5 g7e5 h1e1 e5f6 b1a2 f6g5
  Score (for me): -125 (cp)
  Best Reply: b2b3
  Nodes: 255473

Analyzing move: e7c6
info depth 1 score cp 386 nodes 187 nps 208599 time 0 pv d1d5
info depth 2 score cp 378 nodes 409 nps 241377 time 1 pv f3d5 e6d5
info depth 3 score cp 188 nodes 10670 nps 223259 time 47 pv b2b3 d5d4 g5e4
info depth 4 score cp 275 nodes 25019 nps 249997 time 100 pv b2b3 a7a8 f2e1 c6e5
info depth 5 score cp 236 nodes 64722 nps 250999 time 257 pv b2b3 g7e5 f3e2 c4b2 f4e5
info depth 6 score cp 312 nodes 141768 nps 260329 time 544 pv b2b3 g7e5 f4e5 d5d4 b1a1 c4e5
info depth 7 score cp 278 nodes 378627 nps 268197 time 1411 pv b2b3 c4e5 f4e5 g7e5 f2e3 a3a2 b1a2 a7a8
  Score (for me): -278 (cp)
  Best Reply: b2b3
  Nodes: 378627

============================================================
Summary Comparison
------------------------------------------------------------
Move: d7c6   | Score:   -125 | Best Reply: b2b3   | Nodes: 255473
Move: e7c6   | Score:   -278 | Best Reply: b2b3   | Nodes: 378627

```

---

## Lichess Puzzle 008o6
**FEN:** `Q4rk1/p1p3p1/6P1/8/3P4/7P/q3r3/B4RK1 w - - 2 35`
**Correct Move:** `a8f8`
**Engine Move:** `f1f8`

### Static Evaluation (Root)
```
FEN: Q4rk1/p1p3p1/6P1/8/3P4/7P/q3r3/B4RK1 w - - 2 35
Side to Move: White
Phase: 15 / 24 (MG Weight: 0.62, EG Weight: 0.38)
------------------------------------------------------------
Term                 | White MG | White EG | Black MG | Black EG |   Net MG |   Net EG
------------------------------------------------------------
Material             |     2030 |     2180 |     2200 |     2370 |     -170 |     -190
PST                  |       25 |       10 |       55 |        0 |      -30 |       10
------------------------------------------------------------
Term (Net Only)      |   Net MG |   Net EG
----------------------------------------
King Safety          |     -222 |     -111
Pawn Structure       |       10 |      -10
Coordination         |      -45 |      -65
Mobility             |      -33 |      -13
Threats              |      -55 |      -30
------------------------------------------------------------
Total                |     -545 |     -409
Final Interpolated Score (White Perspective): -494
Final Score (Side to Move): -494

```

### Search Comparison
Comparing Engine Move vs Correct Move:
```
FEN: Q4rk1/p1p3p1/6P1/8/3P4/7P/q3r3/B4RK1 w - - 2 35
Depth: 12
Time Limit: 2000ms
------------------------------------------------------------

Analyzing move: f1f8
info depth 1 score mate 0 nodes 1 nps 4161 time 0 pv
  Score (for me): 30000 (cp)
  Best Reply: None
  Nodes: 1

Analyzing move: a8f8
info depth 1 score mate 0 nodes 1 nps 8388 time 0 pv
  Score (for me): 30000 (cp)
  Best Reply: None
  Nodes: 1

============================================================
Summary Comparison
------------------------------------------------------------
Move: f1f8   | Score:  30000 | Best Reply: None   | Nodes: 1
Move: a8f8   | Score:  30000 | Best Reply: None   | Nodes: 1

```

---
