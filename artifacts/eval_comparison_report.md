# Static Evaluation Comparison Report

This report compares the static evaluations of different positions from three perspectives:
1. **Our Classical Evaluation** (Tapered evaluation: material, PST, mobility, king safety, pawn structure, etc.)
2. **Our NNUE Evaluation** (8-bucket network trained on HalfKAv2_hm features)
3. **Stockfish 17.1 NNUE Evaluation** (Pure NNUE network evaluation)
4. **Stockfish 17.1 Final Evaluation** (Stockfish's final output including scale factor and heuristics)

*Note: All scores are presented in centipawns (cp) and converted to **White's perspective** (positive scores favor White, negative scores favor Black).*

## Comparison Table

| Position Description | STM | Our Classical | Our NNUE | Stockfish NNUE | Stockfish Final |
| :--- | :---: | :---: | :---: | :---: | :---: |
| White missing a2 pawn | W | -56 | -15 | -34 | -42 |
| Black missing a7 pawn | W | +76 | +56 | +50 | +64 |
| White missing b2 pawn | W | -91 | -54 | -89 | -115 |
| Black missing b7 pawn | W | +111 | +114 | +128 | +168 |
| White missing c2 pawn | W | -85 | -22 | -92 | -120 |
| Black missing c7 pawn | W | +105 | +82 | +126 | +166 |
| White missing d2 pawn | W | -15 | -20 | -100 | -131 |
| Black missing d7 pawn | W | +35 | +76 | +136 | +179 |
| White missing e2 pawn | W | -28 | -36 | -103 | -135 |
| Black missing e7 pawn | W | +48 | +92 | +139 | +183 |
| White missing f2 pawn | W | -145 | -95 | -119 | -156 |
| Black missing f7 pawn | W | +165 | +190 | +155 | +204 |
| White missing g2 pawn | W | -91 | -80 | -105 | -137 |
| Black missing g7 pawn | W | +111 | +165 | +142 | +186 |
| White missing h2 pawn | W | -15 | -18 | -33 | -41 |
| Black missing h7 pawn | W | +35 | +64 | +62 | +80 |
| White missing b1 knight | W | -241 | -804 | -344 | -445 |
| Black missing b8 knight | W | +260 | +972 | +398 | +522 |
| White missing g1 knight | W | -245 | -712 | -362 | -472 |
| Black missing g8 knight | W | +264 | +911 | +399 | +524 |
| White missing c1 bishop | W | -284 | -1011 | -401 | -518 |
| Black missing c8 bishop | W | +303 | +1104 | +438 | +570 |
| White missing f1 bishop | W | -300 | -952 | -381 | -488 |
| Black missing f8 bishop | W | +319 | +1098 | +419 | +545 |
| White missing a1 rook | W | -469 | -1132 | -514 | -496 |
| Black missing a8 rook | W | +488 | +1238 | +533 | +507 |
| White missing h1 rook | W | -433 | -1086 | -529 | -515 |
| Black missing h8 rook | W | +452 | +1291 | +550 | +533 |
| White missing d1 queen | W | -888 | -1481 | -787 | -522 |
| Black missing d8 queen | W | +907 | +1779 | +996 | +598 |
| Puzzle 1 (Middlegame sac) | W | -27 | -62 | +30 | +40 |
| Puzzle 2 (Endgame pawn) | W | -321 | -220 | -53 | -52 |
| Puzzle 3 (Endgame minor) | W | -39 | +77 | +75 | +83 |
| Puzzle 4 (Middlegame exposed king) | W | -374 | +59 | -63 | -142 |
| Puzzle 5 (Middlegame active defense) | B | +126 | +89 | -4 | -10 |
