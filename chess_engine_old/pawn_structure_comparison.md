# 兵型結構評估差異報告：Stockfish 11 vs Engine V2

這份報告詳細比較了 **Stockfish 11** 在兵型結構（Pawn Structure）的評估邏輯，與我們目前 `chess_engine_v2` 之間的本質差異。雖然我們的引擎已經具備了基本的孤兵、重疊兵與過兵觀念，甚至加入了「同色疊兵阻擋」的進階判斷，但對照頂尖引擎的設計，我們仍在以下幾個關鍵的「結構型判斷」上存在簡化的盲點。

## 1. 推進路徑的安全控制 (Path Safety to Promotion)

### Engine V2 的現狀
目前的過兵 (Passed Pawn) 評估主要依賴「推進距離」與「國王靠近度 (King Proximity)」。
* 若敵方國王離我們的過兵很近，我們就會扣除這隻過兵的潛在威脅分數（認為它會被國王吃掉或擋住）。
* 但我們並**沒有**去檢查這條升變之路 (`squaresToQueen`) 上，是否佈滿了敵方的攻擊火力（例如車、象）。

### Stockfish 11 的作法 ([evaluate.cpp](file:///C:/Users/ren%20cian/.gemini/antigravity/brain/b3d85ccf-7378-4335-bdac-18b60d652784/sf11_tmp/src/evaluate.cpp) Line 612-634)
當 Stockfish 判定一隻過兵前方沒有敵方兵阻擋時，它會進行極其細緻的路徑火力掃描：
1. 提取前方所有的升變路徑 `squaresToQueen` 與火力控制網 `unsafeSquares`。
2. 如果這條路徑 **完全沒有被敵方攻擊 (100% safe)**，給予高達 **+35** 的額外巨大係數加上推進權重。
3. 如果只有前面的某些格子危險，但下一格 (blockSq) 安全，給予 **+20**。
4. 甚至檢查敵方攻擊路徑的棋子與我方防守的棋子，如果下一格我們有保護，給予額外加分 (`k += 5`)。
這使得 SF 懂得創造「能安全抵達底線的過兵」，避免盲目製造出「看似過兵但立刻成為靶子」的假像。

---

## 2. 被「非兵棋子」阻擋的通路兵 (Non-Pawn Blockaders)

### Engine V2 的現狀
在最新的修改中，我們成功加入了「前方有自己人 (Friendly Pawn) 阻擋過兵」時分數減半的規則。但如果這隻過兵的前方，站著的是**敵方的馬**或是**車**呢？我們依然會給予它全額的過兵推進獎勵。

### Stockfish 11 的作法 ([evaluate.cpp](file:///C:/Users/ren%20cian/.gemini/antigravity/brain/b3d85ccf-7378-4335-bdac-18b60d652784/sf11_tmp/src/evaluate.cpp) Line 612)
Stockfish 會直接判定 `if (pos.empty(blockSq))`。
* 除非過兵的「正前方格子」是完全淨空 ([empty](file:///C:/Users/ren%20cian/.gemini/antigravity/brain/b3d85ccf-7378-4335-bdac-18b60d652784/sf11_tmp/src/position.h#210-213)) 的，否則它**不會**進入上述發放巨大的「路徑安全與國王威脅」獎勵的邏輯。
* 若前方停著敵方的輕/重子（典型的 Blockader 戰術），這隻過兵的價值將會被封印。我們目前因為忽視了非兵棋子的阻擋，常常高估了被死鎖的過兵價值。

---

## 3. 相連作戰的普通兵陣 (Connected Pawns & Phalanx)

### Engine V2 的現狀
引擎只獎勵「相連的 **通路兵 (Connected Passed Pawns)**」。
對於一般的兵，我們只有懲罰（孤兵、重疊、落後），沒有針對「漂亮兵陣」的正向獎勵。

### Stockfish 11 的作法 ([pawns.cpp](file:///C:/Users/ren%20cian/.gemini/antigravity/brain/b3d85ccf-7378-4335-bdac-18b60d652784/sf11_tmp/src/pawns.cpp) Line 133-138)
SF 具備對「普通兵鏈」的強大保護意識：
* **Phalanx (並排兵)**：在同一橫排相鄰的兵（例如 d4 和 e4）。
* **Supported (受保護兵)**：受斜後方同色兵保護的兵。
若一般的兵具備這兩個屬性，SF 會基於對應的 Rank 查表給予 `Connected` 分數。這就是為什麼 Stockfish 開局與中局總是能走出異常堅固的陣型，而我們的引擎相對容易被撕裂防線。

---

## 4. 半開放線上的落後/孤兵缺點放大 (Half-Open File Vulnerabilities)

### Engine V2 的現狀
我們給予 Isolated Pawn (孤兵) 和 Backward Pawn (落後兵) 固定的扣分常數 (`-10`, `-15` 左右)。

### Stockfish 11 的作法 ([pawns.cpp](file:///C:/Users/ren%20cian/.gemini/antigravity/brain/b3d85ccf-7378-4335-bdac-18b60d652784/sf11_tmp/src/pawns.cpp) Line 141-147)
SF 將其區分成 `Isolated` 與 `WeakUnopposed`。
* `Opposed` 代表這隻弱兵（孤兵或落後兵）同一條 File 上還有**敵方的兵**擋著。
* 如果是 `!Opposed` (半開放線)，這意味著這隻弱兵直接曝露在敵方重火力的直射區（沒有敵方兵做緩衝）。此時 SF 會加上一個幾乎等同於孤兵懲罰兩倍的 `WeakUnopposed` 扣分。
我們缺乏這種「動態弱點放大」機制，導致引擎有時不介意在半開放線上留下孤兵而被重子輕易吃掉。

---

## 5. 國王的安全庇護 (Pawn Shelter & Storm)

### Engine V2 的現狀
我們採用了非線性二次方的 Pawn Storm (兵風暴) 模型，並基於王城附近的火力點進行扣分，這點在防守上已經表現不錯。

### Stockfish 11 的作法 ([pawns.cpp](file:///C:/Users/ren%20cian/.gemini/antigravity/brain/b3d85ccf-7378-4335-bdac-18b60d652784/sf11_tmp/src/pawns.cpp) Line 185-212)
SF 將保護國王的兵（Shelter）依據「離王城的距離」與「目前的 Rank」建立了一個複雜的二維表格 `ShelterStrength[distance][rank]`。
更重要的是，若敵方兵向前推進（Storm），SF 會精確計算 **"Blocked Storm"** 與 **"Unblocked Storm"**。如果敵方推進的兵被我方兵頂住（Blocked），威脅會大幅減弱；若是毫無阻礙（Unblocked）撲來，則面臨毀滅性的二次方扣分。

---

## 結論與下一步建議

目前的 `engine_v2` 在兵型結構的基礎上是健全的，但上述的 **1 (推進路徑控制)** 與 **2 (非兵棋子阻擋)** 是過兵評估的最後一塊拼圖。
如果要在不增加過度運算複雜度的前提下搾取 Elo：
1. **立即性修改**：可以優先加入「非兵棋子阻擋過兵 (Non-Pawn Blockaders)」的分數衰減，這只需一行位元運算 `occupancy_bbs` 即可完成，C/P 值極高。
2. **結構性修改**：若要讓中局更堅固，實作一般兵的「相連/並排陣型獎勵 (Phalanx)」能讓引擎學會更好的佈陣。
