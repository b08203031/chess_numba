# 已結案報告總覽（2026）

本目錄收納 **搜尋／歷史啟發／NMP 相關、已完成並有最終定論的階段計畫與戰役報告**。內容僅供考古與查證歷史數據，**不再維護**；現行規格請看各列「現行取代文件」。  
每份檔案開頭已加註「已結案歸檔」橫幅與原路徑；檔內相對連結已隨搬移重寫。

| 報告 | 範圍 | 最終定論 | 現行取代文件 |
| :--- | :--- | :--- | :--- |
| [REPORT_2026-07-15_16_search_campaign](REPORT_2026-07-15_16_search_campaign.md) | 2026-07 搜尋樹效率戰役（200k nodes） | 剪枝 margin 座標下降（R1–R3）已達平台；抄 SF11 NMP / ProbCut 公式無法瘦樹；對戰約 72–74 局後手動中止（SPRT 仍 Continue、無優勢） | [SEARCH_ANALYSIS](../../chess_engine/classical/SEARCH_ANALYSIS.md) |
| [NMP_EXPERIMENT_PLAN](NMP_EXPERIMENT_PLAN.md) | 2026-07-16 NMP 節點效率診斷與實驗計畫 | 診斷：關 NMP → 約 1.18× 節點；計畫從未改生產常數，已由 Phase 8 取代 | [Phase 8 計畫](PHASE8_NMP_SF19_ALIGNMENT_PLAN.md) |
| [SEARCH_OPTIMIZATION_DIAGNOSTIC_20260804](SEARCH_OPTIMIZATION_DIAGNOSTIC_20260804.md) | 搜尋診斷、SF11/SF18 對照與 P0–P6 | P0/P1/P3 組合未通過固定節點對戰，production 回退；相關 classifier 保留為 `diag_enabled` shadow diagnostics，不影響候選集合、排序或剪枝 | [SEARCH_ANALYSIS](../../chess_engine/classical/SEARCH_ANALYSIS.md) |
| [HISTORY_REFORM_PLAN](HISTORY_REFORM_PLAN.md) | 歷史啟發五階段重構 | 現用基線凍結於 Phase 1a；P2 已回退；P3–5 暫緩（禁止再開整包 P2/P3） | [HISTORY_HEURISTICS_CN](../../chess_engine/classical/HISTORY_HEURISTICS_CN.md) |
| [QUIET_ORDERING_DECOUPLING_PLAN](QUIET_ORDERING_DECOUPLING_PLAN.md) | MovePicker 排序與歷史剪枝解耦 | 架構解耦完成；其後 E54（排序權重平滑化對 SF19）裁定回滾（2026-09-26） | [SEARCH_EXP_LOG](../../tournament_analysis/SEARCH_EXP_LOG.md) |
| [SEARCH_PROPOSALS_E47_E50](SEARCH_PROPOSALS_E47_E50.md) | E47–E50 搜尋候選提案 | E47（ProbCut 自適應，+12.17 Elo）、E48（LMR ALL-Node）、E50（廢除將軍延伸，400 局 @ n300k：53.25%，+22.62 ±20.50 Elo）均已採納 | [SEARCH_EXP_LOG](../../tournament_analysis/SEARCH_EXP_LOG.md) |
| [SEARCH_AUDIT_AND_SF19_PROPOSALS_20261019](SEARCH_AUDIT_AND_SF19_PROPOSALS_20261019.md) | 搜尋全檔審計與 SF19 對齊路線圖 | Bug 修復與路線圖已落地；E84（SEE 修補 + Step 14 前置合法性預檢，600 局 LTC 400k）+10.43 Elo 採納並同步 `classical_old` | [SEARCH_ANALYSIS](../../chess_engine/classical/SEARCH_ANALYSIS.md)、[SEE_ANALYSIS](../../chess_engine/classical/SEE_ANALYSIS.md) |
| [PHASE7_HISTORY_LMR_ALIGNMENT_PLAN](PHASE7_HISTORY_LMR_ALIGNMENT_PLAN.md) | Phase 6 Plan A 複盤與歷史動力鏈（Phase 7） | Plan A 400 局 @ 300k：+19.13 ±22.05 Elo；Phase 7 的 1,000 局為 +0.7 ±14.0 Elo（`inconclusive_at_cap`），維持 Plan A 常數（956 / 1600 / 128 / 1024）；42 槽位免重編譯調參架構與 `tune_colab_runner_p7.ipynb` 保留 | [tune_search/README](../../tune_search/README.md) |
| [PHASE8_NMP_SF19_ALIGNMENT_PLAN](PHASE8_NMP_SF19_ALIGNMENT_PLAN.md) | NMP 體系對齊 SF19 Step 10 | 600 局驗收 136W / 349D / 115L（51.75%，+12.17 ±17.70 Elo，平均深度 +0.47），已同步 `classical_old` | [SEARCH_ANALYSIS](../../chess_engine/classical/SEARCH_ANALYSIS.md) |
| [REPORT_2026-10-14_nmp_alignment_campaign](REPORT_2026-10-14_nmp_alignment_campaign.md) | Phase 8 戰役總結（含 Numba 結構化陣列別名修復） | 同上；NMP 極簡架構採納 | [SEARCH_ANALYSIS](../../chess_engine/classical/SEARCH_ANALYSIS.md) |

> 歸檔準則：計畫已實施／回退並有最終 Elo 或回滾結論，且現行規格已由其他主文件承接。需要重啟某條路線時，先讀對應報告的「定論」章節以避免重複踩坑。

> 註：部分報告內的附屬 JSON／圖表（如 `archive/tournament_analysis_2026-08/`）與上游 `stockfish_repo/`、`stockfish_11/` 連結，因未納入版控或僅為空目錄，點擊可能失效；僅作歷史佐證。
