import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TestSearchCoreContracts(unittest.TestCase):
    def test_diagnostics_are_opt_in(self):
        engine_types = (ROOT / 'chess_engine/classical/engine_types.py').read_text(encoding='utf-8')
        search = (ROOT / 'chess_engine/classical/search.py').read_text(encoding='utf-8')

        self.assertIn('continuation_correction_history=None, enable_diagnostics=False', engine_types)
        self.assertIn("('diag_enabled', numba.boolean)", engine_types)
        self.assertIn('if search_context.diag_enabled:', search)
        self.assertIn('def _diag_add(', search)

    def test_exclusion_search_without_alternative_fails_low(self):
        search = (ROOT / 'chess_engine/classical/search.py').read_text(encoding='utf-8')
        terminal = search.index('if legal_moves_tried == 0 and pruned_moves == 0:')
        exclusion = search.index('if is_exclusion_search:', terminal)
        fail_low = search.index('np.int32(original_alpha)', exclusion)
        mate = search.index('if is_currently_in_check:', exclusion)

        self.assertLess(exclusion, mate)
        self.assertLess(fail_low, mate)
        self.assertNotIn('searched_move_count', search)

    def test_capture_see_for_lmr_is_lazy(self):
        search = (ROOT / 'chess_engine/classical/search.py').read_text(encoding='utf-8')
        move_loop = search.index('# --- Pre-move checks for extensions and move type ---')
        pruning = search.index('# --- NEW: Shallow Depth Pruning', move_loop)
        lazy_comment = search.index('Capture LMR consumes SEE(0)', pruning)
        make_move = search.index('# --- Make the move ---', lazy_comment)

        self.assertNotIn('is_bad_capture = not _see_ge_jit', search[move_loop:pruning])
        self.assertLess(lazy_comment, make_move)

    def test_qsearch_quiet_evasions_use_a_bounded_history_bucket(self):
        constants = (ROOT / 'chess_engine/classical/constants.py').read_text(encoding='utf-8')
        heuristics = (ROOT / 'chess_engine/classical/search_heuristics.py').read_text(encoding='utf-8')
        scorer = heuristics.index('def score_captures_with_tt_lazy(')
        next_scorer = heuristics.index('def score_captures(', scorer)
        body = heuristics[scorer:next_scorer]

        self.assertIn('QS_EVASION_CAPTURE_BUCKET = 20000', constants)
        self.assertIn('QS_EVASION_QUIET_CLAMP = 5000', constants)
        self.assertIn('base_score = get_quiet_stat_score(', body)
        self.assertIn('QS_EVASION_CAPTURE_BUCKET', body)
        self.assertIn('QS_EVASION_QUIET_CLAMP', body)

    def test_assistant_discards_warmup_search_state(self):
        assistant = (ROOT / 'assistant.py').read_text(encoding='utf-8')
        warmup_bestmove = assistant.index('if self.warming_up:', assistant.index('elif msg["type"] == "bestmove":'))
        restore_book = assistant.index('# Restore user OwnBook preference', warmup_bestmove)
        reset = assistant.index('self.engine.send_command("ucinewgame")', warmup_bestmove)

        self.assertLess(reset, restore_book)

    def test_qsearch_checks_bypass_delta_but_not_final_see(self):
        search = (ROOT / 'chess_engine/classical/search.py').read_text(encoding='utf-8')
        delta = search.index('if delta_prune_candidate:')
        protected = search.index('_diag_add(search_context, DIAG_Q_CHECK_PROTECT)', delta)
        final_see = search.index('if ENABLE_SEE_IN_QUIESCENCE', protected)

        self.assertLess(delta, protected)
        self.assertLess(protected, final_see)
        self.assertNotIn('q_ply >', search[delta:protected])

    def test_tournament_warmup_can_trace_both_engines(self):
        match_core = (ROOT / 'tools/match_core.py').read_text(encoding='utf-8')

        self.assertIn('show_info: bool = False', match_core)
        self.assertIn('print(f"[warmup {self.name}] {line}", flush=True)', match_core)
        self.assertIn('verbose=show_info', match_core)
        self.assertIn('show_info=(worker_id == 0)', match_core)

    def test_qsearch_forced_evasion_legality_remains_after_scoring(self):
        search = (ROOT / 'chess_engine/classical/search.py').read_text(encoding='utf-8')
        scorer = search.index('score_captures_with_tt_lazy(')
        geometry = search.index('# Baseline qsearch legality geometry', scorer)
        selection = search.index('# Lazy Selection Sort: find the best remaining move', geometry)
        double_check = search.index('qs_checker_count > 1', selection)
        make_move = search.index('unmake_info = make_move(', double_check)

        self.assertLess(scorer, geometry)
        self.assertLess(geometry, selection)
        self.assertIn('SQUARES_BETWEEN[our_king_sq_qs, qs_checker_sq]', search[geometry:selection])
        self.assertLess(double_check, make_move)

    def test_qsearch_evasion_prefilter_is_shadow_only(self):
        search = (ROOT / 'chess_engine/classical/search.py').read_text(encoding='utf-8')
        shadow = search.index('# P0 shadow diagnostics')
        scorer = search.index('score_captures_with_tt_lazy(', shadow)
        body = search[shadow:scorer]

        self.assertIn('if is_currently_in_check and search_context.diag_enabled:', body)
        self.assertIn('_classify_main_in_check_move(', body)
        self.assertIn('DIAG_Q_EVASION_PREFILTER', body)
        self.assertNotIn('moves[compacted_move_count]', search)
        self.assertNotIn('move_count = compacted_move_count', search)

    def test_main_quiet_checks_use_check_see_route(self):
        search = (ROOT / 'chess_engine/classical/search.py').read_text(encoding='utf-8')
        shallow = search.index('# --- NEW: Shallow Depth Pruning')
        check_route = search.index('if pre_move_gives_check:', shallow)
        make_move = search.index('# --- Make the move ---', check_route)

        self.assertIn('TUNE_SEE_CAP_MARGIN', search[check_route:make_move])
        self.assertIn('DIAG_MAIN_CHECK_SEE_ROUTE', search[check_route:make_move])

    def test_main_quiet_scorer_uses_baseline_local_geometry(self):
        search = (ROOT / 'chess_engine/classical/search.py').read_text(encoding='utf-8')
        heuristics = (ROOT / 'chess_engine/classical/search_heuristics.py').read_text(encoding='utf-8')
        call = search[search.index('score_quiets('):search.index('search_return_type =', search.index('score_quiets('))]
        scorer_start = heuristics.index('def score_quiets(')
        scorer_end = heuristics.index('def compute_lmr_reduction_1024(', scorer_start)
        scorer = heuristics[scorer_start:scorer_end]

        self.assertNotIn('check_sq_pawn, check_sq_knight', call)
        self.assertNotIn('check_sq_pawn_arg', scorer)
        self.assertIn('check_sq_computed = False', scorer)
        self.assertIn('get_bishop_attacks(enemy_king_sq, all_occ)', scorer)
        self.assertIn('get_rook_attacks(enemy_king_sq, all_occ)', scorer)
        self.assertIn('DIAG_MAIN_SCORE_GEOMETRY_REBUILD', scorer)

    def test_p1_legality_prefilter_is_shadow_only(self):
        search = (ROOT / 'chess_engine/classical/search.py').read_text(encoding='utf-8')
        move_loop = search.index('# --- Pre-move checks for extensions and move type ---')
        shadow = search.index('# P1 shadow diagnostics', move_loop)
        shallow = search.index('# --- NEW: Shallow Depth Pruning', shadow)
        legality = search.index('# --- Pre-make Legality Fast Path', shallow)

        shadow_body = search[shadow:shallow]
        self.assertIn('if search_context.diag_enabled:', shadow_body)
        self.assertIn('DIAG_MAIN_QUIET_PRE_PIN_ILLEGAL', shadow_body)
        self.assertIn('DIAG_MAIN_QUIET_PRE_KING_ILLEGAL', shadow_body)
        self.assertNotIn('continue', shadow_body)
        self.assertLess(shallow, legality)

    def test_aspiration_diagnostics_are_depth_indexed_and_opt_in(self):
        constants = (ROOT / 'chess_engine/classical/constants.py').read_text(encoding='utf-8')
        search = (ROOT / 'chess_engine/classical/search.py').read_text(encoding='utf-8')

        self.assertIn('DIAG_ASPIRATION_DEPTH_SLOTS = MAX_PLY + 1', constants)
        self.assertIn('DIAG_ASPIRATION_FAIL_LOW_BASE', constants)
        self.assertIn('DIAG_ASPIRATION_FAIL_HIGH_BASE', constants)
        self.assertIn('DIAG_ASPIRATION_WASTED_NODES_BASE', constants)
        self.assertIn('ASPIRATION_WINDOW_BASE = 20', constants)
        self.assertIn('aspiration_depth_slot = min(', search)
        self.assertIn('DIAG_ASPIRATION_ITERATIONS_BASE + aspiration_depth_slot', search)
        self.assertIn('DIAG_ASPIRATION_WASTED_NODES_BASE + aspiration_depth_slot', search)
        self.assertIn('def _collect_aspiration_diag(', search)
        self.assertIn('"aspiration_by_depth": by_depth', search)

    def test_p3_deep_cost_diagnostics_are_opt_in_and_semantics_preserving(self):
        constants = (ROOT / 'chess_engine/classical/constants.py').read_text(encoding='utf-8')
        search = (ROOT / 'chess_engine/classical/search.py').read_text(encoding='utf-8')
        bench = (ROOT / 'tools/hist_diag_bench.py').read_text(encoding='utf-8')

        self.assertIn('DIAG_Q_CAP_NONCHECK', constants)
        self.assertIn('DIAG_Q_CAP_CHECK', constants)
        self.assertIn('DIAG_Q_CAP_FORCING_MOVES', constants)
        self.assertIn('DIAG_MAIN_IN_CHECK_PRE_ILLEGAL', constants)
        self.assertIn('DIAG_TT_VERIFY_TRY', constants)
        self.assertIn('DIAG_TT_VERIFY_SAVED_CUT', constants)
        self.assertIn('_classify_main_in_check_move(', search)
        self.assertIn('DIAG_Q_CAP_FORCING', search)
        self.assertIn('DIAG_MAIN_IN_CHECK_FALLBACK', search)
        self.assertIn('DIAG_TT_VERIFY_SKIP', search)
        # P3 must remain a measurement-only pass: the legacy TT cutoff and
        # qsearch cap returns are still present after their counters.
        self.assertIn('if run_cutoff:', search)
        self.assertIn('if q_ply >= MAX_QUIESCENCE_DEPTH * 2:', search)
        self.assertIn('if q_ply >= MAX_QUIESCENCE_DEPTH:', search)
        self.assertIn('"tt_verify_reject_pct"', search)
        self.assertIn('"q_cap_forcing_pct"', search)
        self.assertIn('"tt_verify_saved_cut_pct"', bench)

    def test_main_in_check_prefilter_is_shadow_only(self):
        search = (ROOT / 'chess_engine/classical/search.py').read_text(encoding='utf-8')
        picker = search.index('def get_next_move(')
        captures = search.index('if is_in_check and search_context.diag_enabled:', picker)
        capture_score = search.index('score_captures_lazy(', captures)
        quiets = search.index('if is_in_check and search_context.diag_enabled:', capture_score)
        quiet_score = search.index('score_quiets(', quiets)
        call = search.index('main_checker_count_s,', quiet_score)

        self.assertIn('_classify_and_diag_main_in_check_move(', search[captures:capture_score])
        self.assertIn('_classify_and_diag_main_in_check_move(', search[quiets:quiet_score])
        self.assertNotIn('compacted_captures', search)
        self.assertNotIn('compacted_quiets', search)
        self.assertLess(captures, capture_score)
        self.assertLess(quiets, quiet_score)
        self.assertIn('main_evasion_targets_s', search[call:call + 120])

    def test_p4_diagnostics_are_shadow_only_after_production_ab(self):
        constants = (ROOT / 'chess_engine/classical/constants.py').read_text(encoding='utf-8')
        search = (ROOT / 'chess_engine/classical/search.py').read_text(encoding='utf-8')
        heuristics = (ROOT / 'chess_engine/classical/search_heuristics.py').read_text(encoding='utf-8')
        bench = (ROOT / 'tools/hist_diag_bench.py').read_text(encoding='utf-8')

        self.assertIn('DIAG_ORDER_TRY_BASE', constants)
        self.assertIn('DIAG_ORDER_CUT_BASE', constants)
        self.assertIn('DIAG_LMR_RESEARCH_REJECT', constants)
        self.assertIn('DIAG_POST_LMR_BONUS_SAMPLE', constants)
        self.assertIn('DIAG_POST_LMR_MALUS_SAMPLE', constants)

        lmr_start = search.index('# --- 1024-scale LMR (Phase B) ---')
        lmr_end = search.index('unmake_move(piece_bbs, occupancy_bbs, game_state, move, unmake_info)', lmr_start)
        lmr_body = search[lmr_start:lmr_end]
        self.assertIn('if search_context.diag_enabled:', lmr_body)
        self.assertIn('DIAG_LMR_RESEARCH_KEEP', lmr_body)
        self.assertIn('DIAG_LMR_RESEARCH_REJECT', lmr_body)
        self.assertIn('DIAG_POST_LMR_BONUS_SAMPLE', lmr_body)
        self.assertIn('DIAG_POST_LMR_MALUS_SAMPLE', lmr_body)
        self.assertNotIn('update_continuation_histories_for_move(', lmr_body)
        lmr_stat_start = heuristics.index('def get_lmr_stat_score(')
        lmr_stat_end = heuristics.index('def get_lmr_reduction(', lmr_stat_start)
        self.assertNotIn('butterfly_history', heuristics[lmr_stat_start:lmr_stat_end])

        cutoff = search.index('if alpha >= beta:', lmr_end)
        history_update = search.index('# Phase 1a: linear bonus/malus', cutoff)
        self.assertIn('DIAG_ORDER_CUT_BASE + diag_picker_source_s', search[cutoff:history_update])
        self.assertIn('"lmr_research_reject_pct"', search)
        self.assertIn('"post_lmr_bonus_sample"', search)
        self.assertIn('"order_good_quiet_cut_pct"', bench)
        self.assertIn('"post_lmr_malus_sample"', bench)
        self.assertIn('"order_source_cut_coverage_pct"', bench)
        self.assertIn('mean(num) / mean(den)', bench)

        self.assertIn('DIAG_HP_CONT_COALITION_DECISIVE', constants)
        self.assertIn('get_quiet_stat_components(', search)
        hp_start = search.index('# P5/P6 attribution:')
        hp_end = search.index('continue', hp_start)
        self.assertIn('if search_context.diag_enabled:', search[hp_start - 120:hp_end])
        self.assertNotIn('update_', search[hp_start:hp_end])
        self.assertIn('hist_prune_cont_coalition_decisive_pct', search)
        self.assertIn('HISTORY_PRUNE_CONT1_FLOOR = -4096', constants)
        prune_start = search.index('# --- Step 14b: History Pruning ---')
        prune_end = search.index('# --- Step 14c: Futility Pruning', prune_start)
        self.assertIn('HISTORY_PRUNE_CONT1_FLOOR,', search[prune_start:prune_end])
        self.assertIn('DIAG_HP_CONT1_CAP_RESCUE', search[prune_start:prune_end])
        scorer_start = heuristics.index('def score_quiets(')
        scorer_end = heuristics.index('def compute_lmr_reduction_1024(', scorer_start)
        self.assertNotIn('HISTORY_PRUNE_CONT1_FLOOR', heuristics[scorer_start:scorer_end])
        self.assertIn('--sequence-plies', bench)
        self.assertIn('--paired-cold', bench)
        self.assertIn('--keep-sequence-tt', bench)
        self.assertIn('def _search_sequence_seed(', bench)
        self.assertIn('game_history.append(int(g_state[4]))', bench)
        self.assertIn('make_move(p_bbs, o_bbs, g_state, best_move)', bench)
        self.assertIn('warm_ctx.transposition_table.fill(0)', bench)

    def test_uci_reuses_context_and_resets_owned_history(self):
        main = (ROOT / 'main.py').read_text(encoding='utf-8')
        loop = main.index('while True:')
        allocation = main.index('global_search_context = SearchContext(')
        go_branch = main.index('elif command == "go":')

        self.assertLess(allocation, loop)
        self.assertNotIn('SearchContext(', main[go_branch:])
        self.assertIn('global_search_context.counter_moves.fill(0)', main)
        self.assertIn('global_search_context.low_ply_history.fill(0)', main)

    def test_pruning_and_reduction_safety_guards_remain_wired(self):
        search = (ROOT / 'chess_engine/classical/search.py').read_text(encoding='utf-8')

        self.assertIn('not is_currently_in_check and not is_pv and not is_exclusion_search', search)
        self.assertIn('not is_currently_in_check and not follow_pv', search)
        self.assertIn('can_prune_quiet = not pre_move_gives_check', search)
        self.assertIn('if is_giving_check_after_move:\n                        lmr = max(0, lmr - 1)', search)
        self.assertIn('ENABLE_ALPHA_RAISE_DEPTH_REDUCTION and ply > 0', search)
        self.assertIn('and not is_decisive(evaluation)', search)

    def test_insufficient_material_and_check_contract(self):
        from chess_engine.classical.fen_parser import parse_fen
        from chess_engine.classical.move_generator import has_sufficient_material

        # White Knight vs Black Bishop (can mate: e7g6#)
        pb_kn_kb, _, _ = parse_fen("6bk/4N3/7K/8/8/8/8/8 w - - 0 1")
        self.assertTrue(has_sufficient_material(pb_kn_kb))

        # White Knight vs Black Knight (can mate via helpmate)
        pb_kn_kn, _, _ = parse_fen("7k/4N3/7K/8/8/8/8/5n2 w - - 0 1")
        self.assertTrue(has_sufficient_material(pb_kn_kn))

        # Opposite color bishops (can mate)
        pb_kb_kb_opp, _, _ = parse_fen("kb6/1B6/K7/8/8/8/8/8 b - - 0 1")
        self.assertTrue(has_sufficient_material(pb_kb_kb_opp))

        # Same color bishops (cannot mate -> dead position)
        pb_kb_kb_same, _, _ = parse_fen("k7/1b6/K7/3B4/8/8/8/8 b - - 0 1")
        self.assertFalse(has_sufficient_material(pb_kb_kb_same))

        # Lone king vs minor
        pb_kn_k, _, _ = parse_fen("6Nk/8/7K/8/8/8/8/8 b - - 0 1")
        self.assertFalse(has_sufficient_material(pb_kn_k))

        # Verify search.py guards draw claims against king in check
        search = (ROOT / 'chess_engine/classical/search.py').read_text(encoding='utf-8')
        self.assertIn('if halfmove_clock >= FIFTY_MOVE_RULE_LIMIT or not has_sufficient_material(piece_bbs):', search)
        self.assertIn('if not _is_in_check_jit(piece_bbs, occupancy_bbs, game_state):', search)


if __name__ == '__main__':
    unittest.main()

