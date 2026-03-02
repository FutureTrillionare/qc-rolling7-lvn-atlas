from AlgorithmImports import *
from collections import deque
from datetime import timedelta
import math
import statistics
import pandas as pd


class Rolling7LVNAtlasMES_QC(QCAlgorithm):

    def initialize(self):
        # ---- Backtest window (edit freely) ----
        self.set_start_date(2024, 1, 2)
        self.set_end_date(2024, 3, 1)
        self.set_cash(100000)

        # Use Chicago timezone since your sessions/MOR are based on it
        self.set_time_zone(TimeZones.CHICAGO)

        # ---- Strategy parameters (match your Sierra defaults) ----
        self.ROLLING_DAYS = 7

        # LVN detection
        self.MIN_PROM = 0.25
        self.MIN_WIDTH_TICKS = 4
        self.MERGE_CENTER_DIST_TICKS = 2
        self.PROM_WINDOW = 200

        # Traffic map
        self.HVN_Q = 0.75
        self.LVN_Q = 0.25
        self.MIN_SHELF_W = 10
        self.MIN_LANE_W = 8
        self.MERGE_ZONE_GAP = 2
        self.SMOOTH_FRICTION_3TAP = True

        # PoLR
        self.POLR_ATR60_MULT = 1.5
        self.POLR_R_MIN = 60
        self.POLR_R_MAX = 240
        self.POLR_BIAS_THRESH = 20.0
        self.POLR_SHELF_PENALTY = 15.0
        self.POLR_COST_W = 70.0
        self.POLR_LANE_W = 100.0

        # Plan/Execution
        self.FRONT_RUN = 2
        self.PLAN_STOP_BUF = 6
        self.S1_ATR5_MULT = 0.6
        self.S1_VOL_MULT = 1.2
        self.BREAKOUT_CLOSE_TICKS = 2
        self.S1_RETEST_TIMEOUT_MIN = 90

        self.LANE_SPEED_BARS_HALF = 6
        self.S2_REQUIRE_POLR_ALIGN = True

        # Session filter (match your defaults: allow LONDON + NYAM)
        self.ENABLE_SESSION_FILTER = True
        self.ALLOW_ASIA = False
        self.ALLOW_LONDON = True
        self.ALLOW_NYAM = True
        self.ALLOW_NYMID = False
        self.ALLOW_NYPM = False

        self.MAX_TRADES_PER_DAY = 1
        self.QTY = 1

        # ---- Futures subscription (MES) ----
        self.future = self.add_future(
            Futures.Indices.MicroSP500EMini,
            Resolution.MINUTE,
            extended_market_hours=True,
            data_mapping_mode=DataMappingMode.OPEN_INTEREST,
            data_normalization_mode=DataNormalizationMode.BACKWARDS_RATIO,
            contract_depth_offset=0
        )
        self.future.set_filter(0, 182)
        self.settings.seed_initial_prices = True  # lets you trade right after subscription

        # Tick size for MES is 0.25 index points
        # (QC SymbolProperties also contains this, but we keep it explicit)
        self.tick_size = 0.25

        # ---- State ----
        self.today = None
        self.trades_taken_today = 0

        # Rolling map state (rebuilt daily)
        self.map_built_for = None
        self.comp_min_tick = 0
        self.comp_max_tick = 0
        self.comp_poc_tick = 0
        self.vol_dense = []
        self.friction = []
        self.hvn_shelves = []  # list of (loTick, hiTick)
        self.lvn_lanes = []    # list of (loTick, hiTick)

        # PoLR state
        self.polr_bias = 0  # 0 neutral, +1 up, -1 down
        self.polr_r = 0
        self.score_up = 0.0
        self.score_dn = 0.0
        self.boundary_tick = 0
        self.boundary_type = 0  # 1 HVN_TOP,2 HVN_BOTTOM,3 LANE_TOP,4 LANE_BOTTOM

        # Plan state
        self.plan_active = False
        self.plan_style = 0  # 1 S1, 2 S2
        self.plan_bias = 0
        self.plan_created_time = None
        self.plan_boundary_tick = 0
        self.plan_entry_tick = 0
        self.plan_stop_tick = 0
        self.plan_tp1_tick = 0
        self.plan_submitted = False

        # Lane tracking for S2
        self.lane_tracking_idx = None
        self.lane_tracking_enter_bar = None
        self.lane_tracking_enter_side = 0

        # Bar counter (minute bars)
        self.bar_index = 0

        # ATR indicators (5m and 60m)
        self.atr5 = AverageTrueRange(14, MovingAverageType.SIMPLE)
        self.atr60 = AverageTrueRange(14, MovingAverageType.SIMPLE)

        self.consolidate(self.future.symbol, timedelta(minutes=5), self._on_5m)
        self.consolidate(self.future.symbol, timedelta(minutes=60), self._on_60m)

        # Volume median last 30m
        self.last30m_vols = deque(maxlen=30)

        # Order tickets (OCO bracket)
        self.entry_ticket = None
        self.tp_ticket = None
        self.sl_ticket = None
        self.pending_bracket = None  # dict with stop/tp ticks

    # ------------------------- Data handlers -------------------------

    def on_data(self, data: Slice):
        # Use the currently mapped contract for trading
        sym = self.future.mapped
        bar = data.bars.get(sym, None)
        if bar is None:
            # Sometimes only continuous updates; try continuous as fallback
            bar = data.bars.get(self.future.symbol, None)
            if bar is None:
                return
            sym = self.future.symbol

        self.bar_index += 1

        # Day boundary reset
        d = self.time.date()
        if self.today != d:
            self.today = d
            self.trades_taken_today = 0
            self._reset_plan()
            self._cancel_all_open_orders()

        # Track volume window for S1 participation
        self.last30m_vols.append(float(bar.volume))

        # Rebuild rolling map once per day (first bar of day)
        if self.map_built_for != d:
            built = self._rebuild_rolling7_map()
            self.map_built_for = d if built else None

        if self.polr_bias == 0 or self.boundary_tick == 0:
            return

        close_tick = self._px_to_tick(bar.close)
        high_tick = self._px_to_tick(bar.high)
        low_tick = self._px_to_tick(bar.low)

        # Try create plan (S1 / S2)
        self._try_create_plan(close_tick, high_tick, low_tick)

        # Execute plan (submit orders)
        self._execute_engine(close_tick, high_tick, low_tick, sym)

    def on_order_event(self, order_event: OrderEvent):
        # OCO bracket management
        if self.entry_ticket and order_event.order_id == self.entry_ticket.order_id:
            if order_event.status == OrderStatus.FILLED and self.pending_bracket:
                sym = self.entry_ticket.symbol
                side = self.pending_bracket["side"]
                stop_tick = self.pending_bracket["stop_tick"]
                tp_tick = self.pending_bracket["tp_tick"]

                qty = abs(self.entry_ticket.quantity)
                if side > 0:
                    self.sl_ticket = self.stop_market_order(sym, -qty, stop_tick * self.tick_size, "SL")
                    if tp_tick:
                        self.tp_ticket = self.limit_order(sym, -qty, tp_tick * self.tick_size, "TP")
                else:
                    self.sl_ticket = self.stop_market_order(sym, qty, stop_tick * self.tick_size, "SL")
                    if tp_tick:
                        self.tp_ticket = self.limit_order(sym, qty, tp_tick * self.tick_size, "TP")

                self.pending_bracket = None

        # Cancel sibling when TP or SL fills
        if self.tp_ticket and order_event.order_id == self.tp_ticket.order_id and order_event.status == OrderStatus.FILLED:
            if self.sl_ticket:
                self.transactions.cancel_order(self.sl_ticket.order_id, "OCO cancel SL")
            self._clear_tickets()

        if self.sl_ticket and order_event.order_id == self.sl_ticket.order_id and order_event.status == OrderStatus.FILLED:
            if self.tp_ticket:
                self.transactions.cancel_order(self.tp_ticket.order_id, "OCO cancel TP")
            self._clear_tickets()

    def _on_5m(self, bar: TradeBar):
        self.atr5.update(bar)

    def _on_60m(self, bar: TradeBar):
        self.atr60.update(bar)

    # ------------------------- Engine logic -------------------------

    def _try_create_plan(self, close_tick: int, high_tick: int, low_tick: int):
        # Don’t create new plans if we already have one active
        if self.plan_active:
            # Lane tracking reset if we leave lane
            return

        # Need indicators ready
        if not (self.atr5.is_ready and self.atr60.is_ready):
            return

        atr5_ticks = max(1, int(round(self.atr5.current.value / self.tick_size)))
        med_vol = self._median_last30m()
        if med_vol <= 0:
            return
        vol_ratio = (self.last30m_vols[-1] / med_vol) if self.last30m_vols else 0.0

        bar_range_ticks = max(0, high_tick - low_tick)

        # ---- Style 1 (breakout + displacement + participation) ----
        close_break = False
        if self.polr_bias > 0:
            close_break = close_tick >= (self.boundary_tick + self.BREAKOUT_CLOSE_TICKS)
        else:
            close_break = close_tick <= (self.boundary_tick - self.BREAKOUT_CLOSE_TICKS)

        displacement = float(bar_range_ticks) >= (self.S1_ATR5_MULT * float(atr5_ticks))
        participation = vol_ratio >= self.S1_VOL_MULT

        if close_break and displacement and participation:
            self.plan_active = True
            self.plan_style = 1
            self.plan_bias = self.polr_bias
            self.plan_boundary_tick = self.boundary_tick

            self.plan_entry_tick = (self.boundary_tick + self.FRONT_RUN) if self.polr_bias > 0 else (self.boundary_tick - self.FRONT_RUN)
            self.plan_stop_tick = (self.boundary_tick - self.PLAN_STOP_BUF) if self.polr_bias > 0 else (self.boundary_tick + self.PLAN_STOP_BUF)

            # TP1: next shelf beyond entry (simple)
            self.plan_tp1_tick = self._next_target_tick(self.plan_entry_tick, self.plan_bias)

            self.plan_created_time = self.time
            self.plan_submitted = False
            self.debug(f"S1 PLAN {self.today} bias={self.plan_bias} boundary={self.boundary_tick} entry={self.plan_entry_tick} stop={self.plan_stop_tick} tp={self.plan_tp1_tick}")
            return

        # ---- Style 2 (lane mid-cross fast) ----
        lane_idx = self._lane_index_containing(close_tick)
        if lane_idx is None:
            self.lane_tracking_idx = None
            return

        if self.lane_tracking_idx != lane_idx:
            self.lane_tracking_idx = lane_idx
            self.lane_tracking_enter_bar = self.bar_index
            lo, hi = self.lvn_lanes[lane_idx]
            # enter side: -1 from above, +1 from below
            prev_close = close_tick
            self.lane_tracking_enter_side = 0
            # we can only infer side using current bar location; good enough for this QC test
            if prev_close < lo:
                self.lane_tracking_enter_side = -1
            elif prev_close > hi:
                self.lane_tracking_enter_side = 1

        lo, hi = self.lvn_lanes[lane_idx]
        mid = (lo + hi) // 2
        bars_to_mid = self.bar_index - (self.lane_tracking_enter_bar or self.bar_index)

        trig = False
        if self.polr_bias > 0 and self.lane_tracking_enter_side == -1 and close_tick >= mid and bars_to_mid <= self.LANE_SPEED_BARS_HALF:
            trig = True
        if self.polr_bias < 0 and self.lane_tracking_enter_side == 1 and close_tick <= mid and bars_to_mid <= self.LANE_SPEED_BARS_HALF:
            trig = True
        if not trig:
            return

        self.plan_active = True
        self.plan_style = 2
        self.plan_bias = self.polr_bias
        self.plan_boundary_tick = self.boundary_tick
        self.plan_entry_tick = (mid + self.FRONT_RUN) if self.polr_bias > 0 else (mid - self.FRONT_RUN)
        self.plan_stop_tick = (lo - self.PLAN_STOP_BUF) if self.polr_bias > 0 else (hi + self.PLAN_STOP_BUF)
        self.plan_tp1_tick = self._next_target_tick(hi if self.polr_bias > 0 else lo, self.plan_bias)

        self.plan_created_time = self.time
        self.plan_submitted = False
        self.debug(f"S2 PLAN {self.today} bias={self.plan_bias} lane=[{lo},{hi}] mid={mid} entry={self.plan_entry_tick} stop={self.plan_stop_tick} tp={self.plan_tp1_tick}")

    def _execute_engine(self, close_tick: int, high_tick: int, low_tick: int, sym: Symbol):
        if self.trades_taken_today >= self.MAX_TRADES_PER_DAY:
            return
        if self.entry_ticket or self.portfolio.invested:
            return
        if not self.plan_active or self.plan_submitted:
            return
        if self.ENABLE_SESSION_FILTER and not self._session_allowed():
            return

        side = 1 if self.plan_bias > 0 else -1

        # S1: wait for retest of entry within timeout
        if self.plan_style == 1:
            mins_since = (self.time - self.plan_created_time).total_seconds() / 60.0
            if mins_since > self.S1_RETEST_TIMEOUT_MIN:
                self.debug("S1 TIMEOUT cancel plan")
                self._reset_plan()
                return

            # Must be at least 1 minute after creation (matches your Sierra minsSince>=1.0)
            if mins_since < 1.0:
                return

            retest = (low_tick <= self.plan_entry_tick <= high_tick)
            if not retest:
                return

            self._submit_bracket(sym, side, "LIMIT", self.plan_entry_tick, self.plan_stop_tick, self.plan_tp1_tick, tag="R7LVN_S1")
            self.plan_submitted = True
            self.trades_taken_today += 1
            return

        # S2: market if within 2 ticks else limit
        if self.plan_style == 2:
            if self.S2_REQUIRE_POLR_ALIGN:
                if side > 0 and self.polr_bias <= 0:
                    return
                if side < 0 and self.polr_bias >= 0:
                    return

            dist = abs(close_tick - self.plan_entry_tick)
            entry_type = "MARKET" if dist <= 2 else "LIMIT"
            self._submit_bracket(sym, side, entry_type, self.plan_entry_tick, self.plan_stop_tick, self.plan_tp1_tick, tag="R7LVN_S2")
            self.plan_submitted = True
            self.trades_taken_today += 1
            return

    # ------------------------- Order helpers -------------------------

    def _submit_bracket(self, sym: Symbol, side: int, entry_type: str, entry_tick: int, stop_tick: int, tp_tick: int, tag: str):
        qty = self.QTY if side > 0 else -self.QTY

        # Entry
        if entry_type == "LIMIT":
            self.entry_ticket = self.limit_order(sym, qty, entry_tick * self.tick_size, tag)
        else:
            self.entry_ticket = self.market_order(sym, qty, tag)

        # Store bracket to place on fill (OCO logic in OnOrderEvent)
        self.pending_bracket = {
            "side": side,
            "stop_tick": stop_tick,
            "tp_tick": tp_tick
        }
        self.debug(f"ORDER {tag} entryType={entry_type} side={side} entry={entry_tick} stop={stop_tick} tp={tp_tick}")

    def _clear_tickets(self):
        self.entry_ticket = None
        self.tp_ticket = None
        self.sl_ticket = None
        self.pending_bracket = None
        self._reset_plan()

    def _cancel_all_open_orders(self):
        for ticket in self.transactions.get_open_order_tickets():
            self.transactions.cancel_order(ticket.order_id, "Daily reset cancel")

    def _history_frame_for_symbol(self, history_df, prefer_contains="MES"):
        if history_df is None or history_df.empty:
            return None

        index = history_df.index
        if not isinstance(index, pd.MultiIndex):
            return history_df

        level_names = list(index.names)
        symbol_level = 0
        for i, name in enumerate(level_names):
            if name and "symbol" in str(name).lower():
                symbol_level = i
                break

        try:
            keys = list(index.get_level_values(symbol_level).unique())
        except Exception as ex:
            self.debug(f"History MultiIndex key extraction failed: {ex}")
            return None

        if not keys:
            self.debug("History MultiIndex had no symbol keys")
            return None

        selected_key = None
        prefer_upper = str(prefer_contains).upper()
        for key in keys:
            if prefer_upper in str(key).upper():
                selected_key = key
                break

        if selected_key is None:
            selected_key = keys[0]

        try:
            return history_df.xs(selected_key, level=symbol_level)
        except Exception as ex:
            keys_preview = [str(k) for k in keys[:5]]
            self.debug(
                f"History slice failed key={selected_key} level={symbol_level} "
                f"available_keys={keys_preview} err={ex}"
            )
            return None

    # ------------------------- Map rebuild (rolling 7) -------------------------

    def _rebuild_rolling7_map(self) -> bool:
        # Pull ~8 days of minute bars and build 7 completed days composite
        sym = self.future.symbol  # use continuous for stability across rolls
        history = self.history(sym, timedelta(days=self.ROLLING_DAYS + 1), Resolution.MINUTE)
        if history.empty:
            self.debug("No history returned for rebuild (data plan?)")
            return False

        is_multi = isinstance(history.index, pd.MultiIndex)
        keys_preview = []
        if is_multi:
            try:
                keys_preview = [str(k) for k in history.index.get_level_values(0).unique()[:5]]
            except Exception:
                keys_preview = []

        # Collect by date (completed days only)
        # History df index: (symbol, time) multiindex in QC
        df = self._history_frame_for_symbol(history, prefer_contains="MES")
        if df is None or df.empty:
            self.debug(
                f"History selection failed shape={history.shape} is_multi={is_multi} "
                f"keys={keys_preview}"
            )
            return False

        first_ts = df.index.min() if len(df.index) else None
        last_ts = df.index.max() if len(df.index) else None
        self.debug(
            f"History diagnostics shape={history.shape} is_multi={is_multi} "
            f"keys={keys_preview} slice_first={first_ts} slice_last={last_ts}"
        )

        today = self.time.date()
        by_day = {}
        for t, row in df.iterrows():
            d = t.date()
            if d >= today:
                continue
            close_px = float(row["close"])
            vol = float(row["volume"])
            tick = self._px_to_tick(close_px)
            by_day.setdefault(d, {})
            by_day[d][tick] = by_day[d].get(tick, 0.0) + vol

        days = sorted(by_day.keys())[-self.ROLLING_DAYS:]
        if len(days) < self.ROLLING_DAYS:
            self.debug(f"Not enough completed days for rolling map: have {len(days)}")
            return False

        # Composite histogram
        comp = {}
        for d in days:
            for tick, vol in by_day[d].items():
                comp[tick] = comp.get(tick, 0.0) + vol

        if not comp:
            return False

        self.comp_min_tick = min(comp.keys())
        self.comp_max_tick = max(comp.keys())
        n = self.comp_max_tick - self.comp_min_tick + 1
        self.vol_dense = [0.0] * n
        for i in range(n):
            self.vol_dense[i] = comp.get(self.comp_min_tick + i, 0.0)

        # Composite POC
        self.comp_poc_tick = self.comp_min_tick + max(range(n), key=lambda i: self.vol_dense[i])

        # Build friction + zones + PoLR
        self._build_friction_and_zones()
        self._compute_polr(current_px_tick=self._px_to_tick(self.securities[sym].price))

        self.debug(f"REBUILD {today} compRange=[{self.comp_min_tick},{self.comp_max_tick}] POC={self.comp_poc_tick} shelves={len(self.hvn_shelves)} lanes={len(self.lvn_lanes)} bias={self.polr_bias} boundary={self.boundary_tick}")
        return True

    def _build_friction_and_zones(self):
        n = len(self.vol_dense)
        if n <= 0:
            self.friction = []
            self.hvn_shelves = []
            self.lvn_lanes = []
            return

        vs = list(self.vol_dense)
        if self.SMOOTH_FRICTION_3TAP and n >= 3:
            sm = [0.0] * n
            for i in range(n):
                li = max(0, i - 1)
                ri = min(n - 1, i + 1)
                sm[i] = (vs[li] + vs[i] + vs[ri]) / 3.0
            vs = sm

        tmp = sorted(vs)
        denom = max(1, n - 1)
        self.friction = [0.0] * n
        for i in range(n):
            # rank as lower_bound / (n-1)
            r = self._lower_bound(tmp, vs[i])
            self.friction[i] = float(r) / float(denom)

        # shelves and lanes
        shelves = []
        lanes = []

        i = 0
        while i < n:
            if self.friction[i] >= self.HVN_Q:
                j = i
                while j + 1 < n and self.friction[j + 1] >= self.HVN_Q:
                    j += 1
                lo = self.comp_min_tick + i
                hi = self.comp_min_tick + j
                if (hi - lo + 1) >= self.MIN_SHELF_W:
                    shelves.append((lo, hi))
                i = j + 1
            else:
                i += 1

        i = 0
        while i < n:
            if self.friction[i] <= self.LVN_Q:
                j = i
                while j + 1 < n and self.friction[j + 1] <= self.LVN_Q:
                    j += 1
                lo = self.comp_min_tick + i
                hi = self.comp_min_tick + j
                if (hi - lo + 1) >= self.MIN_LANE_W:
                    lanes.append((lo, hi))
                i = j + 1
            else:
                i += 1

        self.hvn_shelves = self._merge_zones(shelves, self.MERGE_ZONE_GAP)
        self.lvn_lanes = self._merge_zones(lanes, self.MERGE_ZONE_GAP)

    def _compute_polr(self, current_px_tick: int):
        self.polr_bias = 0
        self.boundary_tick = 0
        self.boundary_type = 0

        if not self.friction:
            return

        atr60_ticks = max(1, int(round(self.atr60.current.value / self.tick_size))) if self.atr60.is_ready else 120
        raw_r = int(round(self.POLR_ATR60_MULT * float(atr60_ticks)))
        self.polr_r = min(self.POLR_R_MAX, max(self.POLR_R_MIN, raw_r))

        up_lo = current_px_tick
        up_hi = current_px_tick + self.polr_r
        dn_lo = current_px_tick - self.polr_r
        dn_hi = current_px_tick

        lane_up = 0
        lane_dn = 0
        cost_up = 0.0
        cost_dn = 0.0

        for t in range(up_lo, up_hi + 1):
            if t < self.comp_min_tick or t > self.comp_max_tick:
                continue
            f = self.friction[t - self.comp_min_tick]
            if f <= self.LVN_Q:
                lane_up += 1
            cost_up += f

        for t in range(dn_lo, dn_hi + 1):
            if t < self.comp_min_tick or t > self.comp_max_tick:
                continue
            f = self.friction[t - self.comp_min_tick]
            if f <= self.LVN_Q:
                lane_dn += 1
            cost_dn += f

        denom = float(max(1, self.polr_r))
        lane_up_frac = float(lane_up) / denom
        lane_dn_frac = float(lane_dn) / denom
        cost_up_avg = cost_up / denom
        cost_dn_avg = cost_dn / denom

        shelves_up = self._count_intersections(self.hvn_shelves, up_lo, up_hi)
        shelves_dn = self._count_intersections(self.hvn_shelves, dn_lo, dn_hi)

        self.score_up = self.POLR_LANE_W * lane_up_frac - self.POLR_COST_W * cost_up_avg - self.POLR_SHELF_PENALTY * float(shelves_up)
        self.score_dn = self.POLR_LANE_W * lane_dn_frac - self.POLR_COST_W * cost_dn_avg - self.POLR_SHELF_PENALTY * float(shelves_dn)

        if (self.score_up - self.score_dn) > self.POLR_BIAS_THRESH:
            self.polr_bias = 1
        elif (self.score_dn - self.score_up) > self.POLR_BIAS_THRESH:
            self.polr_bias = -1

        # boundary selection (nearest HVN or lane boundary in bias direction)
        if self.polr_bias > 0:
            hvn_cand = None
            lane_cand = None

            for lo, hi in self.hvn_shelves:
                if hi <= current_px_tick:
                    continue
                hvn_cand = hi if (hvn_cand is None or (hi - current_px_tick) < (hvn_cand - current_px_tick)) else hvn_cand

            for lo, hi in self.lvn_lanes:
                if hi <= current_px_tick:
                    continue
                lane_cand = hi if (lane_cand is None or (hi - current_px_tick) < (lane_cand - current_px_tick)) else lane_cand

            # pick nearest
            if hvn_cand is not None and (lane_cand is None or (hvn_cand - current_px_tick) <= (lane_cand - current_px_tick)):
                self.boundary_tick = hvn_cand
                self.boundary_type = 1
            elif lane_cand is not None:
                self.boundary_tick = lane_cand
                self.boundary_type = 3

        elif self.polr_bias < 0:
            hvn_cand = None
            lane_cand = None

            for lo, hi in self.hvn_shelves:
                if lo >= current_px_tick:
                    continue
                hvn_cand = lo if (hvn_cand is None or (current_px_tick - lo) < (current_px_tick - hvn_cand)) else hvn_cand

            for lo, hi in self.lvn_lanes:
                if lo >= current_px_tick:
                    continue
                lane_cand = lo if (lane_cand is None or (current_px_tick - lo) < (current_px_tick - lane_cand)) else lane_cand

            if hvn_cand is not None and (lane_cand is None or (current_px_tick - hvn_cand) <= (current_px_tick - lane_cand)):
                self.boundary_tick = hvn_cand
                self.boundary_type = 2
            elif lane_cand is not None:
                self.boundary_tick = lane_cand
                self.boundary_type = 4

    # ------------------------- Utility -------------------------

    def _px_to_tick(self, px: float) -> int:
        return int(round(px / self.tick_size))

    def _median_last30m(self) -> float:
        if len(self.last30m_vols) < 10:
            return 0.0
        return statistics.median(self.last30m_vols)

    def _session_allowed(self) -> bool:
        if not self.ENABLE_SESSION_FILTER:
            return True
        m = self.time.hour * 60 + self.time.minute
        # your session buckets
        if m >= 1080 or m <= 119:
            return self.ALLOW_ASIA
        if 120 <= m <= 509:
            return self.ALLOW_LONDON
        if 510 <= m <= 719:
            return self.ALLOW_NYAM
        if 720 <= m <= 839:
            return self.ALLOW_NYMID
        return self.ALLOW_NYPM

    def _reset_plan(self):
        self.plan_active = False
        self.plan_style = 0
        self.plan_bias = 0
        self.plan_created_time = None
        self.plan_boundary_tick = 0
        self.plan_entry_tick = 0
        self.plan_stop_tick = 0
        self.plan_tp1_tick = 0
        self.plan_submitted = False
        self.lane_tracking_idx = None
        self.lane_tracking_enter_bar = None
        self.lane_tracking_enter_side = 0

    def _lane_index_containing(self, px_tick: int):
        for i, (lo, hi) in enumerate(self.lvn_lanes):
            if lo <= px_tick <= hi:
                return i
        return None

    def _next_target_tick(self, from_tick: int, bias: int) -> int:
        # Simple TP: next HVN shelf boundary in direction of bias
        if bias > 0:
            candidates = [lo for (lo, hi) in self.hvn_shelves if lo > from_tick]
            return (min(candidates) - self.FRONT_RUN) if candidates else 0
        else:
            candidates = [hi for (lo, hi) in self.hvn_shelves if hi < from_tick]
            return (max(candidates) + self.FRONT_RUN) if candidates else 0

    def _merge_zones(self, zones, gap_ticks: int):
        if not zones:
            return []
        zones = sorted(zones, key=lambda z: (z[0], z[1]))
        merged = [zones[0]]
        for lo, hi in zones[1:]:
            last_lo, last_hi = merged[-1]
            if lo <= (last_hi + gap_ticks + 1):
                merged[-1] = (last_lo, max(last_hi, hi))
            else:
                merged.append((lo, hi))
        return merged

    def _count_intersections(self, zones, lo, hi) -> int:
        c = 0
        for zlo, zhi in zones:
            if zhi >= lo and zlo <= hi:
                c += 1
        return c

    def _lower_bound(self, arr, x):
        # first idx where arr[idx] >= x
        lo, hi = 0, len(arr)
        while lo < hi:
            mid = (lo + hi) // 2
            if arr[mid] < x:
                lo = mid + 1
            else:
                hi = mid
        return lo
