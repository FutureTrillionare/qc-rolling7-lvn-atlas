from AlgorithmImports import *
from collections import deque
from datetime import timedelta
import statistics
import pandas as pd


class Rolling7LVNAtlasMES_QC(QCAlgorithm):

    def initialize(self):
        # ---- Backtest window: April 2024 ----
        self.set_start_date(2024, 4, 1)
        self.set_end_date(2024, 5, 1)
        self.set_cash(100000)

        # Chicago timezone (your session logic)
        self.set_time_zone(TimeZones.CHICAGO)

        # ---- Strategy parameters (match Sierra defaults) ----
        self.ROLLING_DAYS = 7

        # Trade-day rollover: shift timestamps so Sun 17:00 CT -> Monday trade-date
        # (5pm CT + 7h = midnight next day)
        self.TRADE_DAY_ROLLOVER_HOURS = 7
        self.MIN_DAY_COVERAGE_PCT = 0.50  # similar to Sierra (used only to filter very partial days)

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

        # Session filter (allow LONDON + NYAM)
        self.ENABLE_SESSION_FILTER = True
        self.ALLOW_ASIA = False
        self.ALLOW_LONDON = True
        self.ALLOW_NYAM = True
        self.ALLOW_NYMID = False
        self.ALLOW_NYPM = False

        self.MAX_TRADES_PER_DAY = 1
        self.QTY = 1

        # ---- Futures subscription (MES continuous) ----
        self.future = self.add_future(
            Futures.Indices.MICRO_SP_500_E_MINI,
            Resolution.MINUTE,
            extended_market_hours=True,
            data_mapping_mode=DataMappingMode.OPEN_INTEREST,
            data_normalization_mode=DataNormalizationMode.BACKWARDS_RATIO,
            contract_depth_offset=0
        )
        self.future.set_filter(0, 182)
        self.settings.seed_initial_prices = True

        self.tick_size = 0.25

        # ---- State ----
        self.today_trade_date = None
        self.trades_taken_today = 0

        # Rolling map state (rebuilt once per trade-day)
        self.map_attempted_for = None
        self.map_built_for = None
        self.comp_min_tick = 0
        self.comp_max_tick = 0
        self.comp_poc_tick = 0
        self.vol_dense = []
        self.friction = []
        self.hvn_shelves = []
        self.lvn_lanes = []

        # PoLR state
        self.polr_bias = 0  # 0 neutral, +1 up, -1 down
        self.polr_r = 0
        self.score_up = 0.0
        self.score_dn = 0.0
        self.boundary_tick = 0
        self.boundary_type = 0

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

        self.bar_index = 0

        # ATR indicators (5m and 60m)
        self.atr5 = AverageTrueRange(14, MovingAverageType.SIMPLE)
        self.atr60 = AverageTrueRange(14, MovingAverageType.SIMPLE)

        self.consolidate(self.future.symbol, timedelta(minutes=5), self._on_5m)
        self.consolidate(self.future.symbol, timedelta(minutes=60), self._on_60m)

        # Volume median last 30m
        self.last30m_vols = deque(maxlen=30)

        # Order tickets (bracket on fill)
        self.entry_ticket = None
        self.tp_ticket = None
        self.sl_ticket = None
        self.pending_bracket = None

    # ------------------------- Time helpers -------------------------

    def _trade_date(self, ts) -> object:
        # ts can be QC datetime, python datetime, or pandas Timestamp
        try:
            if isinstance(ts, pd.Timestamp):
                ts = ts.to_pydatetime()
        except Exception:
            pass
        return (ts + timedelta(hours=self.TRADE_DAY_ROLLOVER_HOURS)).date()

    # ------------------------- Data handlers -------------------------

    def on_data(self, data: Slice):
        sym = self.future.mapped
        if sym is None:
            return

        bar = data.bars.get(sym, None)
        if bar is None:
            bar = data.bars.get(self.future.symbol, None)
            if bar is None:
                return

        self.bar_index += 1

        # Trade-day reset (Sunday night joins Monday)
        td = self._trade_date(self.time)
        if self.today_trade_date != td:
            self.today_trade_date = td
            self.trades_taken_today = 0
            self._reset_plan()
            self._cancel_all_open_orders()

        # Track volume window for S1 participation
        self.last30m_vols.append(float(bar.volume))

        # Rebuild rolling map once per trade-day (attempt only once to prevent spam)
        if self.map_attempted_for != td:
            self.map_attempted_for = td
            built = self._rebuild_rolling7_map()
            self.map_built_for = td if built else None

        if self.polr_bias == 0 or self.boundary_tick == 0:
            return

        close_tick = self._px_to_tick(bar.close)
        high_tick = self._px_to_tick(bar.high)
        low_tick = self._px_to_tick(bar.low)

        self._try_create_plan(close_tick, high_tick, low_tick)
        self._execute_engine(close_tick, high_tick, low_tick, sym)

    def on_order_event(self, order_event: OrderEvent):
        if self.entry_ticket and order_event.order_id == self.entry_ticket.order_id:
            if order_event.status == OrderStatus.FILLED and self.pending_bracket:
                sym = self.entry_ticket.symbol
                side = self.pending_bracket["side"]
                stop_tick = self.pending_bracket["stop_tick"]
                tp_tick = self.pending_bracket["tp_tick"]
                qty = abs(self.entry_ticket.quantity)

                if side > 0:
                    self.sl_ticket = self.stop_market_order(sym, -qty, stop_tick * self.tick_size)
                    if tp_tick:
                        self.tp_ticket = self.limit_order(sym, -qty, tp_tick * self.tick_size)
                else:
                    self.sl_ticket = self.stop_market_order(sym, qty, stop_tick * self.tick_size)
                    if tp_tick:
                        self.tp_ticket = self.limit_order(sym, qty, tp_tick * self.tick_size)

                self.pending_bracket = None

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
        if self.plan_active:
            return

        if not (self.atr5.is_ready and self.atr60.is_ready):
            return

        atr5_ticks = max(1, int(round(self.atr5.current.value / self.tick_size)))
        med_vol = self._median_last30m()
        if med_vol <= 0:
            return
        vol_ratio = (self.last30m_vols[-1] / med_vol) if self.last30m_vols else 0.0

        bar_range_ticks = max(0, high_tick - low_tick)

        # ---- Style 1 ----
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
            self.plan_tp1_tick = self._next_target_tick(self.plan_entry_tick, self.plan_bias)

            self.plan_created_time = self.time
            self.plan_submitted = False
            self.debug(f"S1 PLAN td={self.today_trade_date} bias={self.plan_bias} boundary={self.boundary_tick} entry={self.plan_entry_tick} stop={self.plan_stop_tick} tp={self.plan_tp1_tick}")
            return

        # ---- Style 2 ----
        lane_idx = self._lane_index_containing(close_tick)
        if lane_idx is None:
            self.lane_tracking_idx = None
            return

        if self.lane_tracking_idx != lane_idx:
            self.lane_tracking_idx = lane_idx
            self.lane_tracking_enter_bar = self.bar_index
            lo, hi = self.lvn_lanes[lane_idx]
            self.lane_tracking_enter_side = 0
            if close_tick < lo:
                self.lane_tracking_enter_side = -1
            elif close_tick > hi:
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
        self.debug(f"S2 PLAN td={self.today_trade_date} bias={self.plan_bias} lane=[{lo},{hi}] mid={mid} entry={self.plan_entry_tick} stop={self.plan_stop_tick} tp={self.plan_tp1_tick}")

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

        if self.plan_style == 1:
            mins_since = (self.time - self.plan_created_time).total_seconds() / 60.0
            if mins_since > self.S1_RETEST_TIMEOUT_MIN:
                self.debug("S1 TIMEOUT cancel plan")
                self._reset_plan()
                return
            if mins_since < 1.0:
                return

            retest = (low_tick <= self.plan_entry_tick <= high_tick)
            if not retest:
                return

            self._submit_bracket(sym, side, "LIMIT", self.plan_entry_tick, self.plan_stop_tick, self.plan_tp1_tick, tag="R7LVN_S1")
            self.plan_submitted = True
            self.trades_taken_today += 1
            return

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
        trade_sym = self.future.mapped if self.future.mapped else sym
        qty = self.QTY if side > 0 else -self.QTY

        if entry_type == "LIMIT":
            self.entry_ticket = self.limit_order(trade_sym, qty, entry_tick * self.tick_size)
        else:
            self.entry_ticket = self.market_order(trade_sym, qty)

        self.pending_bracket = {"side": side, "stop_tick": stop_tick, "tp_tick": tp_tick, "tag": tag}
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

    # ------------------------- Map rebuild (rolling 7) -------------------------

    def _rebuild_rolling7_map(self) -> bool:
        # extra buffer to survive weekends/partials
        lookback_days = self.ROLLING_DAYS + 7

        sym = self.future.symbol
        history = self.history(sym, timedelta(days=lookback_days), Resolution.MINUTE)
        if history is None or history.empty:
            self.debug("No history returned for rebuild")
            return False

        df = history

        # Slice MultiIndex by symbol if present
        if isinstance(df.index, pd.MultiIndex):
            names = [str(n).lower() if n else "" for n in df.index.names]
            level = None
            for i, n in enumerate(names):
                if "symbol" in n:
                    level = i
                    break
            if level is None:
                level = 0

            try:
                df = df.xs(sym, level=level)
            except Exception:
                # fallback: take first key at that level
                try:
                    first_key = df.index.get_level_values(level).unique()[0]
                    df = df.xs(first_key, level=level)
                except Exception as ex:
                    self.debug(f"History xs slice failed: {ex}")
                    return False

        # Force DatetimeIndex
        if isinstance(df.index, pd.MultiIndex):
            try:
                df = df.copy()
                df.index = pd.to_datetime(df.index.get_level_values(-1))
            except Exception as ex:
                self.debug(f"History index normalize failed: {ex}")
                return False

        if not isinstance(df.index, (pd.DatetimeIndex,)):
            try:
                df = df.copy()
                df.index = pd.to_datetime(df.index)
            except Exception as ex:
                self.debug(f"History datetime conversion failed: {ex}")
                return False

        today_td = self._trade_date(self.time)

        by_day = {}
        coverage = {}

        for ts, row in df.iterrows():
            td = self._trade_date(ts)
            if td >= today_td:
                continue

            coverage[td] = coverage.get(td, 0) + 1

            close_px = float(row["close"])
            vol = float(row["volume"])
            if vol <= 0:
                continue

            tick = self._px_to_tick(close_px)
            by_day.setdefault(td, {})
            by_day[td][tick] = by_day[td].get(tick, 0.0) + vol

        min_cov = int(1440 * self.MIN_DAY_COVERAGE_PCT)
        completed_days = [d for d in sorted(by_day.keys()) if coverage.get(d, 0) >= min_cov]

        days = completed_days[-self.ROLLING_DAYS:]
        if len(days) < self.ROLLING_DAYS:
            self.debug(f"Not enough completed trade-days for rolling map: have {len(days)} need {self.ROLLING_DAYS}")
            return False

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

        self.comp_poc_tick = self.comp_min_tick + max(range(n), key=lambda i: self.vol_dense[i])

        self._build_friction_and_zones()

        # current price tick for PoLR
        px = None
        if self.future.mapped and self.securities.contains_key(self.future.mapped):
            px = self.securities[self.future.mapped].price
        if px is None or px == 0:
            px = self.securities[sym].price

        self._compute_polr(current_px_tick=self._px_to_tick(px))

        self.debug(f"REBUILD td={today_td} used_days=[{days[0]}..{days[-1]}] comp=[{self.comp_min_tick},{self.comp_max_tick}] POC={self.comp_poc_tick} shelves={len(self.hvn_shelves)} lanes={len(self.lvn_lanes)} bias={self.polr_bias} boundary={self.boundary_tick}")
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
            r = self._lower_bound(tmp, vs[i])
            self.friction[i] = float(r) / float(denom)

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
        lo, hi = 0, len(arr)
        while lo < hi:
            mid = (lo + hi) // 2
            if arr[mid] < x:
                lo = mid + 1
            else:
                hi = mid
        return lo
