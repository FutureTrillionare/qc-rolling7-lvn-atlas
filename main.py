from AlgorithmImports import *
import pandas as pd


class Rolling7LvnAtlasAlgorithm(QCAlgorithm):
    def Initialize(self) -> None:
        self.SetStartDate(2024, 1, 1)
        self.SetCash(100000)

        # MES canonical future: do not hardcode '/MES' strings.
        self._mes_future = self.AddFuture(
            Futures.Indices.MICRO_SP_500_E_MINI,
            Resolution.Minute,
            dataNormalizationMode=DataNormalizationMode.BackwardsRatio,
            dataMappingMode=DataMappingMode.OpenInterest,
            contractDepthOffset=0,
        )

        self._rolling7_map = {}
        self._lookback_bars = 7 * 24 * 60

    def _select_history_for_symbol(self, history_df: pd.DataFrame, symbol: Symbol) -> pd.DataFrame:
        """Select symbol rows from LEAN History() output safely across index variants."""
        if history_df is None or history_df.empty:
            return pd.DataFrame()

        if not isinstance(history_df.index, pd.MultiIndex):
            # Some calls return a flat DatetimeIndex; return as-is for caller handling.
            return history_df

        selectors = [symbol]
        if hasattr(symbol, "Value"):
            selectors.append(symbol.Value)
        selectors.append(str(symbol))
        if hasattr(symbol, "ID"):
            selectors.append(symbol.ID)

        for selector in selectors:
            try:
                return history_df.xs(selector, level=0, drop_level=False)
            except Exception:
                continue

        keys_preview = []
        try:
            keys_preview = list(history_df.index.get_level_values(0).unique()[:20])
        except Exception:
            pass

        self.Debug(
            f"History symbol selection failed for {symbol}. "
            f"Available level-0 keys (first 20): {keys_preview}"
        )
        return pd.DataFrame()

    def _rebuild_rolling7_map(self) -> bool:
        sym = self._mes_future.Symbol
        history = self.History(sym, self._lookback_bars, Resolution.Minute)

        if history is None or history.empty:
            self.Debug(f"Rolling7 rebuild skipped for {sym}: empty history")
            return False

        df = self._select_history_for_symbol(history, sym)
        if df.empty:
            self.Debug(f"Rolling7 rebuild skipped for {sym}: no symbol rows in history")
            return False

        first_ts = df.index.min() if len(df.index) > 0 else None
        last_ts = df.index.max() if len(df.index) > 0 else None
        self.Debug(
            f"Rolling7 history request symbol={sym}, shape={df.shape}, "
            f"is_multiindex={isinstance(df.index, pd.MultiIndex)}, "
            f"first_ts={first_ts}, last_ts={last_ts}"
        )

        self._rolling7_map[sym] = df
        return True

    def OnData(self, data: Slice) -> None:
        if not self._rolling7_map:
            self._rebuild_rolling7_map()

        mapped_contract = self._mes_future.Mapped
        if mapped_contract is None:
            return

        # Orders must target mapped contract, not the canonical continuous symbol.
        if not self.Portfolio.Invested and data.Bars.ContainsKey(mapped_contract):
            self.MarketOrder(mapped_contract, 1)
