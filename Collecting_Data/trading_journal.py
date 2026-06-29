import os
import uuid
import logging
import threading
import pandas as pd
from datetime import datetime, timezone

logger = logging.getLogger("TradingJournal")

class TradingJournal:
    def __init__(
        self,
        journal_root: str,
        mode: str = "live",
    ):
        self.journal_root = journal_root
        self.mode = mode
        self._locks = {}
        self._locks_lock = threading.Lock()
        self._signal_cache = {}
        self._cache_lock = threading.Lock()

        try:
            if not os.path.exists(self.journal_root):
                os.makedirs(self.journal_root)

            if self.mode == "training":
                os.makedirs(os.path.join(self.journal_root, "training", "signals"), exist_ok=True)
                os.makedirs(os.path.join(self.journal_root, "training", "outcomes"), exist_ok=True)
            else:
                os.makedirs(os.path.join(self.journal_root, self.mode), exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create journal directories: {e}")

    def _get_lock(self, filepath: str) -> threading.Lock:
        with self._locks_lock:
            if filepath not in self._locks:
                self._locks[filepath] = threading.Lock()
            return self._locks[filepath]

    def _get_filepath(self, strategy: str, symbol: str, timeframe: str, event_type: str) -> str:
        if self.mode == "training":
            if event_type == "outcome":
                return os.path.join(
                    self.journal_root, "training", "outcomes",
                    f"{strategy}_{symbol}_{timeframe}_outcomes.csv"
                )
            else:
                return os.path.join(
                    self.journal_root, "training", "signals",
                    f"{strategy}_{symbol}_{timeframe}_signals.csv"
                )
        else:
            return os.path.join(
                self.journal_root, self.mode,
                f"{strategy}_{symbol}_{timeframe}_full.csv"
            )

    def _write_row(self, filepath: str, row_data: dict):
        lock = self._get_lock(filepath)
        try:
            with lock:
                file_exists = os.path.exists(filepath)
                if not file_exists:
                    # NEW FILE case
                    df = pd.DataFrame([row_data])
                    base_cols = ["event_id", "signal_id", "event_type", "system_timestamp", "bar_timestamp", "strategy", "symbol", "timeframe", "signal_type", "direction"]
                    other_cols = [c for c in df.columns if c not in base_cols]
                    df = df[base_cols + other_cols]

                    tmp_path = f"{filepath}.{uuid.uuid4()}.tmp"
                    df.to_csv(tmp_path, index=False)
                    os.replace(tmp_path, filepath)
                    logger.info(f"Logged {row_data.get('event_type')} for {row_data.get('signal_id')} in {filepath} (new file)")
                    return

                # Exists, read header
                with open(filepath, 'r') as f:
                    header = f.readline().strip().split(',')

            # Check for new columns
            new_cols = [k for k in row_data.keys() if k not in header]

            if new_cols:
                logger.warning(f"New columns detected: {new_cols}. Triggering rewrite for {filepath}")
                with lock:
                    try:
                        df = pd.read_csv(filepath)
                    except Exception as e:
                        logger.error(f"Failed to read CSV for rewrite {filepath}: {e}")
                        return

                new_row_df = pd.DataFrame([row_data])
                df = pd.concat([df, new_row_df], ignore_index=True)
                df = df.fillna("")

                tmp_path = f"{filepath}.{uuid.uuid4()}.tmp"
                df.to_csv(tmp_path, index=False)
                with lock:
                    os.replace(tmp_path, filepath)
            else:
                # Simple append
                df_row = pd.DataFrame([row_data])
                df_row = df_row.reindex(columns=header).fillna("")
                with lock:
                    df_row.to_csv(filepath, mode='a', header=False, index=False)

            logger.info(f"Logged {row_data.get('event_type')} for {row_data.get('signal_id')} in {filepath}")
        except Exception as e:
            logger.error(f"Failed to write to journal {filepath}: {e}")

    def _get_base_data(self, event_type: str, signal_id: str, bar_timestamp: str, strategy: str, symbol: str, timeframe: str, signal_type: str, direction: int):
        return {
            "event_id": str(uuid.uuid4()),
            "signal_id": signal_id,
            "event_type": event_type,
            "system_timestamp": datetime.now(timezone.utc).isoformat(),
            "bar_timestamp": bar_timestamp,
            "strategy": strategy,
            "symbol": symbol,
            "timeframe": timeframe,
            "signal_type": signal_type,
            "direction": direction,
        }

    def _cache_signal(self, signal_id, data):
        with self._cache_lock:
            self._signal_cache[signal_id] = {
                "strategy": data["strategy"],
                "symbol": data["symbol"],
                "timeframe": data["timeframe"],
                "signal_type": data["signal_type"],
                "direction": data["direction"],
                "bar_timestamp": data["bar_timestamp"]
            }

    def _get_signal_context(self, signal_id):
        with self._cache_lock:
            return self._signal_cache.get(signal_id)

    def log_signal(
        self,
        signal_type: str,
        symbol: str,
        timeframe: str,
        direction: int,
        entry_price: float,
        sl_price: float,
        tp_level: int,
        stage: str,
        strategy: str,
        signal_category: str,
        bar_timestamp: str,
        extra_fields: dict = None,
    ) -> str:
        signal_id = str(uuid.uuid4())
        data = self._get_base_data("signal", signal_id, bar_timestamp, strategy, symbol, timeframe, signal_type, direction)
        data.update({
            "entry_price": entry_price,
            "sl_price": sl_price,
            "tp_level": tp_level,
            "stage": stage,
            "signal_category": signal_category,
            "risk_pct_default": "",
        })
        if extra_fields:
            data.update(extra_fields)

        self._cache_signal(signal_id, data)

        filepath = self._get_filepath(strategy, symbol, timeframe, "signal")
        self._write_row(filepath, data)
        return signal_id

    def log_order_open(
        self,
        signal_id: str,
        ticket: int,
        actual_entry: float,
        actual_sl: float,
        actual_tp: float,
        lot_size: float,
        risk_pct: float,
        extra_fields: dict = None,
    ) -> None:
        ctx = self._get_signal_context(signal_id)
        if not ctx:
            logger.error(f"Signal context not found for {signal_id}")
            return

        data = self._get_base_data("order_open", signal_id, ctx["bar_timestamp"], ctx["strategy"], ctx["symbol"], ctx["timeframe"], ctx["signal_type"], ctx["direction"])
        data.update({
            "ticket": ticket,
            "actual_entry": actual_entry,
            "actual_sl": actual_sl,
            "actual_tp": actual_tp,
            "lot_size": lot_size,
            "risk_pct": risk_pct,
        })
        if extra_fields:
            data.update(extra_fields)

        filepath = self._get_filepath(ctx["strategy"], ctx["symbol"], ctx["timeframe"], "order_open")
        self._write_row(filepath, data)

    def log_order_failure(
        self,
        signal_id: str,
        reason: str,
        extra_fields: dict = None,
    ) -> None:
        ctx = self._get_signal_context(signal_id)
        if not ctx:
            logger.error(f"Signal context not found for {signal_id}")
            return

        data = self._get_base_data("order_failure", signal_id, ctx["bar_timestamp"], ctx["strategy"], ctx["symbol"], ctx["timeframe"], ctx["signal_type"], ctx["direction"])
        data.update({
            "reason": reason,
        })
        if extra_fields:
            data.update(extra_fields)

        filepath = self._get_filepath(ctx["strategy"], ctx["symbol"], ctx["timeframe"], "order_failure")
        self._write_row(filepath, data)

    def log_partial_close(
        self,
        signal_id: str,
        ticket: int,
        stage_reached: int,
        closed_volume: float,
        close_price: float,
        new_sl: float,
        extra_fields: dict = None,
    ) -> None:
        ctx = self._get_signal_context(signal_id)
        if not ctx:
            logger.error(f"Signal context not found for {signal_id}")
            return

        data = self._get_base_data("partial_close", signal_id, ctx["bar_timestamp"], ctx["strategy"], ctx["symbol"], ctx["timeframe"], ctx["signal_type"], ctx["direction"])
        data.update({
            "ticket": ticket,
            "stage_reached": stage_reached,
            "closed_volume": closed_volume,
            "close_price": close_price,
            "new_sl": new_sl,
        })
        if extra_fields:
            data.update(extra_fields)

        filepath = self._get_filepath(ctx["strategy"], ctx["symbol"], ctx["timeframe"], "partial_close")
        self._write_row(filepath, data)

    def log_outcome(
        self,
        signal_id: str,
        ticket: int,
        outcome: str,
        close_price: float,
        pnl_dollars: float,
        duration_seconds: int,
        extra_fields: dict = None,
    ) -> None:
        ctx = self._get_signal_context(signal_id)
        if not ctx:
            logger.error(f"Signal context not found for {signal_id}")
            return

        data = self._get_base_data("outcome", signal_id, ctx["bar_timestamp"], ctx["strategy"], ctx["symbol"], ctx["timeframe"], ctx["signal_type"], ctx["direction"])
        data.update({
            "ticket": ticket,
            "outcome": outcome,
            "close_price": close_price,
            "pnl_dollars": pnl_dollars,
            "duration_seconds": duration_seconds,
        })
        if extra_fields:
            data.update(extra_fields)

        filepath = self._get_filepath(ctx["strategy"], ctx["symbol"], ctx["timeframe"], "outcome")
        self._write_row(filepath, data)

    def add_fields(
        self,
        signal_id: str,
        extra_fields: dict,
    ) -> None:
        ctx = self._get_signal_context(signal_id)
        if not ctx:
            logger.error(f"Signal context not found for {signal_id}")
            return

        data = self._get_base_data("enrichment", signal_id, ctx["bar_timestamp"], ctx["strategy"], ctx["symbol"], ctx["timeframe"], ctx["signal_type"], ctx["direction"])
        if extra_fields:
            data.update(extra_fields)

        filepath = self._get_filepath(ctx["strategy"], ctx["symbol"], ctx["timeframe"], "enrichment")
        self._write_row(filepath, data)

if __name__ == "__main__":
    import shutil

    test_root = "test_journal"
    if os.path.exists(test_root):
        shutil.rmtree(test_root)

    journal = TradingJournal(test_root, mode="live")

    # 1. Basic signal log
    sid = journal.log_signal(
        signal_type="standard", symbol="EURUSD", timeframe="M5", direction=1,
        entry_price=1.1000, sl_price=1.0950, tp_level=2, stage="multi",
        strategy="mm", signal_category="standard", bar_timestamp="2023-10-27T10:00:00Z"
    )
    print(f"Signal ID: {sid}")

    # 2. log_order_open
    journal.log_order_open(sid, 12345, 1.1001, 1.0950, 1.1100, 0.1, 1.0)

    # 3. log_outcome
    journal.log_outcome(sid, 12345, "tp2", 1.1100, 100.0, 3600)

    # 4. extra_fields
    sid2 = journal.log_signal(
        signal_type="reversal", symbol="GBPUSD", timeframe="M5", direction=-1,
        entry_price=1.2500, sl_price=1.2550, tp_level=1, stage="single",
        strategy="unity", signal_category="reversal", bar_timestamp="2023-10-27T11:00:00Z",
        extra_fields={"news_sentiment": 0.8, "session": "london"}
    )

    # 5. add_fields enrichment
    journal.add_fields(sid, {"confidence_score": 0.91})

    # 6. New column addition
    sid3 = journal.log_signal(
        signal_type="standard", symbol="EURUSD", timeframe="M5", direction=1,
        entry_price=1.1010, sl_price=1.0960, tp_level=1, stage="single",
        strategy="mm", signal_category="standard", bar_timestamp="2023-10-27T10:05:00Z"
    )
    journal.log_order_open(sid3, 12346, 1.1011, 1.0960, 1.1060, 0.1, 1.0, extra_fields={"new_col": "val"})

    # 7 & 8 Routing test
    t_journal = TradingJournal(test_root, mode="training")
    t_sid = t_journal.log_signal(
        signal_type="standard", symbol="EURUSD", timeframe="M5", direction=1,
        entry_price=1.1000, sl_price=1.0950, tp_level=2, stage="multi",
        strategy="mm", signal_category="standard", bar_timestamp="2023-10-27T10:00:00Z"
    )
    t_journal.log_outcome(t_sid, 12347, "sl", 1.0945, -50.0, 1800)

    # 9. Thread safety smoke test
    def worker(i):
        journal.log_signal(
            signal_type="standard", symbol="THREAD", timeframe="M5", direction=1,
            entry_price=1.1000, sl_price=1.0950, tp_level=1, stage="single",
            strategy="test", signal_category="standard", bar_timestamp=f"2023-10-27T12:00:{i:02d}Z"
        )

    threads = []
    for i in range(10):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print("Tests completed. Check 'test_journal' directory.")
