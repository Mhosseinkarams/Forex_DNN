import os
import json
import uuid
import logging
import threading
import pandas as pd
from datetime import datetime, timezone
from Collecting_Data.position_lifecycle import PositionLifecycle

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
                os.makedirs(os.path.join(self.journal_root, "training", "positions"), exist_ok=True)
            else:
                os.makedirs(os.path.join(self.journal_root, self.mode, "events"), exist_ok=True)
                os.makedirs(os.path.join(self.journal_root, self.mode, "positions"), exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create journal directories: {e}")

    def _get_lock(self, filepath: str) -> threading.Lock:
        with self._locks_lock:
            if filepath not in self._locks:
                self._locks[filepath] = threading.Lock()
            return self._locks[filepath]

    def _get_filepath(self, strategy: str, symbol: str, timeframe: str, event_type: str) -> str:
        if event_type == "lifecycle":
            if self.mode == "training":
                return os.path.join(
                    self.journal_root, "training", "positions",
                    f"{strategy}_{symbol}_{timeframe}_positions.csv"
                )
            else:
                return os.path.join(
                    self.journal_root, self.mode, "positions",
                    f"{strategy}_{symbol}_{timeframe}_positions.csv"
                )

        if self.mode == "training":
            if event_type == "outcome" or event_type == "position_closed":
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
            # Layer 1: Event Journal
            return os.path.join(
                self.journal_root, self.mode, "events",
                f"{strategy}_{symbol}_{timeframe}_events.csv"
            )

    def _write_row(self, filepath: str, row_data: dict):
        lock = self._get_lock(filepath)
        try:
            with lock:
                file_exists = os.path.exists(filepath)
                if not file_exists:
                    # NEW FILE case
                    df = pd.DataFrame([row_data])

                    # Layer 1 events have these base cols. Layer 2 positions might not have all of them prefixed correctly.
                    # PositionLifecycle to_csv_row uses signal_ prefix for most context fields.
                    base_cols = ["event_id", "signal_id", "event_type", "system_timestamp", "bar_timestamp", "strategy", "symbol", "timeframe", "signal_type", "direction"]
                    # If this is a Layer 2 summary, it might not have these exact names.
                    present_base = [c for c in base_cols if c in df.columns]
                    other_cols = [c for c in df.columns if c not in present_base]
                    df = df[present_base + other_cols]
                    
                    tmp_path = f"{filepath}.{uuid.uuid4()}.tmp"
                    df.to_csv(tmp_path, index=False)
                    os.replace(tmp_path, filepath)

                    ev_type = row_data.get('event_type', 'summary')
                    sig_id = row_data.get('signal_id', row_data.get('signal_signal_id', 'unknown'))
                    logger.info(f"Logged {ev_type} for {sig_id} in {filepath} (new file)")
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
            
            ev_type = row_data.get('event_type', 'summary')
            sig_id = row_data.get('signal_id', row_data.get('signal_signal_id', 'unknown'))
            logger.info(f"Logged {ev_type} for {sig_id} in {filepath}")
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

    def log_event(
        self,
        signal_id: str,
        event_type: str,
        extra_fields: dict = None,
    ) -> None:
        """Generic event logger for Layer 1 Event Journal."""
        ctx = self._get_signal_context(signal_id)
        if not ctx:
            logger.error(f"Signal context not found for {signal_id}")
            return

        data = self._get_base_data(event_type, signal_id, ctx["bar_timestamp"], ctx["strategy"], ctx["symbol"], ctx["timeframe"], ctx["signal_type"], ctx["direction"])
        if extra_fields:
            data.update(extra_fields)
        
        filepath = self._get_filepath(ctx["strategy"], ctx["symbol"], ctx["timeframe"], event_type)
        self._write_row(filepath, data)

    def log_sl_modified(self, signal_id: str, ticket: int, new_sl: float, reason: str = "") -> None:
        self.log_event(signal_id, "sl_modified", {"ticket": ticket, "new_sl": new_sl, "reason": reason})

    def log_tp_modified(self, signal_id: str, ticket: int, new_tp: float, reason: str = "") -> None:
        self.log_event(signal_id, "tp_modified", {"ticket": ticket, "new_tp": new_tp, "reason": reason})

    def log_breakeven(self, signal_id: str, ticket: int, price: float) -> None:
        self.log_event(signal_id, "breakeven", {"ticket": ticket, "price": price})

    def log_trailing_start(self, signal_id: str, ticket: int, distance: float) -> None:
        self.log_event(signal_id, "trailing_start", {"ticket": ticket, "distance": distance})

    def log_position_closed(
        self,
        signal_id: str,
        ticket: int,
        exit_price: float,
        reason: str,
        extra_fields: dict = None,
    ) -> None:
        """Logs a position_closed event to the Event Journal. No PnL or Duration here."""
        data = {"ticket": ticket, "exit_price": exit_price, "reason": reason}
        if extra_fields:
            data.update(extra_fields)
        self.log_event(signal_id, "position_closed", data)

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
        """Legacy support for log_outcome, now routes to position_closed event."""
        logger.warning(f"log_outcome is deprecated. Use log_position_closed for Layer 1 events. routing {signal_id} to position_closed.")
        self.log_position_closed(
            signal_id=signal_id,
            ticket=ticket,
            exit_price=close_price,
            reason=outcome,
            extra_fields=extra_fields
        )

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

    def log_lifecycle(self, lifecycle: PositionLifecycle) -> None:
        """Logs the complete PositionLifecycle object to a CSV summary (Layer 2)."""
        filepath = self._get_filepath(
            lifecycle.signal.strategy,
            lifecycle.signal.symbol,
            lifecycle.signal.timeframe,
            "lifecycle"
        )

        row_data = lifecycle.to_csv_row()
        # Add a record type for consistency if needed, but to_csv_row is already flattened
        self._write_row(filepath, row_data)

        # Also keep JSONL for full fidelity as it was before, but in positions/
        jsonl_path = filepath.replace(".csv", ".jsonl")
        lock = self._get_lock(jsonl_path)
        try:
            row_json = json.dumps(lifecycle.to_dict(), default=str)
            with lock:
                with open(jsonl_path, 'a') as f:
                    f.write(row_json + "\n")
            logger.info(f"Logged lifecycle JSON for {lifecycle.signal.signal_id} in {jsonl_path}")
        except Exception as e:
            logger.error(f"Failed to log lifecycle JSON to {jsonl_path}: {e}")

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
