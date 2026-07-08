import os
import json
import pandas as pd
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from Collecting_Data.position_lifecycle import PositionLifecycle, PositionLifecycleBuilder

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

class TradeAuditor:
    def __init__(
        self,
        journal_root: str = "Journals",
        state_dir: str = "State",
        mode: str = "live"
    ):
        self.journal_root = journal_root
        self.state_dir = state_dir
        self.mode = mode
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
        self.logger = logging.getLogger("TradeAuditor")

    def load_journal_data(self) -> pd.DataFrame:
        """
        Purpose:
            Aggregates all chronological event CSVs from the current mode's
            event directory into a single Pandas DataFrame.

        Returns:
            pd.DataFrame: A unified DataFrame of all events, sorted by timestamp.
        """
        all_dfs = []
        mode_path = os.path.join(self.journal_root, self.mode)
        if not os.path.exists(mode_path):
            self.logger.warning(f"Journal path not found: {mode_path}")
            return pd.DataFrame()

        for root, dirs, files in os.walk(mode_path):
            # Skip positions directory in new architecture to avoid mixing summaries with events
            if "positions" in dirs:
                dirs.remove("positions")

            for file in files:
                if file.endswith(".csv"):
                    filepath = os.path.join(root, file)
                    try:
                        df = pd.read_csv(filepath)
                        all_dfs.append(df)
                    except Exception as e:
                        self.logger.error(f"Failed to read {filepath}: {e}")

        if not all_dfs:
            return pd.DataFrame()

        return pd.concat(all_dfs, ignore_index=True)

    def load_state_file(self, filename: str) -> Dict[str, Any]:
        """Loads a framework JSON state file."""
        filepath = os.path.join(self.state_dir, filename)
        if not os.path.exists(filepath):
            self.logger.warning(f"State file not found: {filepath}")
            return {}

        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load {filepath}: {e}")
            return {}

    def get_mt5_history(self, ticket: Optional[int] = None, position: Optional[int] = None) -> Dict[str, Any]:
        """Fetches trade history from MT5."""
        if mt5 is None:
            self.logger.warning("MetaTrader5 not installed. Cannot fetch broker history.")
            return {"deals": [], "orders": []}

        if not mt5.initialize():
            self.logger.error(f"MT5 Initialization failed: {mt5.last_error()}")
            return {"deals": [], "orders": []}

        history_data = {"deals": [], "orders": []}

        # If position is provided, it's the ticket of the position.
        # mt5.history_deals_get(position=...) and mt5.history_orders_get(position=...)

        if position:
            deals = mt5.history_deals_get(position=position)
            if deals:
                history_data["deals"] = [d._asdict() for d in deals]

            orders = mt5.history_orders_get(position=position)
            if orders:
                history_data["orders"] = [o._asdict() for o in orders]

        elif ticket:
            deals = mt5.history_deals_get(ticket=ticket) # This might not be right for position history
            if deals:
                history_data["deals"] = [d._asdict() for d in deals]

            orders = mt5.history_orders_get(ticket=ticket)
            if orders:
                history_data["orders"] = [o._asdict() for o in orders]

        return history_data

    def get_mt5_position(self, ticket: int) -> Optional[Dict[str, Any]]:
        """Fetches current open position from MT5."""
        if mt5 is None:
            return None

        if not mt5.initialize():
            return None

        positions = mt5.positions_get(ticket=ticket)
        if positions:
            return positions[0]._asdict()
        return None

    def find_trade(self, ticket: Optional[int] = None, signal_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Attempts to find a trade by ticket or signal_id across journals.
        """
        journal_df = self.load_journal_data()

        if ticket:
            if not journal_df.empty and 'ticket' in journal_df.columns:
                # Convert ticket to float for comparison if it was read as such, or keep as is
                # Handle cases where ticket might be a float in the CSV (e.g. 123456.0)
                journal_df['ticket_str'] = journal_df['ticket'].apply(lambda x: str(int(float(x))) if pd.notna(x) else "")
                match = journal_df[journal_df['ticket_str'] == str(ticket)]
                if not match.empty:
                    sid = match.iloc[0]['signal_id']
                    return {"ticket": int(ticket), "signal_id": sid}
            return {"ticket": int(ticket), "signal_id": None}

        if signal_id:
            if not journal_df.empty:
                match = journal_df[journal_df['signal_id'] == signal_id]
                if not match.empty:
                    t_match = match.dropna(subset=['ticket'])
                    t = int(float(t_match.iloc[0]['ticket'])) if not t_match.empty else None
                    return {"ticket": t, "signal_id": signal_id}
            return {"ticket": None, "signal_id": signal_id}

        return {}

    def get_latest_trades(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Returns a list of the most recent trades from the journal."""
        journal_df = self.load_journal_data()
        if journal_df.empty:
            return []

        # Sort by system_timestamp descending
        if 'system_timestamp' in journal_df.columns:
            journal_df = journal_df.sort_values(by='system_timestamp', ascending=False)

        # Group by signal_id to get unique trades
        unique_trades = []
        seen_signals = set()

        for _, row in journal_df.iterrows():
            sid = row['signal_id']
            if sid not in seen_signals:
                seen_signals.add(sid)
                ticket = row.get('ticket')
                if pd.isna(ticket):
                    ticket = None
                else:
                    ticket = int(float(ticket))

                unique_trades.append({
                    "signal_id": sid,
                    "ticket": ticket,
                    "symbol": row.get('symbol', 'Unknown'),
                    "timestamp": row.get('system_timestamp', 'Unknown'),
                    "strategy": row.get('strategy', 'Unknown')
                })
                if len(unique_trades) >= limit:
                    break

        return unique_trades

    def reconstruct_trade_lifecycle(self, ticket: Optional[int] = None, signal_id: Optional[str] = None) -> Optional[PositionLifecycle]:
        """
        Purpose:
            The primary entry point for forensic analysis. Rebuilds a trade's
            entire history from discovery to closure.

        Arguments:
            ticket (int): Optional MT5 ticket ID.
            signal_id (str): Optional framework signal UUID.

        Returns:
            Optional[PositionLifecycle]: The fully populated lifecycle object
                                        or None if trade not found.

        Notes:
            Attempts to load from a completed summary (Layer 2) first,
            falling back to event reconstruction (Layer 1) if not yet closed.
        """
        ids = self.find_trade(ticket=ticket, signal_id=signal_id)
        if not ids:
            return None

        ticket = ids.get("ticket")
        signal_id = ids.get("signal_id")

        # TRY TO LOAD COMPLETED SUMMARY FIRST (Layer 2)
        summary_lifecycle = self.load_completed_lifecycle(signal_id)
        if summary_lifecycle:
            return summary_lifecycle

        # FALLBACK: RECONSTRUCT FROM EVENTS (Layer 1)
        journal_df = self.load_journal_data()

        # Gather Broker Data
        broker_data = None
        if ticket:
            broker_data = self.get_mt5_history(position=ticket)
            current_pos = self.get_mt5_position(ticket)
            if current_pos:
                broker_data["current_position"] = current_pos

        # Gather State Data
        state_files = {
            "exit_manager_state": self.load_state_file("exit_manager_state.json"),
            "position_tracker_state": self.load_state_file("position_tracker_state.json"),
            "send_order_state": self.load_state_file("send_order_state.json")
        }

        return PositionLifecycleBuilder.build_from_data(
            signal_id=signal_id,
            journal_df=journal_df,
            broker_data=broker_data,
            state_files=state_files
        )

    def reconstruct_trade(self, ticket: Optional[int] = None, signal_id: Optional[str] = None) -> Dict[str, Any]:
        """
        LEGACY: Reconstructs the full lifecycle of a trade by aggregating data from all sources.
        Now uses PositionLifecycle under the hood for consistency.
        """
        lifecycle = self.reconstruct_trade_lifecycle(ticket=ticket, signal_id=signal_id)
        if not lifecycle:
            return {}

        import dataclasses
        # Convert lifecycle to the old audit_data format for compatibility
        audit_data = {
            "summary": {
                "signal_id": lifecycle.signal.signal_id,
                "ticket": lifecycle.execution.ticket,
                "strategy": lifecycle.signal.strategy,
                "symbol": lifecycle.signal.symbol,
                "direction": "BUY" if lifecycle.signal.direction == 1 else "SELL",
                "timeframe": lifecycle.signal.timeframe,
                "signal_category": lifecycle.signal.signal_category,
            },
            "signal": dataclasses.asdict(lifecycle.signal),
            "execution": dataclasses.asdict(lifecycle.execution),
            "tracking": {}, # This was snapshot of current open risk, lifecycle.management has some of it
            "exit_manager": {},
            "outcome": dataclasses.asdict(lifecycle.outcome),
            "journal": [], # Builder currently doesn't keep full journal, but it's okay for legacy
            "broker": {},
            "recovery": {},
            "consistency": {},
            "timeline": []
        }

        # We can still run consistency checks on the audit_data if needed,
        # but the goal is to move towards lifecycle.

        return audit_data

    def generate_ascii_timeline(self, timeline: List[Dict[str, Any]]) -> str:
        """Generates an ASCII representation of the trade timeline."""
        lines = []
        for i, entry in enumerate(timeline):
            ts = entry.get("timestamp", "Unknown")
            event = entry.get("event", "Unknown")
            lines.append(f"{ts}")
            lines.append(f"  {event}")
            if i < len(timeline) - 1:
                lines.append("    │")
                lines.append("    ▼")
        return "\n".join(lines)

    def format_report_markdown(self, lifecycle: PositionLifecycle) -> str:
        """Formats the PositionLifecycle as a Markdown report."""
        return lifecycle.to_markdown()

    def run_consistency_checks(self, data: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
        """Runs consistency checks and anomaly detection."""
        summary = data.get("summary", {})
        journal = data.get("journal", [])
        broker = data.get("broker", {})
        tracking = data.get("tracking", {})
        exit_mgr = data.get("exit_manager", {})

        checks = {}

        # 1. Journal Completeness
        event_types = [e.get("event_type") for e in journal]
        checks["signal_event"] = {"status": "PASS" if "signal" in event_types else "FAIL", "msg": ""}
        checks["order_event"] = {"status": "PASS" if "order_open" in event_types else "FAIL", "msg": ""}
        checks["outcome_event"] = {"status": "PASS" if "outcome" in event_types else "WARNING", "msg": "No outcome event found in journal"}

        # 2. Broker Consistency
        if broker.get("deals"):
            deals = broker.get("deals")
            journal_entry = next((e for e in journal if e.get("event_type") == "order_open"), {})
            if journal_entry:
                broker_entry = next((d for d in deals if d.get("entry") == 0), {})
                if broker_entry:
                    price_diff = abs(broker_entry.get("price", 0) - journal_entry.get("actual_entry", 0))
                    if price_diff > 0.0001:
                        checks["price_match"] = {"status": "FAIL", "msg": f"Entry price mismatch: Broker {broker_entry.get('price')} vs Journal {journal_entry.get('actual_entry')}"}
                    else:
                        checks["price_match"] = {"status": "PASS", "msg": ""}
        else:
            checks["broker_history"] = {"status": "WARNING", "msg": "No broker history found"}

        # 3. State Restoration
        checks["tracker_restoration"] = {"status": "PASS" if tracking else "WARNING", "msg": "No active tracking record found"}
        checks["exit_manager_restoration"] = {"status": "PASS" if exit_mgr else "WARNING", "msg": "No ExitManager record found"}

        # 4. Anomalies
        anomalies = []
        if event_types.count("signal") > 1:
            anomalies.append("Duplicate signal events detected")
        if event_types.count("order_open") > 1:
            anomalies.append("Multiple order_open events detected for one signal")

        checks["anomalies"] = {"status": "PASS" if not anomalies else "FAIL", "msg": "; ".join(anomalies)}

        return checks

    def reconstruct_all(self) -> List[PositionLifecycle]:
        """Reconstructs all trades found in the journal."""
        journal_df = self.load_journal_data()
        if journal_df.empty:
            return []
            
        unique_signals = journal_df['signal_id'].unique()
        lifecycles = []
        for sid in unique_signals:
            lc = self.reconstruct_trade_lifecycle(signal_id=sid)
            if lc:
                lifecycles.append(lc)
        return lifecycles

    def load_completed_lifecycle(self, signal_id: str) -> Optional[PositionLifecycle]:
        """Attempts to load a completed PositionLifecycle from the summary CSV or JSONL."""
        mode_path = os.path.join(self.journal_root, self.mode)
        pos_path = os.path.join(mode_path, "positions")

        if not os.path.exists(pos_path):
            return None

        for root, dirs, files in os.walk(pos_path):
            for file in files:
                if file.endswith(".jsonl"):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r') as f:
                            for line in f:
                                data = json.loads(line)
                                if data.get('signal', {}).get('signal_id') == signal_id:
                                    # Convert back to object
                                    from Collecting_Data.position_lifecycle import SignalInfo, ExecutionInfo, ManagementInfo, OutcomeInfo

                                    sig = SignalInfo(**data['signal'])
                                    exc = ExecutionInfo(**data['execution'])
                                    mgt = ManagementInfo(**data['management'])
                                    out = OutcomeInfo(**data['outcome'])
                                    return PositionLifecycle(signal=sig, execution=exc, management=mgt, outcome=out)
                    except Exception as e:
                        self.logger.error(f"Failed to read lifecycle from {filepath}: {e}")
        return None

    def save_reports(self, lifecycle: PositionLifecycle, output_dir: str = "AuditReports"):
        """
        Purpose:
            Persists the audited lifecycle to disk in both human-readable
            (Markdown) and machine-readable (JSON) formats.

        Arguments:
            lifecycle (PositionLifecycle): The object to save.
            output_dir (str): Destination directory.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        ticket = lifecycle.execution.ticket

        # Markdown
        md_content = self.format_report_markdown(lifecycle)
        md_path = os.path.join(output_dir, f"audit_ticket_{ticket}.md")
        with open(md_path, "w") as f:
            f.write(md_content)

        # JSON
        json_path = os.path.join(output_dir, f"audit_ticket_{ticket}.json")
        with open(json_path, "w") as f:
            f.write(lifecycle.to_json())

        self.logger.info(f"Reports saved: {md_path}, {json_path}")

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Trade Auditor Developer Tool")
    parser.add_argument("--ticket", type=int, help="MT5 ticket number to audit")
    parser.add_argument("--signal-id", type=str, help="Signal ID to audit")
    parser.add_argument("--latest", action="store_true", help="Show latest trades")
    parser.add_argument("--mode", type=str, default="live", choices=["live", "backtest", "validation", "training"], help="Trading mode")
    parser.add_argument("--format", type=str, default="console", choices=["console", "markdown", "json"], help="Output format")

    args = parser.parse_args()

    auditor = TradeAuditor(mode=args.mode)

    if args.latest:
        latest = auditor.get_latest_trades()
        if not latest:
            print("No trades found in journals.")
        else:
            print("\nLATEST TRADES")
            print("-" * 60)
            for i, t in enumerate(latest):
                print(f"{i+1}. Ticket: {t['ticket']} | Symbol: {t['symbol']} | SID: {t['signal_id']} | {t['timestamp']}")
            print("-" * 60)
            choice = input("\nSelect trade number to audit (or Enter to cancel): ")
            if choice.isdigit() and 1 <= int(choice) <= len(latest):
                trade = latest[int(choice)-1]
                lifecycle = auditor.reconstruct_trade_lifecycle(ticket=trade['ticket'], signal_id=trade['signal_id'])
            else:
                sys.exit(0)
    elif args.ticket or args.signal_id:
        lifecycle = auditor.reconstruct_trade_lifecycle(ticket=args.ticket, signal_id=args.signal_id)
    else:
        parser.print_help()
        sys.exit(0)

    if not lifecycle:
        print("Trade not found.")
        sys.exit(1)

    # Output
    if args.format == "console":
        print(auditor.format_report_markdown(lifecycle))
    elif args.format == "markdown":
        auditor.save_reports(lifecycle)
        print("Markdown report saved in AuditReports/")
    elif args.format == "json":
        auditor.save_reports(lifecycle)
        print("JSON report saved in AuditReports/")
