import os
import json
import pandas as pd
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

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
        """Loads all journal CSVs for the current mode into a single DataFrame."""
        all_dfs = []
        mode_path = os.path.join(self.journal_root, self.mode)
        if not os.path.exists(mode_path):
            self.logger.warning(f"Journal path not found: {mode_path}")
            return pd.DataFrame()

        for root, dirs, files in os.walk(mode_path):
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

    def reconstruct_trade(self, ticket: Optional[int] = None, signal_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Reconstructs the full lifecycle of a trade by aggregating data from all sources.
        """
        ids = self.find_trade(ticket=ticket, signal_id=signal_id)
        if not ids:
            return {}

        ticket = ids.get("ticket")
        signal_id = ids.get("signal_id")

        audit_data = {
            "summary": {},
            "signal": {},
            "execution": {},
            "tracking": {},
            "exit_manager": {},
            "outcome": {},
            "journal": [],
            "broker": {},
            "recovery": {},
            "consistency": {},
            "timeline": []
        }

        # 1. Journal Data (Events)
        journal_df = self.load_journal_data()
        if not journal_df.empty:
            relevant_events = journal_df[journal_df['signal_id'] == signal_id].sort_values(by='system_timestamp')
            for _, row in relevant_events.iterrows():
                event = row.to_dict()
                audit_data["journal"].append(event)
                audit_data["timeline"].append({
                    "timestamp": event.get("system_timestamp"),
                    "event": event.get("event_type"),
                    "details": event
                })

                # Extract summary info from signal event
                if event.get("event_type") == "signal":
                    audit_data["signal"] = event
                    audit_data["summary"].update({
                        "signal_id": signal_id,
                        "strategy": event.get("strategy"),
                        "symbol": event.get("symbol"),
                        "direction": "BUY" if event.get("direction") == 1 else "SELL",
                        "timeframe": event.get("timeframe"),
                        "signal_category": event.get("signal_category"),
                    })

                if event.get("event_type") == "order_open":
                    audit_data["execution"] = event
                    if not ticket:
                        ticket = int(float(event.get("ticket")))

                if event.get("event_type") == "outcome":
                    audit_data["outcome"] = event

        audit_data["summary"]["ticket"] = ticket

        # 2. State Files
        # SendOrder State (Category)
        so_state = self.load_state_file("send_order_state.json")
        if ticket and str(ticket) in so_state:
            audit_data["summary"]["signal_category"] = so_state[str(ticket)]

        # PositionTracker State
        pt_state = self.load_state_file("position_tracker_state.json")
        for pos in pt_state.get("positions", []):
            if pos.get("ticket") == ticket:
                audit_data["tracking"] = pos
                break

        # ExitManager State
        em_state = self.load_state_file("exit_manager_state.json")
        tracked_tickets = em_state.get("tracked_tickets", {})
        if ticket and str(ticket) in tracked_tickets:
            audit_data["exit_manager"] = tracked_tickets[str(ticket)]

        # 3. MT5 History
        if ticket:
            history = self.get_mt5_history(position=ticket)
            audit_data["broker"] = history

            # Enrich timeline with broker events
            for deal in history.get("deals", []):
                audit_data["timeline"].append({
                    "timestamp": datetime.fromtimestamp(deal.get("time"), tz=timezone.utc).isoformat(),
                    "event": f"broker_deal_{deal.get('entry')}", # 0=entry, 1=exit
                    "details": deal
                })

            # Current open position?
            current_pos = self.get_mt5_position(ticket)
            if current_pos:
                audit_data["broker"]["current_position"] = current_pos

        # Sort timeline
        audit_data["timeline"].sort(key=lambda x: x["timestamp"] if x["timestamp"] else "")

        # 4. Consistency Checks
        audit_data["consistency"] = self.run_consistency_checks(audit_data)

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

    def format_report_markdown(self, data: Dict[str, Any]) -> str:
        """Formats the audit data as a Markdown report."""
        summary = data.get("summary", {})
        signal = data.get("signal", {})
        execution = data.get("execution", {})
        tracking = data.get("tracking", {})
        exit_mgr = data.get("exit_manager", {})
        outcome = data.get("outcome", {})
        timeline = data.get("timeline", [])
        checks = data.get("consistency", {})

        report = []
        report.append("==================================================")
        report.append("TRADE AUDIT REPORT")
        report.append("==================================================")

        report.append("\nTrade Summary")
        report.append("--------------------------------------------------")
        report.append(f"Signal ID: {summary.get('signal_id')}")
        report.append(f"Ticket: {summary.get('ticket')}")
        report.append(f"Strategy: {summary.get('strategy')}")
        report.append(f"Environment: {self.mode.capitalize()}")
        report.append(f"Symbol: {summary.get('symbol')}")
        report.append(f"Direction: {summary.get('direction')}")
        report.append(f"Timeframe: {summary.get('timeframe')}")
        report.append(f"Signal Category: {summary.get('signal_category')}")

        report.append("\nSignal Information")
        report.append("--------------------------------------------------")
        report.append(f"Signal Generated: {signal.get('system_timestamp')}")
        report.append(f"Bar Time: {signal.get('bar_timestamp')}")
        # Parse extra_fields if present
        extra = signal.get('extra_fields', {})
        if isinstance(extra, str):
            try:
                import ast
                extra = ast.literal_eval(extra)
            except:
                pass

        if isinstance(extra, dict):
            for k, v in extra.items():
                report.append(f"{k}: {v}")

        report.append("\nExecution")
        report.append("--------------------------------------------------")
        report.append(f"Actual Entry: {execution.get('actual_entry')}")
        report.append(f"Actual SL: {execution.get('actual_sl')}")
        report.append(f"Actual TP: {execution.get('actual_tp')}")
        report.append(f"Lot Size: {execution.get('lot_size')}")
        report.append(f"Risk %: {execution.get('risk_pct')}")

        # Calculate latency if possible
        if signal.get('system_timestamp') and execution.get('system_timestamp'):
            try:
                s_time = datetime.fromisoformat(signal.get('system_timestamp').replace('Z', '+00:00'))
                e_time = datetime.fromisoformat(execution.get('system_timestamp').replace('Z', '+00:00'))
                latency = (e_time - s_time).total_seconds()
                report.append(f"Execution Latency: {latency:.3f}s")
            except:
                pass

        report.append("\nPosition Tracking")
        report.append("--------------------------------------------------")
        report.append(f"Current PnL: ${tracking.get('floating_pnl')}")
        report.append(f"Remaining Risk: ${tracking.get('remaining_risk_dollars')}")

        report.append("\nExit Manager")
        report.append("--------------------------------------------------")
        report.append(f"Stage Mode: {exit_mgr.get('stage')}")
        report.append(f"Current Stage: {exit_mgr.get('current_stage_reached')}")
        report.append(f"Final TP Stage: {exit_mgr.get('final_tp')}")
        if exit_mgr.get('tp_prices'):
            report.append(f"TP Ladder: {exit_mgr.get('tp_prices')}")

        report.append("\nTrade Outcome")
        report.append("--------------------------------------------------")
        report.append(f"Exit Time: {outcome.get('system_timestamp')}")
        report.append(f"Duration: {outcome.get('duration_seconds')}s")
        report.append(f"Outcome Label: {outcome.get('outcome')}")
        report.append(f"Close Price: {outcome.get('close_price')}")
        report.append(f"PnL Dollars: ${outcome.get('pnl_dollars')}")

        report.append("\nConsistency Checks")
        report.append("--------------------------------------------------")
        for name, result in checks.items():
            report.append(f"{name:<25}: {result.get('status')} {result.get('msg')}")

        report.append("\nTimeline")
        report.append("--------------------------------------------------")
        report.append("```\n" + self.generate_ascii_timeline(timeline) + "\n```")

        return "\n".join(report)

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

    def save_reports(self, data: Dict[str, Any], output_dir: str = "AuditReports"):
        """Saves reports in Markdown and JSON formats."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        ticket = data.get("summary", {}).get("ticket", "unknown")

        # Markdown
        md_content = self.format_report_markdown(data)
        md_path = os.path.join(output_dir, f"audit_ticket_{ticket}.md")
        with open(md_path, "w") as f:
            f.write(md_content)

        # JSON
        json_path = os.path.join(output_dir, f"audit_ticket_{ticket}.json")
        with open(json_path, "w") as f:
            json.dump(data, f, indent=4, default=str)

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
                data = auditor.reconstruct_trade(ticket=trade['ticket'], signal_id=trade['signal_id'])
            else:
                sys.exit(0)
    elif args.ticket or args.signal_id:
        data = auditor.reconstruct_trade(ticket=args.ticket, signal_id=args.signal_id)
    else:
        parser.print_help()
        sys.exit(0)

    if not data or not data.get("summary"):
        print("Trade not found.")
        sys.exit(1)

    # Output
    if args.format == "console":
        print(auditor.format_report_markdown(data))
    elif args.format == "markdown":
        auditor.save_reports(data)
        print("Markdown report saved in AuditReports/")
    elif args.format == "json":
        auditor.save_reports(data)
        print("JSON report saved in AuditReports/")
