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

from PositionManager.position_lifecycle import PositionLifecycle, PositionLifecycleBuilder

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

        if position:
            deals = mt5.history_deals_get(position=position)
            if deals:
                history_data["deals"] = [d._asdict() for d in deals]

            orders = mt5.history_orders_get(position=position)
            if orders:
                history_data["orders"] = [o._asdict() for o in orders]

        elif ticket:
            # Note: history_deals_get(ticket=...) returns specific deals,
            # usually we want history for the whole position
            deals = mt5.history_deals_get(ticket=ticket)
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

        if 'system_timestamp' in journal_df.columns:
            journal_df = journal_df.sort_values(by='system_timestamp', ascending=False)

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

    def reconstruct_lifecycle(self, ticket: Optional[int] = None, signal_id: Optional[str] = None) -> Optional[PositionLifecycle]:
        """
        Reconstructs the full lifecycle of a trade by aggregating data from all sources
        and returning a canonical PositionLifecycle object.
        """
        ids = self.find_trade(ticket=ticket, signal_id=signal_id)
        if not ids:
            return None

        ticket = ids.get("ticket")
        signal_id = ids.get("signal_id")

        # 1. Journal Data (Events)
        journal_events = []
        journal_df = self.load_journal_data()
        if not journal_df.empty:
            relevant_events = journal_df[journal_df['signal_id'] == signal_id].sort_values(by='system_timestamp')
            journal_events = [row.to_dict() for _, row in relevant_events.iterrows()]

        # 2. Broker Data
        broker_data = {}
        if ticket:
            broker_data = self.get_mt5_history(position=ticket)
            current_pos = self.get_mt5_position(ticket)
            if current_pos:
                broker_data["current_position"] = current_pos

        # 3. Framework State
        framework_state = {}
        # ExitManager State
        em_state = self.load_state_file("exit_manager_state.json")
        tracked_tickets = em_state.get("tracked_tickets", {})
        if ticket and str(ticket) in tracked_tickets:
            framework_state["exit_manager"] = tracked_tickets[str(ticket)]

        # PositionTracker State
        pt_state = self.load_state_file("position_tracker_state.json")
        for pos in pt_state.get("positions", []):
            if pos.get("ticket") == ticket:
                framework_state["tracking"] = pos
                break

        return PositionLifecycleBuilder.build_from_reconstruction(
            ticket=ticket,
            signal_id=signal_id,
            journal_events=journal_events,
            broker_data=broker_data,
            framework_state=framework_state
        )

    def generate_audit_data(self, lifecycle: PositionLifecycle) -> Dict[str, Any]:
        """
        Generates additional audit-specific data (timeline, consistency checks)
        from a PositionLifecycle object.
        """
        audit_data = {
            "lifecycle": lifecycle.to_dict(),
            "timeline": [],
            "consistency": {}
        }

        # Build timeline from journal events (stored in lifecycle.management for now or we need them raw)
        # Actually, let's just use the lifecycle's internal data for timeline if possible.
        # But for now, since we need to keep the audit report looking similar:

        # We can reconstruct a basic timeline from lifecycle
        if lifecycle.signal.signal_timestamp:
            audit_data["timeline"].append({
                "timestamp": lifecycle.signal.signal_timestamp,
                "event": "signal",
                "details": lifecycle.signal.to_dict()
            })

        if lifecycle.execution.ticket:
            audit_data["timeline"].append({
                "timestamp": lifecycle.signal.signal_timestamp, # Approx
                "event": "order_open",
                "details": lifecycle.execution.to_dict()
            })

        for pc in lifecycle.management.partial_closes:
            audit_data["timeline"].append({
                "timestamp": pc.get("system_timestamp"),
                "event": "partial_close",
                "details": pc
            })

        if lifecycle.outcome.exit_timestamp:
            audit_data["timeline"].append({
                "timestamp": lifecycle.outcome.exit_timestamp,
                "event": "outcome",
                "details": lifecycle.outcome.to_dict()
            })

        audit_data["timeline"].sort(key=lambda x: x["timestamp"] if x["timestamp"] else "")

        # Consistency Checks
        audit_data["consistency"] = self.run_consistency_checks(lifecycle)

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

    def format_report_markdown(self, lifecycle: PositionLifecycle, audit_extra: Dict[str, Any]) -> str:
        """Formats the audit data as a Markdown report."""
        checks = audit_extra.get("consistency", {})
        timeline = audit_extra.get("timeline", [])

        # We can use lifecycle.to_markdown() as a base or build a custom one
        report = []
        report.append("==================================================")
        report.append("TRADE AUDIT REPORT (RECONSTRUCTED)")
        report.append("==================================================")

        report.append("\nTrade Summary")
        report.append("--------------------------------------------------")
        report.append(f"Signal ID: {lifecycle.signal.signal_id}")
        report.append(f"Ticket: {lifecycle.execution.ticket}")
        report.append(f"Strategy: {lifecycle.signal.strategy}")
        report.append(f"Environment: {self.mode.capitalize()}")
        report.append(f"Symbol: {lifecycle.signal.symbol}")
        report.append(f"Direction: {'BUY' if lifecycle.signal.direction == 1 else 'SELL'}")
        report.append(f"Timeframe: {lifecycle.signal.timeframe}")
        report.append(f"Signal Category: {lifecycle.signal.signal_category}")

        report.append("\nSignal Information")
        report.append("--------------------------------------------------")
        report.append(f"Signal Generated: {lifecycle.signal.signal_timestamp}")
        report.append(f"Bar Time: {lifecycle.signal.bar_timestamp}")
        for k, v in lifecycle.signal.indicator_snapshot.items():
            report.append(f"{k}: {v}")

        report.append("\nExecution")
        report.append("--------------------------------------------------")
        report.append(f"Actual Entry: {lifecycle.execution.actual_entry}")
        report.append(f"Actual SL: {lifecycle.execution.initial_stop_loss}")
        report.append(f"Actual TP: {lifecycle.execution.initial_take_profit}")
        report.append(f"Lot Size: {lifecycle.execution.initial_volume}")
        report.append(f"Risk %: {lifecycle.execution.risk_percent}")
        report.append(f"Execution Latency: {lifecycle.execution.execution_latency:.3f}s")

        report.append("\nPosition Management")
        report.append("--------------------------------------------------")
        report.append(f"Current Stage: {lifecycle.management.current_stage}")
        report.append(f"Partial Closes: {len(lifecycle.management.partial_closes)}")

        report.append("\nTrade Outcome")
        report.append("--------------------------------------------------")
        report.append(f"Exit Time: {lifecycle.outcome.exit_timestamp}")
        report.append(f"Duration: {lifecycle.outcome.duration}s")
        report.append(f"Outcome Label: {lifecycle.outcome.result}")
        report.append(f"Close Price: {lifecycle.outcome.close_price}")
        report.append(f"PnL Dollars: ${lifecycle.outcome.realized_profit}")
        report.append(f"R-Multiple: {lifecycle.outcome.r_multiple:.2f}R")

        report.append("\nConsistency Checks")
        report.append("--------------------------------------------------")
        for name, result in checks.items():
            report.append(f"{name:<25}: {result.get('status')} {result.get('msg')}")

        report.append("\nTimeline")
        report.append("--------------------------------------------------")
        report.append("```\n" + self.generate_ascii_timeline(timeline) + "\n```")

        return "\n".join(report)

    def run_consistency_checks(self, lifecycle: PositionLifecycle) -> Dict[str, Dict[str, str]]:
        """Runs consistency checks on the lifecycle object."""
        checks = {}

        # 1. Completion Status
        checks["lifecycle_status"] = {
            "status": "PASS" if lifecycle.outcome.status == "completed" else "WARNING",
            "msg": f"Position status: {lifecycle.outcome.status}"
        }

        # 2. Latency Check
        if lifecycle.execution.execution_latency > 5.0:
            checks["execution_latency"] = {"status": "FAIL", "msg": f"High latency: {lifecycle.execution.execution_latency:.2f}s"}
        else:
            checks["execution_latency"] = {"status": "PASS", "msg": ""}

        # 3. SL/TP Validity
        if lifecycle.execution.initial_stop_loss == 0:
            checks["sl_presence"] = {"status": "FAIL", "msg": "No initial Stop Loss recorded"}
        else:
            checks["sl_presence"] = {"status": "PASS", "msg": ""}

        return checks

    def save_reports(self, lifecycle: PositionLifecycle, audit_extra: Dict[str, Any], output_dir: str = "AuditReports"):
        """Saves reports in Markdown and JSON formats."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        ticket = lifecycle.execution.ticket or "unknown"

        # Markdown
        md_content = self.format_report_markdown(lifecycle, audit_extra)
        md_path = os.path.join(output_dir, f"audit_ticket_{ticket}.md")
        with open(md_path, "w") as f:
            f.write(md_content)

        # JSON
        json_path = os.path.join(output_dir, f"audit_ticket_{ticket}.json")
        with open(json_path, "w") as f:
            full_data = {
                "lifecycle": lifecycle.to_dict(),
                "audit": audit_extra
            }
            json.dump(full_data, f, indent=4, default=str)

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

    lifecycle = None
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
                lifecycle = auditor.reconstruct_lifecycle(ticket=trade['ticket'], signal_id=trade['signal_id'])
            else:
                sys.exit(0)
    elif args.ticket or args.signal_id:
        lifecycle = auditor.reconstruct_lifecycle(ticket=args.ticket, signal_id=args.signal_id)
    else:
        parser.print_help()
        sys.exit(0)

    if not lifecycle:
        print("Trade not found or reconstruction failed.")
        sys.exit(1)

    audit_extra = auditor.generate_audit_data(lifecycle)

    # Output
    if args.format == "console":
        print(auditor.format_report_markdown(lifecycle, audit_extra))
    elif args.format == "markdown":
        auditor.save_reports(lifecycle, audit_extra)
        print("Markdown report saved in AuditReports/")
    elif args.format == "json":
        auditor.save_reports(lifecycle, audit_extra)
        print("JSON report saved in AuditReports/")
