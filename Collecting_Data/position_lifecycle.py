import json
import dataclasses
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from datetime import datetime, timezone

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

# Exit Profiles
EXIT_PROFILE_STANDARD = "standard"   # TP1 -> BE -> TP2
EXIT_PROFILE_SINGLE = "single"       # TP1 -> Full Close

@dataclass(frozen=True)
class SignalInfo:
    signal_id: str
    strategy: str
    signal_category: str
    exit_profile: str
    symbol: str
    timeframe: str
    direction: int
    signal_timestamp: str
    bar_timestamp: str
    indicator_snapshot: Dict[str, Any] = field(default_factory=dict)
    market_snapshot: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class ExecutionInfo:
    ticket: int
    magic_number: int
    requested_entry: float
    actual_entry: float
    average_entry: float
    initial_volume: float
    remaining_volume: float
    risk_percent: float
    risk_amount: float
    initial_stop_loss: float
    initial_take_profit: float
    spread: float
    slippage: float
    execution_latency: float
    broker_order_ids: List[int] = field(default_factory=list)
    broker_deal_ids: List[int] = field(default_factory=list)

@dataclass(frozen=True)
class ManagementInfo:
    partial_closes: List[Dict[str, Any]] = field(default_factory=list)
    breakeven_events: List[Dict[str, Any]] = field(default_factory=list)
    trailing_events: List[Dict[str, Any]] = field(default_factory=list)
    stop_loss_modifications: List[Dict[str, Any]] = field(default_factory=list)
    take_profit_modifications: List[Dict[str, Any]] = field(default_factory=list)
    maximum_favorable_excursion: float = 0.0
    maximum_adverse_excursion: float = 0.0
    highest_profit: float = 0.0
    lowest_profit: float = 0.0
    time_in_market: float = 0.0
    current_stage: int = 0
    management_events: List[Dict[str, Any]] = field(default_factory=list)

@dataclass(frozen=True)
class OutcomeInfo:
    exit_timestamp: str
    average_exit_price: float
    close_price: float
    realized_profit: float
    profit_points: float
    profit_pips: float
    profit_percent: float
    r_multiple: float
    result: str             # WIN, LOSS, BREAKEVEN
    strategy_reason: str    # e.g., TP1, TrailingStop
    broker_reason: str
    deal_count: int
    partial_close_count: int
    duration: float
    status: str

@dataclass(frozen=True)
class PositionLifecycle:
    signal: SignalInfo
    execution: ExecutionInfo
    management: ManagementInfo
    outcome: OutcomeInfo

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=4, default=str)

    def to_csv_row(self) -> Dict[str, Any]:
        """Flattens the object into a single-level dictionary for CSV logging."""
        flat = {}
        for section_name, section_obj in dataclasses.asdict(self).items():
            if isinstance(section_obj, dict):
                for key, value in section_obj.items():
                    flat[f"{section_name}_{key}"] = value
        return flat

    def to_markdown(self) -> str:
        """Returns a Markdown representation of the lifecycle."""
        md = []
        md.append(f"# Position Lifecycle - Ticket {self.execution.ticket}")
        md.append(f"**Signal ID:** `{self.signal.signal_id}`")
        md.append(f"**Status:** {self.outcome.status}")

        md.append("\n## 1. Signal Information")
        for f in dataclasses.fields(self.signal):
            val = getattr(self.signal, f.name)
            md.append(f"- **{f.name}:** {val}")

        md.append("\n## 2. Execution Information")
        for f in dataclasses.fields(self.execution):
            val = getattr(self.execution, f.name)
            md.append(f"- **{f.name}:** {val}")

        md.append("\n## 3. Management Information")
        for f in dataclasses.fields(self.management):
            val = getattr(self.management, f.name)
            md.append(f"- **{f.name}:** {val}")

        md.append("\n## 4. Outcome Information")
        for f in dataclasses.fields(self.outcome):
            val = getattr(self.outcome, f.name)
            md.append(f"- **{f.name}:** {val}")

        return "\n".join(md)

class PositionLifecycleBuilder:
    @staticmethod
    def build_from_data(
        signal_id: str,
        journal_df: pd.DataFrame,
        broker_data: Dict[str, Any] = None,
        state_files: Dict[str, Any] = None
    ) -> Optional[PositionLifecycle]:
        """
        Reconstructs a PositionLifecycle object from various data sources.
        """
        if journal_df.empty:
            return None

        relevant_events = journal_df[journal_df['signal_id'] == signal_id].sort_values(by='system_timestamp')
        if relevant_events.empty:
            return None

        # 1. Extract Signal Info
        signal_event = relevant_events[relevant_events['event_type'] == 'signal']
        if signal_event.empty:
            return None

        sig_row = signal_event.iloc[0]

        # Parse extra_fields/indicators if they exist in CSV
        indicator_snapshot = {}
        if 'extra_fields' in sig_row:
             # Try to parse if it's a string representation of a dict
             try:
                 import ast
                 if isinstance(sig_row['extra_fields'], str):
                    indicator_snapshot = ast.literal_eval(sig_row['extra_fields'])
             except:
                 pass

        signal_info = SignalInfo(
            signal_id=signal_id,
            strategy=sig_row.get('strategy', 'Unknown'),
            signal_category=sig_row.get('signal_category', 'Unknown'),
            exit_profile=sig_row.get('exit_profile', 'Unknown'),
            symbol=sig_row.get('symbol', 'Unknown'),
            timeframe=sig_row.get('timeframe', 'Unknown'),
            direction=int(sig_row.get('direction', 0)),
            signal_timestamp=sig_row.get('system_timestamp', 'Unknown'),
            bar_timestamp=sig_row.get('bar_timestamp', 'Unknown'),
            indicator_snapshot=indicator_snapshot
        )

        # 2. Extract Execution Info
        order_event = relevant_events[relevant_events['event_type'] == 'order_open']
        execution_info = None
        ticket = None

        if not order_event.empty:
            exec_row = order_event.iloc[0]
            ticket = int(float(exec_row.get('ticket', 0)))

            # Calculate latency
            latency = 0.0
            try:
                s_time = datetime.fromisoformat(sig_row['system_timestamp'].replace('Z', '+00:00'))
                e_time = datetime.fromisoformat(exec_row['system_timestamp'].replace('Z', '+00:00'))
                latency = (e_time - s_time).total_seconds()
            except:
                pass

            execution_info = ExecutionInfo(
                ticket=ticket,
                magic_number=0, # Need to get from broker or state
                requested_entry=float(sig_row.get('entry_price', 0)),
                actual_entry=float(exec_row.get('actual_entry', 0)),
                average_entry=float(exec_row.get('actual_entry', 0)),
                initial_volume=float(exec_row.get('lot_size', 0)),
                remaining_volume=0.0,
                risk_percent=float(exec_row.get('risk_pct', 0)),
                risk_amount=0.0,
                initial_stop_loss=float(exec_row.get('actual_sl', 0)),
                initial_take_profit=float(exec_row.get('actual_tp', 0)),
                spread=0.0,
                slippage=float(exec_row.get('actual_entry', 0)) - float(sig_row.get('entry_price', 0)),
                execution_latency=latency,
                broker_order_ids=[],
                broker_deal_ids=[]
            )
        else:
            # Minimal execution info if order_open is missing
            execution_info = ExecutionInfo(
                ticket=0, magic_number=0, requested_entry=0, actual_entry=0, average_entry=0,
                initial_volume=0, remaining_volume=0, risk_percent=0, risk_amount=0,
                initial_stop_loss=0, initial_take_profit=0, spread=0, slippage=0,
                execution_latency=0, broker_order_ids=[], broker_deal_ids=[]
            )

        # 3. Extract Management Info
        partial_events = relevant_events[relevant_events['event_type'] == 'partial_close']
        partial_closes = []
        for _, row in partial_events.iterrows():
            partial_closes.append(row.to_dict())

        management_info = ManagementInfo(
            partial_closes=partial_closes,
            current_stage=len(partial_closes),
            management_events=[row.to_dict() for _, row in relevant_events.iterrows() if row['event_type'] in ['enrichment', 'partial_close']]
        )

        # 4. Extract Outcome Info
        # In new architecture, we look for position_closed event
        closed_event = relevant_events[relevant_events['event_type'] == 'position_closed']

        # Backward compatibility: also check for legacy 'outcome' event type
        if closed_event.empty:
            closed_event = relevant_events[relevant_events['event_type'] == 'outcome']

        outcome_info = None

        if not closed_event.empty:
            out_row = closed_event.iloc[0]

            # In new architecture, PnL is NOT in the position_closed event.
            # We will rely on broker history if available, or stay 0 for now.
            pnl = float(out_row.get('pnl_dollars', 0))

            # Determine a preliminary result label
            res_label = "BREAKEVEN"
            if pnl > 0: res_label = "WIN"
            elif pnl < 0: res_label = "LOSS"

            outcome_info = OutcomeInfo(
                exit_timestamp=out_row.get('system_timestamp', 'Unknown'),
                average_exit_price=float(out_row.get('exit_price', out_row.get('close_price', 0))),
                close_price=float(out_row.get('exit_price', out_row.get('close_price', 0))),
                realized_profit=pnl,
                profit_points=0.0,
                profit_pips=0.0,
                profit_percent=0.0,
                r_multiple=0.0,
                result=res_label,
                strategy_reason=out_row.get('reason', out_row.get('outcome', 'Unknown')),
                broker_reason='',
                deal_count=0,
                partial_close_count=len(partial_closes),
                duration=float(out_row.get('duration_seconds', 0)),
                status='completed'
            )
        else:
             outcome_info = OutcomeInfo(
                exit_timestamp='', average_exit_price=0, close_price=0, realized_profit=0,
                profit_points=0, profit_pips=0, profit_percent=0, r_multiple=0,
                result='open', strategy_reason='', broker_reason='',
                deal_count=0, partial_close_count=len(partial_closes), duration=0, status='open'
            )

        # 5. Enrich with Broker Data if available
        if broker_data:
            deals = broker_data.get('deals', [])
            if deals:
                deal_ids = [d.get('ticket') if isinstance(d, dict) else d.ticket for d in deals]
                # Re-calculate execution info from deals
                entry_deal = next((d for d in deals if (d.get('entry') if isinstance(d, dict) else d.entry) == 0), None)
                if entry_deal:
                    ed = entry_deal if isinstance(entry_deal, dict) else entry_deal._asdict()
                    execution_info = dataclasses.replace(
                        execution_info,
                        actual_entry=ed.get('price'),
                        broker_deal_ids=deal_ids,
                        magic_number=ed.get('magic')
                    )

                # Re-calculate outcome info from deals
                exit_deals = [d for d in deals if (d.get('entry') if isinstance(d, dict) else d.entry) == 1]
                if exit_deals:
                    total_profit = sum((d.get('profit') if isinstance(d, dict) else d.profit) or 0.0 for d in deals)
                    last_exit = max(exit_deals, key=lambda d: d.get('time') if isinstance(d, dict) else d.time)
                    lex = last_exit if isinstance(last_exit, dict) else last_exit._asdict()

                    # Determine result label
                    res_label = "BREAKEVEN"
                    if total_profit > 0.01: # Use a small threshold for floating point
                        res_label = "WIN"
                    elif total_profit < -0.01:
                        res_label = "LOSS"

                    outcome_info = dataclasses.replace(
                        outcome_info,
                        realized_profit=total_profit,
                        exit_timestamp=datetime.fromtimestamp(lex.get('time'), tz=timezone.utc).isoformat(),
                        broker_reason=str(lex.get('reason')),
                        deal_count=len(deals),
                        result=res_label,
                        status='completed'
                    )

        # 6. Enrich with State Files
        if state_files:
            # e.g. from ExitManager state
            em_state = state_files.get('exit_manager_state', {})
            tracked = em_state.get('tracked_tickets', {})
            if ticket and str(ticket) in tracked:
                t_state = tracked[str(ticket)]
                management_info = dataclasses.replace(
                    management_info,
                    current_stage=t_state.get('current_stage_reached', 0)
                )

        return PositionLifecycle(
            signal=signal_info,
            execution=execution_info,
            management=management_info,
            outcome=outcome_info
        )
