from dataclasses import dataclass, field, asdict
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

@dataclass(frozen=True)
class SignalInfo:
    signal_id: str = ""
    strategy: str = ""
    signal_category: str = ""
    symbol: str = ""
    timeframe: str = ""
    direction: int = 0
    signal_timestamp: str = ""
    bar_timestamp: str = ""
    indicator_snapshot: Dict[str, Any] = field(default_factory=dict)
    market_snapshot: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class ExecutionInfo:
    ticket: Optional[int] = None
    magic_number: Optional[int] = None
    requested_entry: float = 0.0
    actual_entry: float = 0.0
    average_entry: float = 0.0
    initial_volume: float = 0.0
    remaining_volume: float = 0.0
    risk_percent: float = 0.0
    risk_amount: float = 0.0
    initial_stop_loss: float = 0.0
    initial_take_profit: float = 0.0
    spread: float = 0.0
    slippage: float = 0.0
    execution_latency: float = 0.0
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
    time_in_market: int = 0
    current_stage: int = 0
    management_events: List[Dict[str, Any]] = field(default_factory=list)

@dataclass(frozen=True)
class OutcomeInfo:
    exit_timestamp: str = ""
    average_exit_price: float = 0.0
    close_price: float = 0.0
    realized_profit: float = 0.0
    profit_points: float = 0.0
    profit_pips: float = 0.0
    profit_percent: float = 0.0
    r_multiple: float = 0.0
    result: str = ""
    strategy_reason: str = ""
    broker_reason: str = ""
    deal_count: int = 0
    partial_close_count: int = 0
    duration: int = 0
    status: str = ""

@dataclass(frozen=True)
class PositionLifecycle:
    signal: SignalInfo = field(default_factory=SignalInfo)
    execution: ExecutionInfo = field(default_factory=ExecutionInfo)
    management: ManagementInfo = field(default_factory=ManagementInfo)
    outcome: OutcomeInfo = field(default_factory=OutcomeInfo)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=4, default=str)

    def to_csv_row(self) -> Dict[str, Any]:
        """Flattens the object into a single-level dictionary for CSV logging."""
        flat = {}
        d = self.to_dict()
        for section, fields in d.items():
            for k, v in fields.items():
                if isinstance(v, (list, dict)):
                    flat[f"{section}_{k}"] = json.dumps(v)
                else:
                    flat[f"{section}_{k}"] = v
        return flat

    def to_markdown(self) -> str:
        """Generates a Markdown summary of the position lifecycle."""
        lines = [
            "## Position Lifecycle Report",
            f"- **Signal ID:** `{self.signal.signal_id}`",
            f"- **Ticket:** `{self.execution.ticket}`",
            "",
            "### 1. Signal",
            f"- **Strategy:** {self.signal.strategy}",
            f"- **Symbol:** {self.signal.symbol}",
            f"- **Direction:** {'BUY' if self.signal.direction == 1 else 'SELL'}",
            f"- **Category:** {self.signal.signal_category}",
            f"- **Timeframe:** {self.signal.timeframe}",
            f"- **Timestamp:** {self.signal.signal_timestamp}",
            "",
            "### 2. Execution",
            f"- **Entry Price:** {self.execution.actual_entry}",
            f"- **Volume:** {self.execution.initial_volume}",
            f"- **Risk:** {self.execution.risk_percent}% (${self.execution.risk_amount})",
            f"- **SL / TP:** {self.execution.initial_stop_loss} / {self.execution.initial_take_profit}",
            f"- **Latency:** {self.execution.execution_latency:.3f}s",
            "",
            "### 3. Management",
            f"- **Partial Closes:** {len(self.management.partial_closes)}",
            f"- **Current Stage:** {self.management.current_stage}",
            f"- **MFE / MAE:** {self.management.maximum_favorable_excursion:.2f} / {self.management.maximum_adverse_excursion:.2f}",
            "",
            "### 4. Outcome",
            f"- **Result:** {self.outcome.result}",
            f"- **Profit:** ${self.outcome.realized_profit} ({self.outcome.profit_pips:.1f} pips)",
            f"- **R-Multiple:** {self.outcome.r_multiple:.2f}R",
            f"- **Duration:** {self.outcome.duration}s",
            f"- **Status:** {self.outcome.status}"
        ]
        return "\n".join(lines)

class PositionLifecycleBuilder:
    """
    Builder responsible for constructing PositionLifecycle objects from various data sources.
    """
    @staticmethod
    def build_from_reconstruction(
        ticket: Optional[int] = None,
        signal_id: Optional[str] = None,
        journal_events: List[Dict[str, Any]] = None,
        broker_data: Dict[str, Any] = None,
        framework_state: Dict[str, Any] = None
    ) -> PositionLifecycle:

        journal_events = journal_events or []
        broker_data = broker_data or {}
        framework_state = framework_state or {}

        # 1. Extract Signal Info
        signal_event = next((e for e in journal_events if e.get("event_type") == "signal"), {})

        # indicator_snapshot and market_snapshot might be in extra_fields
        extra_fields = signal_event.get("extra_fields", {})
        if isinstance(extra_fields, str):
            try:
                import ast
                extra_fields = ast.literal_eval(extra_fields)
            except:
                extra_fields = {}

        # Some fields might be top-level in the journal row
        sig_info = SignalInfo(
            signal_id=signal_id or signal_event.get("signal_id", ""),
            strategy=signal_event.get("strategy", ""),
            signal_category=signal_event.get("signal_category", ""),
            symbol=signal_event.get("symbol", ""),
            timeframe=signal_event.get("timeframe", ""),
            direction=int(signal_event.get("direction", 0)) if signal_event.get("direction") else 0,
            signal_timestamp=signal_event.get("system_timestamp", ""),
            bar_timestamp=signal_event.get("bar_timestamp", ""),
            indicator_snapshot=extra_fields, # For now, putting extra_fields here
            market_snapshot={} # Placeholder
        )

        # 2. Extract Execution Info
        order_event = next((e for e in journal_events if e.get("event_type") == "order_open"), {})

        deals = broker_data.get("deals", [])
        entry_deal = next((d for d in deals if d.get("entry") == 0), {}) # 0 = DEAL_ENTRY_IN

        latency = 0.0
        if sig_info.signal_timestamp and order_event.get("system_timestamp"):
            try:
                s_time = datetime.fromisoformat(sig_info.signal_timestamp.replace('Z', '+00:00'))
                e_time = datetime.fromisoformat(order_event.get("system_timestamp").replace('Z', '+00:00'))
                latency = (e_time - s_time).total_seconds()
            except:
                pass

        exec_info = ExecutionInfo(
            ticket=ticket or int(order_event.get("ticket")) if order_event.get("ticket") else None,
            magic_number=entry_deal.get("magic"),
            requested_entry=float(signal_event.get("entry_price", 0.0)),
            actual_entry=float(order_event.get("actual_entry", entry_deal.get("price", 0.0))),
            average_entry=float(entry_deal.get("price", 0.0)),
            initial_volume=float(order_event.get("lot_size", entry_deal.get("volume", 0.0))),
            remaining_volume=0.0, # Will be updated if partial closes exist
            risk_percent=float(order_event.get("risk_pct", 0.0)),
            risk_amount=0.0, # Can calculate: balance * risk_percent
            initial_stop_loss=float(order_event.get("actual_sl", 0.0)),
            initial_take_profit=float(order_event.get("actual_tp", 0.0)),
            spread=0.0, # From broker data if available
            slippage=0.0, # actual_entry - requested_entry
            execution_latency=latency,
            broker_order_ids=[d.get("order") for d in deals if d.get("order")],
            broker_deal_ids=[d.get("ticket") for d in deals if d.get("ticket")]
        )

        # 3. Extract Management Info
        partial_events = [e for e in journal_events if e.get("event_type") == "partial_close"]
        exit_mgr_state = framework_state.get("exit_manager", {})

        mgmt_info = ManagementInfo(
            partial_closes=partial_events,
            current_stage=exit_mgr_state.get("current_stage_reached", 0),
            # MAE/MFE would need more complex calculation from bar data during trade
        )

        # 4. Extract Outcome Info
        outcome_event = next((e for e in journal_events if e.get("event_type") == "outcome"), {})

        exit_deals = [d for d in deals if d.get("entry") == 1] # 1 = DEAL_ENTRY_OUT
        realized_profit = sum(d.get("profit", 0.0) for d in deals)

        duration = 0
        if sig_info.signal_timestamp and outcome_event.get("system_timestamp"):
            try:
                s_time = datetime.fromisoformat(sig_info.signal_timestamp.replace('Z', '+00:00'))
                o_time = datetime.fromisoformat(outcome_event.get("system_timestamp").replace('Z', '+00:00'))
                duration = int((o_time - s_time).total_seconds())
            except:
                pass

        # R-Multiple calculation
        r_multiple = 0.0
        risk_dist = abs(exec_info.actual_entry - exec_info.initial_stop_loss)
        if risk_dist > 0:
            # Simple approximation: (exit_price - entry_price) / risk_dist
            # More accurate would be based on actual profit vs initial risk amount
            exit_price = float(outcome_event.get("close_price", 0.0))
            if exit_price == 0 and exit_deals:
                exit_price = exit_deals[-1].get("price", 0.0)

            pnl_points = (exit_price - exec_info.actual_entry) * sig_info.direction
            r_multiple = pnl_points / risk_dist

        out_info = OutcomeInfo(
            exit_timestamp=outcome_event.get("system_timestamp", ""),
            average_exit_price=exit_deals[-1].get("price", 0.0) if exit_deals else 0.0,
            close_price=float(outcome_event.get("close_price", 0.0)),
            realized_profit=realized_profit,
            profit_points=0.0, # (close_price - actual_entry) * direction
            profit_pips=0.0,   # profit_points / point / 10
            profit_percent=0.0,
            r_multiple=r_multiple,
            result=outcome_event.get("outcome", ""),
            duration=duration or int(outcome_event.get("duration_seconds", 0)),
            status="completed" if outcome_event else "open",
            deal_count=len(deals),
            partial_close_count=len(partial_events)
        )

        return PositionLifecycle(
            signal=sig_info,
            execution=exec_info,
            management=mgmt_info,
            outcome=out_info
        )
