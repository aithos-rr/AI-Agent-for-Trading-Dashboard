from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime
from typing import Any, List, Optional
import json

import psycopg2
from dotenv import load_dotenv
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel


# Carica variabili d'ambiente da .env (se presente)
load_dotenv()


DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL non impostata. Imposta la variabile d'ambiente, "
        "ad esempio: postgresql://user:password@localhost:5432/trading_db",
    )


@contextmanager
def get_connection():
    """Context manager che restituisce una connessione PostgreSQL.

    Usa il DSN in DATABASE_URL.
    """

    conn = psycopg2.connect(DATABASE_URL)
    try:
        yield conn
    finally:
        conn.close()


# =====================
# Modelli di risposta API
# =====================


class BalancePoint(BaseModel):
    timestamp: datetime
    balance_usd: float


class OpenPosition(BaseModel):
    id: int
    snapshot_id: int
    symbol: str
    side: str
    size: float
    entry_price: Optional[float]
    mark_price: Optional[float]
    pnl_usd: Optional[float]
    leverage: Optional[str]
    snapshot_created_at: datetime


class BotOperation(BaseModel):
    id: int
    created_at: datetime
    operation: str
    symbol: Optional[str]
    direction: Optional[str]
    target_portion_of_balance: Optional[float]
    leverage: Optional[float]
    raw_payload: Any
    system_prompt: Optional[str]


class WinLossStats(BaseModel):
    wins: int
    losses: int
    win_rate: float


class CostStats(BaseModel):
    model_cost_total: float
    exchange_fee_total: float
    tax_total: float


# =====================
# App FastAPI + Template Jinja2
# =====================


app = FastAPI(
    title="Trading Agent Dashboard API",
    description=(
        "API per leggere i dati del trading agent dal database Postgres: "
        "saldo nel tempo, posizioni aperte, operazioni del bot con full prompt."
    ),
    version="0.3.1",
)

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


# =====================
# Endpoint API JSON
# =====================


@app.get("/balance", response_model=List[BalancePoint])
def get_balance() -> List[BalancePoint]:
    """Restituisce TUTTA la storia del saldo (balance_usd) ordinata nel tempo.

    I dati sono presi dalla tabella `account_snapshots`.
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT created_at, balance_usd
                FROM account_snapshots
                ORDER BY created_at ASC;
                """
            )
            rows = cur.fetchall()

    return [
        BalancePoint(timestamp=row[0], balance_usd=float(row[1]))
        for row in rows
    ]


@app.get("/open-positions", response_model=List[OpenPosition])
def get_open_positions() -> List[OpenPosition]:
    """Restituisce le posizioni aperte dell'ULTIMO snapshot disponibile.

    - Prende l'ultimo record da `account_snapshots`.
    - Recupera le posizioni corrispondenti da `open_positions`.
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            # Ultimo snapshot
            cur.execute(
                """
                SELECT id, created_at
                FROM account_snapshots
                ORDER BY created_at DESC
                LIMIT 1;
                """
            )
            row = cur.fetchone()
            if not row:
                return []
            snapshot_id = row[0]
            snapshot_created_at = row[1]

            # Posizioni aperte per quello snapshot
            cur.execute(
                """
                SELECT
                    id,
                    snapshot_id,
                    symbol,
                    side,
                    size,
                    entry_price,
                    mark_price,
                    pnl_usd,
                    leverage
                FROM open_positions
                WHERE snapshot_id = %s
                ORDER BY symbol ASC, id ASC;
                """,
                (snapshot_id,),
            )
            rows = cur.fetchall()

    return [
        OpenPosition(
            id=row[0],
            snapshot_id=row[1],
            symbol=row[2],
            side=row[3],
            size=float(row[4]),
            entry_price=float(row[5]) if row[5] is not None else None,
            mark_price=float(row[6]) if row[6] is not None else None,
            pnl_usd=float(row[7]) if row[7] is not None else None,
            leverage=row[8],
            snapshot_created_at=snapshot_created_at,
        )
        for row in rows
    ]


@app.get("/bot-operations", response_model=List[BotOperation])
def get_bot_operations(
    limit: int = Query(
        12,
        ge=1,
        le=500,
        description="Numero massimo di operazioni da restituire (default 12)",
    ),
) -> List[Any]:
    """
    Restituisce le ultime `limit` operazioni del bot con:
    - system prompt
    - reasoning
    - full prompt
    - indicatori REALI presi da indicators_contexts (by ticker)
    """

    with get_connection() as conn:
        with conn.cursor() as cur:

            # 1) Preleva le operazioni
            cur.execute(
                """
                SELECT
                    bo.id,
                    bo.created_at,
                    bo.operation,
                    bo.symbol,
                    bo.direction,
                    bo.target_portion_of_balance,
                    bo.leverage,
                    bo.raw_payload,
                    bo.context_id,
                    ac.system_prompt
                FROM bot_operations AS bo
                LEFT JOIN ai_contexts AS ac ON bo.context_id = ac.id
                ORDER BY bo.created_at DESC
                LIMIT %s;
                """,
                (limit,),
            )
            operations_raw = cur.fetchall()

            operations = []
            for row in operations_raw:
                # Parse raw_payload
                raw_payload = row[7]
                if isinstance(raw_payload, str):
                    try:
                        raw_payload = json.loads(raw_payload)
                    except:
                        raw_payload = {}
                elif raw_payload is None:
                    raw_payload = {}
                if not isinstance(raw_payload, dict):
                    raw_payload = {}

                created_at = row[1]
                created_at_fmt = created_at.strftime("%d/%m/%Y %H:%M") if created_at else ""
                
                symbol = row[3]
                
                # Fetch indicators by ticker
                indicators = {}
                if symbol:
                    cur.execute(
                        """
                        SELECT price, ema20, macd, rsi_7, volume_bid, volume_ask
                        FROM indicators_contexts
                        WHERE ticker = %s
                        ORDER BY ts DESC
                        LIMIT 1;
                        """,
                        (symbol,)
                    )
                    ind_row = cur.fetchone()
                    if ind_row:
                        indicators = {
                            "price": ind_row[0],
                            "ema20": ind_row[1],
                            "macd": ind_row[2],
                            "rsi": ind_row[3],
                            "volume_bid": ind_row[4],
                            "volume_ask": ind_row[5],
                        }

                operations.append({
                    "id": row[0],
                    "created_at": created_at,
                    "created_at_fmt": created_at_fmt,
                    "operation": row[2],
                    "symbol": symbol,
                    "direction": row[4],
                    "target_portion_of_balance": float(row[5]) if row[5] else None,
                    "leverage": float(row[6]) if row[6] else None,
                    "raw_payload": raw_payload,
                    "system_prompt": row[9],
                    "indicators": indicators,
                })

    return operations



def compute_closed_positions_stats() -> WinLossStats:
    """Calculate Win/Loss statistics by analyzing open -> close sequences.
    
    Logic:
    1. Fetch all operations ordered by time.
    2. Group by symbol.
    3. Track open positions and match with close operations.
    4. Calculate PNL based on entry/exit price and size.
    """
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            # Fetch all operations to reconstruct history
            cur.execute(
                """
                SELECT symbol, operation, direction, created_at, raw_payload
                FROM bot_operations
                ORDER BY created_at ASC;
                """
            )
            rows = cur.fetchall()
            
    wins = 0
    losses = 0
    
    # Track open positions: symbol -> {entry_price, size, direction}
    open_positions = {}
    
    for symbol, operation, direction, created_at, raw_payload in rows:
        if isinstance(raw_payload, str):
            try:
                payload = json.loads(raw_payload)
            except:
                payload = {}
        else:
            payload = raw_payload if raw_payload else {}
            
        # Normalize operation string
        op_type = operation.lower()
        
        if op_type == 'open':
            # Store open position details
            # Assuming payload has price/size or we use defaults
            price = payload.get("market_data", {}).get("price") or payload.get("entry_price") or 0
            size = payload.get("size", 0) # If size not in payload, might be issue, but proceed
            
            open_positions[symbol] = {
                "entry_price": float(price),
                "direction": direction.lower() if direction else "long",
                "size": float(size)
            }
            
        elif op_type == 'close':
            if symbol in open_positions:
                pos = open_positions[symbol]
                entry_price = pos["entry_price"]
                direction = pos["direction"]
                # Exit price
                exit_price = payload.get("market_data", {}).get("price") or payload.get("exit_price") or 0
                exit_price = float(exit_price)
                
                # Calculate PNL
                # PNL = (Exit - Entry) if Long
                # PNL = (Entry - Exit) if Short
                # We ignore size for win/loss count, just sign matters
                
                if direction == "long":
                    pnl = exit_price - entry_price
                else: # short
                    pnl = entry_price - exit_price
                    
                if pnl > 0:
                    wins += 1
                elif pnl < 0:
                    losses += 1
                
                # Remove from open positions
                del open_positions[symbol]

    total = wins + losses
    win_rate = (wins / total * 100) if total > 0 else 0.0
    
    return WinLossStats(wins=wins, losses=losses, win_rate=win_rate)


# =====================
# Endpoint HTML + HTMX
# =====================


@app.get("/", response_class=HTMLResponse)
@app.get("/ui/dashboard", response_class=HTMLResponse)
async def ui_dashboard(request: Request):
    """Dashboard principale HTML."""
    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request},
    )


@app.get("/ui/performance-overview", response_class=HTMLResponse)
async def ui_performance_overview(request: Request):

    with get_connection() as conn:
        with conn.cursor() as cur:
            # Saldo iniziale
            cur.execute("""
                SELECT balance_usd FROM account_snapshots
                ORDER BY created_at ASC LIMIT 1;
            """)
            initial = cur.fetchone()
            initial_balance = float(initial[0]) if initial else 0

            # Saldo finale
            cur.execute("""
                SELECT balance_usd FROM account_snapshots
                ORDER BY created_at DESC LIMIT 1;
            """)
            current = cur.fetchone()
            current_balance = float(current[0]) if current else 0

    total_pnl = current_balance - initial_balance
    pct = (total_pnl / initial_balance * 100) if initial_balance > 0 else 0

    return templates.TemplateResponse(
        "partials/performance_overview.html",
        {
            "request": request,
            "initial_balance": initial_balance,
            "current_balance": current_balance,
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round(pct, 2),
        },
    )


@app.get("/ui/open-positions", response_class=HTMLResponse)
async def ui_open_positions(request: Request):
    """Partial HTML con le posizioni aperte (ultimo snapshot)."""

    positions = get_open_positions()
    return templates.TemplateResponse(
        "partials/open_positions_table.html",
        {"request": request, "positions": positions},
    )


@app.get("/ui/balance", response_class=HTMLResponse)
async def ui_balance(request: Request):
    """Partial HTML con il grafico del saldo nel tempo."""

    points = get_balance()
    labels = [p.timestamp.isoformat() for p in points]
    values = [p.balance_usd for p in points]
    return templates.TemplateResponse(
        "partials/balance_table.html",
        {"request": request, "labels": labels, "values": values},
    )


@app.get("/ui/bot-operations", response_class=HTMLResponse)
async def ui_bot_operations(request: Request):

    operations = get_bot_operations(limit=12)

    return templates.TemplateResponse(
        "partials/bot_operations_table.html",
        {"request": request, "operations": operations},
    )


@app.get("/ui/win-loss", response_class=HTMLResponse)
async def ui_win_loss(request: Request):

    stats = compute_closed_positions_stats()

    return templates.TemplateResponse(
        "partials/win_loss_chart.html",
        {
            "request": request,
            "wins": stats.wins,
            "losses": stats.losses,
            "winrate": round(stats.win_rate, 2),
        },
    )


@app.get("/ui/costs", response_class=HTMLResponse)
async def ui_costs(request: Request):

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT created_at, total_cost_usd, llm_cost_usd,
                       input_tokens, output_tokens, symbol
                FROM cost_events
                ORDER BY created_at DESC
                LIMIT 20;
            """)
            rows = cur.fetchall()

    cost_labels = [row[0].strftime("%Y-%m-%d %H:%M") for row in rows]
    cost_values = [float(row[1]) for row in rows]

    costs = [
        {
            "created_at": row[0],
            "total_cost": float(row[1]),
            "llm_cost": float(row[2]),
            "input_tokens": row[3],
            "output_tokens": row[4],
            "symbol": row[5],
        }
        for row in rows
    ]

    return templates.TemplateResponse(
        "partials/costs_chart.html",
        {
            "request": request,
            "cost_labels": cost_labels,
            "cost_values": cost_values,
            "costs": costs,
        },
    )


# Comodo per sviluppo locale: `python main.py`
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="localhost", port=8000, reload=True)
