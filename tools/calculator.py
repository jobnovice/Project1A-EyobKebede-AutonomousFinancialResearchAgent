"""
tools/calculator.py
Project1A-EyobKebede-AutonomousFinancialResearchAgent

Calculation engine — local financial computation, no external API.
No rate limits. Never fails due to network issues.

Supports all calculation_types from the schema:
  dcf_valuation      — discounted cash flow intrinsic value
  growth_rate        — period-over-period growth percentage
  cagr               — compound annual growth rate
  margin             — gross / operating / net / EBITDA margin
  ratio              — P/E, EV/EBITDA, D/E, current ratio, etc.
  yoy_change         — year-over-year absolute + percentage change
  peer_percentile    — rank a value within a peer set
  statistical_summary— mean, median, std dev, min, max
  custom             — user-defined formula string (secure AST parsing)

All calculation types support three output_format levels:
  value_only    — final number only
  with_steps    — result + key intermediate steps (default)
  full_breakdown— every intermediate value for full auditability
"""

from __future__ import annotations

import ast
import logging
import math
import statistics
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public tool function
# ---------------------------------------------------------------------------

def calculation_engine(
    calculation_type: str,
    inputs: dict,
    output_format: str = "with_steps",
    unit: str = "raw",
) -> dict:
    """
    Perform a financial calculation on provided inputs.

    Args:
        calculation_type: One of the supported calculation types (see module docstring).
        inputs:           Named input variables — see each handler for required keys.
        output_format:    'value_only' | 'with_steps' | 'full_breakdown'
        unit:              Expected output unit — used for result formatting.

    Returns:
        dict with keys:
            calculation_type, inputs, result, steps, output_format, unit,
            calculated_at, error
    """
    handlers = {
        "dcf_valuation":     _dcf_valuation,
        "growth_rate":        _growth_rate,
        "cagr":              _cagr,
        "margin":            _margin,
        "ratio":             _ratio,
        "yoy_change":        _yoy_change,
        "peer_percentile":   _peer_percentile,
        "statistical_summary": _statistical_summary,
        "custom":            _custom,
    }

    if calculation_type not in handlers:
        return _error_response(
            calculation_type, inputs,
            f"Unknown calculation_type '{calculation_type}'. "
            f"Supported: {list(handlers.keys())}",
            output_format=output_format,
            unit=unit
        )

    try:
        result, steps = handlers[calculation_type](inputs)
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as e:
        return _error_response(calculation_type, inputs, f"Calculation error: {e}", output_format=output_format, unit=unit)
    except Exception as e:
        return _error_response(calculation_type, inputs, f"Unexpected error: {e}", output_format=output_format, unit=unit)

    # Apply output_format filtering
    if output_format == "value_only":
        steps = {}
    elif output_format == "with_steps":
        # Keep only top-level steps, drop granular intermediates
        steps = {k: v for k, v in steps.items() if not k.startswith("_")}
    elif output_format == "full_breakdown":
        # Retain all calculated fields, including private granular intermediates
        pass

    # Format result based on unit
    formatted = _format_result(result, unit)

    return {
        "calculation_type": calculation_type,
        "inputs":           inputs,
        "result":           result,
        "result_formatted": formatted,
        "steps":            steps,
        "output_format":    output_format,
        "unit":             unit,
        "calculated_at":    _now(),
        "error":            None,
    }


# ---------------------------------------------------------------------------
# Calculation handlers — each returns (result: float, steps: dict)
# ---------------------------------------------------------------------------

def _growth_rate(inputs: dict) -> tuple[float, dict]:
    """
    Period-over-period growth rate.
    Inputs: current_value, prior_value, [label]
    """
    current = float(inputs["current_value"])
    prior   = float(inputs["prior_value"])
    label   = inputs.get("label", "value")

    if prior == 0:
        raise ZeroDivisionError(f"prior_value is 0 — cannot compute growth rate for {label}.")

    rate = (current - prior) / abs(prior)
    return rate, {
        "current":  current,
        "prior":    prior,
        "change":   round(current - prior, 4),
        "formula":  "(current - prior) / |prior|",
        "result":   f"{rate * 100:.2f}%",
    }


def _cagr(inputs: dict) -> tuple[float, dict]:
    """
    Compound annual growth rate.
    Inputs: start_value, end_value, periods (years)
    """
    start   = float(inputs["start_value"])
    end     = float(inputs["end_value"])
    periods = float(inputs["periods"])

    if start <= 0:
        raise ValueError("start_value must be positive for CAGR.")
    if periods <= 0:
        raise ValueError("periods must be positive.")

    cagr = (end / start) ** (1 / periods) - 1
    return cagr, {
        "start_value":  start,
        "end_value":    end,
        "periods":      periods,
        "formula":      "(end/start)^(1/periods) - 1",
        "result":       f"{cagr * 100:.2f}%",
        "total_growth": f"{((end/start) - 1) * 100:.2f}%",
    }


def _margin(inputs: dict) -> tuple[float, dict]:
    """
    Margin calculation (gross, operating, net, EBITDA, etc.).
    Inputs: numerator, denominator, [label]
    """
    numerator   = float(inputs["numerator"])
    denominator = float(inputs["denominator"])
    label       = inputs.get("label", "margin")

    if denominator == 0:
        raise ZeroDivisionError(f"denominator (revenue) is 0 — cannot compute {label}.")

    margin = numerator / denominator
    return margin, {
        "numerator":   numerator,
        "denominator": denominator,
        "label":       label,
        "formula":     "numerator / denominator",
        "result":      f"{margin * 100:.2f}%",
    }


def _ratio(inputs: dict) -> tuple[float, dict]:
    """
    Financial ratio computation.
    Inputs: numerator, denominator, ratio_name
    """
    numerator   = float(inputs["numerator"])
    denominator = float(inputs["denominator"])
    name        = inputs.get("ratio_name", "ratio")

    if denominator == 0:
        raise ZeroDivisionError(f"denominator is 0 — cannot compute {name}.")

    ratio = numerator / denominator
    return ratio, {
        "ratio_name":  name,
        "numerator":   numerator,
        "denominator": denominator,
        "formula":     "numerator / denominator",
        "result":      f"{ratio:.2f}x",
    }


def _yoy_change(inputs: dict) -> tuple[float, dict]:
    """
    Year-over-year absolute and percentage change.
    Inputs: current_year_value, prior_year_value, [metric_name]
    """
    current = float(inputs["current_year_value"])
    prior   = float(inputs["prior_year_value"])
    name    = inputs.get("metric_name", "metric")

    if prior == 0:
        raise ZeroDivisionError(f"prior year value is 0 for {name}.")

    pct_change = (current - prior) / abs(prior)
    abs_change = current - prior

    return pct_change, {
        "metric":         name,
        "current_year":   current,
        "prior_year":     prior,
        "absolute_change": round(abs_change, 4),
        "pct_change":      f"{pct_change * 100:.2f}%",
        "direction":      "increase" if pct_change > 0 else "decrease",
    }


def _dcf_valuation(inputs: dict) -> tuple[float, dict]:
    """
    Simplified DCF intrinsic value estimate.
    """
    fcf         = float(inputs["free_cash_flow"])
    g1          = float(inputs["growth_rate_yr1_5"])
    g2          = float(inputs.get("growth_rate_yr6_10", g1 * 0.6))
    g_terminal  = float(inputs.get("terminal_growth", 0.03))
    wacc        = float(inputs["discount_rate"])
    shares      = inputs.get("shares_outstanding")

    if wacc <= g_terminal:
        raise ValueError(
            f"discount_rate ({wacc}) must exceed terminal_growth ({g_terminal})."
        )

    # Phase 1: years 1–5
    phase1_fcfs = []
    pv_phase1   = 0.0
    cf = fcf
    for yr in range(1, 6):
        cf *= (1 + g1)
        pv  = cf / (1 + wacc) ** yr
        phase1_fcfs.append(round(cf, 2))
        pv_phase1 += pv

    # Phase 2: years 6–10
    phase2_fcfs = []
    pv_phase2   = 0.0
    for yr in range(6, 11):
        cf *= (1 + g2)
        pv  = cf / (1 + wacc) ** yr
        phase2_fcfs.append(round(cf, 2))
        pv_phase2 += pv

    # Terminal value (Gordon Growth Model)
    terminal_fcf = cf * (1 + g_terminal)
    terminal_value = terminal_fcf / (wacc - g_terminal)
    pv_terminal    = terminal_value / (1 + wacc) ** 10

    # Total intrinsic value
    intrinsic_value = pv_phase1 + pv_phase2 + pv_terminal

    steps: dict[str, Any] = {
        "free_cash_flow":      fcf,
        "growth_yr1_5":        f"{g1 * 100:.1f}%",
        "growth_yr6_10":       f"{g2 * 100:.1f}%",
        "terminal_growth":     f"{g_terminal * 100:.1f}%",
        "wacc":                f"{wacc * 100:.1f}%",
        "pv_phase1_yr1_5":     round(pv_phase1, 2),
        "pv_phase2_yr6_10":    round(pv_phase2, 2),
        "terminal_value":      round(terminal_value, 2),
        "pv_terminal_value":   round(pv_terminal, 2),
        "intrinsic_value":     round(intrinsic_value, 2),
        "_phase1_fcfs":        phase1_fcfs,
        "_phase2_fcfs":        phase2_fcfs,
    }

    if shares:
        per_share = intrinsic_value / float(shares)
        steps["shares_outstanding"]     = float(shares)
        steps["intrinsic_value_per_share"] = round(per_share, 2)

    return round(intrinsic_value, 2), steps


def _peer_percentile(inputs: dict) -> tuple[float, dict]:
    """
    Rank a company's metric within a peer set.
    """
    company = float(inputs["company_value"])
    peers   = [float(v) for v in inputs["peer_values"]]
    name    = inputs.get("metric_name", "metric")

    if not peers:
        raise ValueError("peer_values must be a non-empty list.")

    all_values = sorted(peers + [company])
    rank = all_values.index(company)
    percentile = rank / (len(all_values) - 1) if len(all_values) > 1 else 1.0

    return percentile, {
        "metric":       name,
        "company":      company,
        "peer_count":   len(peers),
        "peer_min":     min(peers),
        "peer_max":     max(peers),
        "peer_median":  statistics.median(peers),
        "rank":         f"{rank + 1} of {len(all_values)}",
        "percentile":   f"{percentile * 100:.1f}th",
    }


def _statistical_summary(inputs: dict) -> tuple[float, dict]:
    """
    Descriptive statistics for a list of values.
    """
    values = [float(v) for v in inputs["values"]]
    name   = inputs.get("metric_name", "values")

    if not values:
        raise ValueError("values list is empty.")

    mean   = statistics.mean(values)
    steps: dict[str, Any] = {
        "metric":  name,
        "count":   len(values),
        "mean":    round(mean, 4),
        "median":  round(statistics.median(values), 4),
        "min":     round(min(values), 4),
        "max":     round(max(values), 4),
    }
    if len(values) >= 2:
        steps["std_dev"] = round(statistics.stdev(values), 4)
        steps["range"]   = round(max(values) - min(values), 4)

    return round(mean, 4), steps


def _custom(inputs: dict) -> tuple[float, dict]:
    """
    Evaluate a user-defined arithmetic formula securely using AST parsing.
    Inputs: formula (string), variables (dict of name→value)
    
    Only basic arithmetic operations and whitelisted math functions are allowed.
    No execution environments, no builtins, no eval() vulnerabilities.
    """
    formula   = str(inputs.get("formula", ""))
    variables = {k: float(v) for k, v in inputs.get("variables", {}).items()}

    if not formula:
        raise ValueError("formula must be a non-empty string.")

    allowed_funcs = {
        "abs": abs, "round": round, "min": min, "max": max,
        "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
        "exp": math.exp, "pow": math.pow, "pi": math.pi,
    }

    try:
        tree = ast.parse(formula, mode="eval")
        
        def _eval_node(node: ast.AST) -> Any:
            if isinstance(node, ast.Expression):
                return _eval_node(node.body)
            elif isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.Num): #type:ignore # Legacy handling for older Python versions
                return node.n#type: ignore
            elif isinstance(node, ast.Name):
                if node.id in variables:
                    return variables[node.id]
                if node.id in allowed_funcs:
                    return allowed_funcs[node.id]
                raise NameError(f"Identifier '{node.id}' is blocked or not defined.")
            elif isinstance(node, ast.BinOp):
                left = _eval_node(node.left)
                right = _eval_node(node.right)
                if isinstance(node.op, ast.Add): return left + right
                if isinstance(node.op, ast.Sub): return left - right
                if isinstance(node.op, ast.Mult): return left * right
                if isinstance(node.op, ast.Div): return left / right
                if isinstance(node.op, ast.FloorDiv): return left // right
                if isinstance(node.op, ast.Mod): return left % right
                if isinstance(node.op, ast.Pow): return left ** right
                raise TypeError(f"Unsupported binary operator: {type(node.op)}")
            elif isinstance(node, ast.UnaryOp):
                operand = _eval_node(node.operand)
                if isinstance(node.op, ast.UAdd): return +operand
                if isinstance(node.op, ast.USub): return -operand
                raise TypeError(f"Unsupported unary operator: {type(node.op)}")
            elif isinstance(node, ast.Call):
                func = _eval_node(node.func)
                args = [_eval_node(arg) for arg in node.args]
                if callable(func):
                    return func(*args)
                raise TypeError("Specified identifier is not a callable function.")
            else:
                raise TypeError(f"Unsupported syntax structure: {type(node)}")

        res = _eval_node(tree)
        if isinstance(res, (int, float)):
            result = float(res)
        else:
            raise TypeError("Formula did not yield a valid numeric result.")
            
    except Exception as e:
        raise ValueError(f"Formula evaluation failed: {e}. Formula: '{formula}'")

    return result, {
        "formula":   formula,
        "variables": variables,
        "result":    result,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_result(result: float, unit: str) -> str:
    """Format result value based on expected unit."""
    if unit == "percentage":
        return f"{result * 100:.2f}%"
    if unit == "ratio":
        return f"{result:.2f}x"
    if unit == "currency_millions":
        return f"${result:,.1f}M"
    if unit == "currency_billions":
        return f"${result:,.2f}B"
    return str(round(result, 4))


def _error_response(calc_type: str, inputs: dict, message: str, output_format: str = "with_steps", unit: str = "raw") -> dict:
    logger.error("calculation_engine error [%s]: %s", calc_type, message)
    return {
        "calculation_type": calc_type,
        "inputs":           inputs,
        "result":           None,
        "result_formatted": None,
        "steps":            {},
        "output_format":    output_format,
        "unit":             unit,
        "calculated_at":    _now(),
        "error":            message,
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()