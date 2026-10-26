"""Planning arithmetic using user-provided tariffs; not a billing controller."""

import math


def plan_budget(credits=200.0, credits_per_hour=5.3, safety_fraction=0.2, session_minutes=30.0):
    if not all(math.isfinite(x) for x in (credits, credits_per_hour, safety_fraction, session_minutes)):
        raise ValueError("budget inputs must be finite")
    if credits <= 0 or credits_per_hour <= 0 or session_minutes <= 0 or not 0 < safety_fraction < 1:
        raise ValueError("invalid credits, tariff, session or safety reserve")
    usable = credits*(1-safety_fraction)
    session_cost = credits_per_hour*session_minutes/60
    return {"input_credits": credits, "input_credits_per_hour": credits_per_hour,
            "safety_fraction": safety_fraction, "reserved_credits": credits*safety_fraction,
            "usable_credits": usable, "planned_hours": usable/credits_per_hour,
            "session_minutes": session_minutes, "estimated_session_credits": session_cost,
            "whole_sessions_within_plan": math.floor(usable/session_cost),
            "is_estimate": True,
            "limitations": "User tariff may vary. Runtime setup, idle time and other notebook cells also consume resources. Step-boundary watchdog is not a billing cap."}
