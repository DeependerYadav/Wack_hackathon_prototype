import calendar
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Household Electricity Optimization Platform",
    page_icon="\u26A1",
    layout="wide",
)

APPLIANCES = ["TV", "Geyser", "Refrigerator", "AC", "Washing Machine", "Fans", "Other"]


def apply_styles() -> None:
    st.markdown(
        """
        <style>
            .main .block-container {
                padding-top: 1.1rem;
                padding-bottom: 2rem;
            }
            h1, h2, h3 {
                color: #0f172a;
            }
            div[data-testid="metric-container"] {
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 10px;
                padding: 12px;
            }
            .alert-box {
                padding: 0.8rem 1rem;
                border-radius: 10px;
                margin: 0.5rem 0;
                border-left: 6px solid;
            }
            .footer {
                margin-top: 2rem;
                text-align: center;
                color: #64748b;
                font-size: 0.9rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def generate_demo_data(days: int = 45, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(end=pd.Timestamp.now().floor("h"), periods=days * 24, freq="h")

    rows = []
    for ts in timestamps:
        hour = ts.hour

        tv = max(0.0, rng.normal(0.04 + (0.20 if 19 <= hour <= 23 else 0.01), 0.04))
        geyser = max(0.0, rng.normal(0.02 + (0.90 if 6 <= hour <= 8 else 0.55 if 19 <= hour <= 21 else 0.0), 0.20))
        refrigerator = max(0.0, rng.normal(0.12, 0.015))
        ac = max(0.0, rng.normal(0.05 + (0.75 if 12 <= hour <= 17 else 0.45 if 18 <= hour <= 23 else 0.0), 0.18))
        washing_machine = max(0.0, rng.normal(0.55 if hour in [7, 8, 14, 20] and rng.random() < 0.18 else 0.0, 0.10))
        fans = max(0.0, rng.normal(0.06 + (0.24 if 10 <= hour <= 23 else 0.08), 0.05))
        other = max(0.0, rng.normal(0.05, 0.02))

        total = tv + geyser + refrigerator + ac + washing_machine + fans + other
        rows.append(
            {
                "timestamp": ts,
                "TV": tv,
                "Geyser": geyser,
                "Refrigerator": refrigerator,
                "AC": ac,
                "Washing Machine": washing_machine,
                "Fans": fans,
                "Other": other,
                "total_kwh": total,
            }
        )

    df = pd.DataFrame(rows)

    spike_count = max(6, len(df) // 200)
    spike_idx = rng.choice(df.index.to_numpy(), size=spike_count, replace=False)
    spike_mult = rng.uniform(1.8, 2.5, size=spike_count)
    df.loc[spike_idx, "total_kwh"] = df.loc[spike_idx, "total_kwh"] * spike_mult

    return df.round(3)


def parse_meter_csv(uploaded_file) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    try:
        raw = pd.read_csv(uploaded_file)
    except Exception:
        uploaded_file.seek(0)
        try:
            raw = pd.read_csv(uploaded_file, encoding="latin1")
        except Exception as exc:
            return None, f"Unable to read CSV: {exc}"

    if raw.empty:
        return None, "CSV file is empty."

    time_candidates = [
        col
        for col in raw.columns
        if any(token in str(col).strip().lower() for token in ["timestamp", "datetime", "date", "time"])
    ]
    time_col = time_candidates[0] if time_candidates else raw.columns[0]

    raw[time_col] = pd.to_datetime(raw[time_col], errors="coerce")
    raw = raw.dropna(subset=[time_col]).copy()
    if raw.empty:
        return None, "No valid timestamp values were found in the uploaded CSV."

    raw = raw.rename(columns={time_col: "timestamp"})

    for col in raw.columns:
        if col != "timestamp":
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw = raw.fillna(0)

    def normalize_label(label: str) -> str:
        label = str(label).strip().lower().replace("_", " ").replace("-", " ")
        return " ".join(label.split())

    alias_to_target = {
        "tv": "TV",
        "television": "TV",
        "geyser": "Geyser",
        "heater": "Geyser",
        "water heater": "Geyser",
        "waterheater": "Geyser",
        "refrigerator": "Refrigerator",
        "fridge": "Refrigerator",
        "ac": "AC",
        "aircon": "AC",
        "air conditioner": "AC",
        "airconditioner": "AC",
        "air conditioning": "AC",
        "washing machine": "Washing Machine",
        "washing": "Washing Machine",
        "washer": "Washing Machine",
        "laundry": "Washing Machine",
        "fan": "Fans",
        "fans": "Fans",
        "other": "Other",
        "misc": "Other",
        "miscellaneous": "Other",
    }

    rename_map = {}
    for col in raw.columns:
        if col == "timestamp":
            continue
        normalized = normalize_label(col)
        tokens = set(normalized.split())

        target = alias_to_target.get(normalized)
        if target is None:
            if {"washing", "machine"}.issubset(tokens):
                target = "Washing Machine"
            elif {"water", "heater"}.issubset(tokens):
                target = "Geyser"
            elif {"air", "conditioner"}.issubset(tokens) or {"air", "conditioning"}.issubset(tokens):
                target = "AC"
            elif tokens.intersection({"tv", "television"}):
                target = "TV"
            elif tokens.intersection({"fridge", "refrigerator"}):
                target = "Refrigerator"
            elif tokens.intersection({"fan", "fans"}):
                target = "Fans"
            elif tokens.intersection({"other", "misc", "miscellaneous"}):
                target = "Other"

        if target is not None:
            rename_map[col] = target

    raw = raw.rename(columns=rename_map)
    if raw.columns.duplicated().any():
        ts_col = raw["timestamp"]
        other = raw.drop(columns=["timestamp"])
        other = other.groupby(by=other.columns, axis=1).sum()
        raw = pd.concat([ts_col, other], axis=1)

    numeric_cols = [
        col for col in raw.columns if col != "timestamp" and pd.api.types.is_numeric_dtype(raw[col])
    ]
    if not numeric_cols:
        return None, "No numeric consumption columns were detected in the CSV."

    total_candidates = [
        col
        for col in numeric_cols
        if "total" in col.lower() or col.lower() in {"kwh", "usage", "consumption"}
    ]

    if total_candidates:
        raw["total_kwh"] = raw[total_candidates[0]]
    elif "total_kwh" not in raw.columns:
        raw["total_kwh"] = raw[numeric_cols].sum(axis=1)

    cleaned = raw.copy()
    cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"], errors="coerce")
    cleaned = cleaned.dropna(subset=["timestamp"]).sort_values("timestamp")
    cleaned = cleaned.reset_index(drop=True)
    cleaned["total_kwh"] = pd.to_numeric(cleaned["total_kwh"], errors="coerce").fillna(0).clip(lower=0)

    if cleaned.empty:
        return None, "Uploaded data had no valid rows after preprocessing."

    return cleaned, None


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    if "timestamp" not in data.columns:
        data = data.rename(columns={data.columns[0]: "timestamp"})

    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data = data.dropna(subset=["timestamp"]).copy()

    for col in data.columns:
        if col != "timestamp":
            data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)

    if "total_kwh" not in data.columns:
        numeric_cols = [col for col in data.columns if col != "timestamp"]
        data["total_kwh"] = data[numeric_cols].sum(axis=1) if numeric_cols else 0

    data["total_kwh"] = data["total_kwh"].clip(lower=0)
    data = data.sort_values("timestamp").reset_index(drop=True)

    if data.empty:
        data = generate_demo_data()

    return data


def ensure_appliances(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    data = df.copy()
    present = [name for name in APPLIANCES if name in data.columns and name != "Other"]

    if not present:
        data["Other"] = data["total_kwh"]
        return data, ["Other"]

    residual = (data["total_kwh"] - data[present].sum(axis=1)).clip(lower=0)
    data["Other"] = residual

    used = present.copy()
    if data["Other"].sum() > 0:
        used.append("Other")

    return data, used


def detect_anomalies(series: pd.Series) -> pd.Series:
    median = float(series.median())
    mad = float(np.median(np.abs(series - median)))
    if mad == 0:
        return pd.Series(False, index=series.index)
    robust_z = 0.6745 * (series - median) / mad
    return robust_z > 3.5


def predict_bill(df: pd.DataFrame, rate: float) -> tuple[float, float, float, pd.DataFrame]:
    now = pd.Timestamp.now()
    month_start = pd.Timestamp(year=now.year, month=now.month, day=1)
    days_in_month = calendar.monthrange(now.year, now.month)[1]
    month_dates = pd.date_range(month_start, periods=days_in_month, freq="D")

    daily_usage = df.set_index("timestamp").resample("D")["total_kwh"].sum()
    month_daily = daily_usage.reindex(month_dates)

    observed = month_daily.iloc[: now.day].dropna()
    fallback_daily = float(daily_usage.tail(30).mean()) if not daily_usage.empty else 0.0

    if observed.empty:
        avg_daily = fallback_daily
    else:
        avg_daily = float(observed.mean())

    units_so_far = float(avg_daily * now.day)
    predicted_units = float(max(avg_daily * days_in_month, 0.0))
    predicted_bill = float(predicted_units * rate)

    trend_rows = []
    for idx, date in enumerate(month_dates, start=1):
        if idx <= now.day:
            value = month_daily.iloc[idx - 1]
            value = float(value) if pd.notna(value) else avg_daily
            kind = "Actual"
        else:
            value = avg_daily
            kind = "Forecast"
        trend_rows.append({"Date": date, "Units (kWh)": value, "Type": kind})

    return predicted_units, predicted_bill, units_so_far, pd.DataFrame(trend_rows)


def compute_energy_score(df: pd.DataFrame, predicted_units: float) -> tuple[int, str, float]:
    daily = df.set_index("timestamp").resample("D")["total_kwh"].sum()
    avg_daily = float(daily.tail(30).mean()) if not daily.empty else 0.0

    night = df[df["timestamp"].dt.hour.between(0, 4)]["total_kwh"]
    night_baseline = float(night.mean()) if not night.empty else 0.0

    score = 100.0
    score -= max(predicted_units - 350.0, 0.0) / 6.0
    score -= max(avg_daily - 12.0, 0.0) * 2.0
    score -= max(night_baseline - 0.30, 0.0) * 65.0
    score = float(np.clip(score, 0, 100))

    if score >= 80:
        category = "Efficient"
    elif score >= 60:
        category = "Moderate"
    else:
        category = "Inefficient"

    return int(round(score)), category, night_baseline


def compute_alerts(df: pd.DataFrame, rate: float) -> tuple[list[dict], float]:
    temp = df.copy()
    temp["hour"] = temp["timestamp"].dt.hour

    overall_avg = float(temp["total_kwh"].mean()) if not temp.empty else 0.0
    morning_avg = float(temp[temp["hour"].between(6, 9)]["total_kwh"].mean()) if not temp.empty else 0.0
    night_avg = float(temp[temp["hour"].between(0, 4)]["total_kwh"].mean()) if not temp.empty else 0.0
    hidden_baseline = float(temp.groupby("hour")["total_kwh"].mean().min()) if not temp.empty else 0.0

    alerts = []

    morning_ratio = morning_avg / (overall_avg + 1e-9)
    if morning_ratio > 1.35:
        alerts.append(
            {
                "level": "red",
                "title": "Morning Overuse",
                "message": f"High morning load detected (avg {morning_avg:.2f} kWh/hr). Shift heavy appliances away from 6 AM-10 AM.",
            }
        )
    elif morning_ratio > 1.15:
        alerts.append(
            {
                "level": "yellow",
                "title": "Morning Overuse",
                "message": f"Morning usage is slightly elevated at {morning_avg:.2f} kWh/hr.",
            }
        )
    else:
        alerts.append(
            {
                "level": "green",
                "title": "Morning Overuse",
                "message": "Morning usage is within an efficient range.",
            }
        )

    if night_avg > 0.45:
        alerts.append(
            {
                "level": "red",
                "title": "Night Baseline Warning",
                "message": f"Night baseline is high ({night_avg:.2f} kWh/hr). Check standby and always-on devices.",
            }
        )
    elif night_avg > 0.30:
        alerts.append(
            {
                "level": "yellow",
                "title": "Night Baseline Warning",
                "message": f"Night baseline is moderate ({night_avg:.2f} kWh/hr).",
            }
        )
    else:
        alerts.append(
            {
                "level": "green",
                "title": "Night Baseline Warning",
                "message": "Night baseline is healthy.",
            }
        )

    if hidden_baseline > 0.35:
        alerts.append(
            {
                "level": "red",
                "title": "Hidden Load Detection",
                "message": f"Persistent hidden load detected ({hidden_baseline:.2f} kWh/hr minimum).",
            }
        )
    elif hidden_baseline > 0.22:
        alerts.append(
            {
                "level": "yellow",
                "title": "Hidden Load Detection",
                "message": f"Potential hidden load present ({hidden_baseline:.2f} kWh/hr minimum).",
            }
        )
    else:
        alerts.append(
            {
                "level": "green",
                "title": "Hidden Load Detection",
                "message": "No significant hidden load pattern detected.",
            }
        )

    savings_morning = max(morning_avg - overall_avg, 0) * 4 * 30 * rate
    savings_night = max(night_avg - 0.25, 0) * 5 * 30 * rate
    savings_hidden = max(hidden_baseline - 0.18, 0) * 24 * 30 * rate * 0.35
    potential_savings = float(max(savings_morning + savings_night + savings_hidden, 0.0))

    return alerts, potential_savings


def format_hour_window(hour: int) -> str:
    start = pd.Timestamp(year=2000, month=1, day=1, hour=int(hour))
    end = start + pd.Timedelta(hours=1)
    return f"{start.strftime('%I:%M %p').lstrip('0')} - {end.strftime('%I:%M %p').lstrip('0')}"


def generate_usage_suggestions(
    df: pd.DataFrame,
    appliance_cols: list[str],
    rate: float,
) -> tuple[list[dict], float]:
    temp = df.copy()
    temp["hour"] = temp["timestamp"].dt.hour

    hourly_avg = temp.groupby("hour")["total_kwh"].mean()
    if hourly_avg.empty:
        return [], 0.0

    overall_avg = float(hourly_avg.mean())
    usable_appliances = [col for col in appliance_cols if col in temp.columns]
    app_hourly = (
        temp.groupby("hour")[usable_appliances].mean()
        if usable_appliances
        else pd.DataFrame(index=hourly_avg.index)
    )

    suggestions = []
    estimated_total_savings = 0.0

    for hour, avg_usage in hourly_avg.sort_values(ascending=False).head(4).items():
        ratio = float(avg_usage / (overall_avg + 1e-9))
        if ratio < 1.08 and len(suggestions) >= 2:
            continue

        dominant_devices = []
        if not app_hourly.empty and hour in app_hourly.index:
            top_devices = app_hourly.loc[hour].sort_values(ascending=False).head(2)
            dominant_devices = [name for name, val in top_devices.items() if val > 0.02]
        dominant_text = ", ".join(dominant_devices) if dominant_devices else "mixed household load"

        if 0 <= hour <= 5:
            reason = "high overnight baseline suggests standby or always-on equipment."
            action = "Switch off idle electronics, use smart plugs, and timer-control water heaters overnight."
        elif 6 <= hour <= 9:
            reason = "morning overlap of heavy appliances is likely creating the peak."
            action = "Avoid running geyser, washing machine, and other heavy loads in the same hour."
        elif 10 <= hour <= 17:
            reason = "daytime cooling demand is likely driving higher electricity draw."
            action = "Set AC to 24-26C, close curtains, and clean AC filters to reduce compressor runtime."
        else:
            reason = "evening activity and cooling loads are likely peaking together."
            action = "Shift laundry/geyser cycles earlier and reduce simultaneous AC + entertainment + cooking overlap."

        if "AC" in dominant_devices:
            action += " Prioritize AC optimization first."
        elif "Geyser" in dominant_devices:
            action += " Install a geyser timer or reduce water-heating duration."
        elif "Other" in dominant_devices:
            action += " Audit hidden loads on this time slot to identify unknown consumption."

        level = "red" if ratio >= 1.45 else "yellow" if ratio >= 1.20 else "green"
        excess = max(float(avg_usage) - overall_avg, 0.0)
        savings = excess * 30 * rate * 0.30
        estimated_total_savings += savings

        suggestions.append(
            {
                "level": level,
                "time_window": format_hour_window(int(hour)),
                "avg_usage": float(avg_usage),
                "dominant": dominant_text,
                "reason": reason,
                "action": action,
                "savings": float(savings),
            }
        )

    if not suggestions:
        suggestions.append(
            {
                "level": "green",
                "time_window": "All day profile",
                "avg_usage": float(overall_avg),
                "dominant": "balanced load",
                "reason": "no strongly abnormal high-usage time band was detected.",
                "action": "Maintain current pattern and focus on efficient appliance settings.",
                "savings": 0.0,
            }
        )

    return suggestions, float(estimated_total_savings)


def render_alert(level: str, title: str, message: str) -> None:
    color_map = {
        "green": {"bg": "#e8f5e9", "border": "#2e7d32", "text": "#1b5e20"},
        "yellow": {"bg": "#fff8e1", "border": "#f9a825", "text": "#8d6e00"},
        "red": {"bg": "#fdecea", "border": "#d32f2f", "text": "#b71c1c"},
    }
    style = color_map.get(level, color_map["yellow"])
    st.markdown(
        f"""
        <div class="alert-box" style="background:{style['bg']}; border-left-color:{style['border']}; color:{style['text']};">
            <strong>{title}</strong><br>{message}
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_report(
    source: str,
    rate: float,
    today_usage: float,
    month_usage: float,
    predicted_units: float,
    predicted_bill: float,
    score: int,
    category: str,
    savings: float,
) -> str:
    return "\n".join(
        [
            "SmartEnergy AI Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Data Source: {source}",
            "",
            "Key Metrics",
            f"Today Usage (kWh): {today_usage:.2f}",
            f"Month-to-Date Usage (kWh): {month_usage:.2f}",
            f"Predicted Monthly Units (kWh): {predicted_units:.2f}",
            f"Predicted Monthly Bill ($): {predicted_bill:.2f}",
            f"Electricity Rate ($/kWh): {rate:.2f}",
            f"Energy Score: {score}/100 ({category})",
            f"Potential Monthly Savings ($): {savings:.2f}",
        ]
    )


def main() -> None:
    apply_styles()

    st.title("SmartEnergy AI – Household Electricity Insight & Optimization Platform")

    if "uploaded_data" not in st.session_state:
        st.session_state["uploaded_data"] = None
    if "electricity_rate" not in st.session_state:
        st.session_state["electricity_rate"] = 0.15
    if "demo_mode" not in st.session_state:
        st.session_state["demo_mode"] = True

    page = st.sidebar.radio(
        "Navigation",
        [
            "\U0001F3E0 Dashboard",
            "\U0001F4CA Smart Meter Analysis",
            "\U0001F9EE Appliance Calculator",
            "\u26A0 Insights & Alerts",
            "\U0001F4B0 Bill Prediction",
            "\U0001F331 Energy Score",
        ],
    )

    st.sidebar.toggle("Demo Mode", key="demo_mode")

    if page == "\U0001F4CA Smart Meter Analysis":
        uploaded = st.file_uploader("Upload Smart Meter CSV", type=["csv"])
        if uploaded is not None:
            with st.spinner("Processing uploaded CSV..."):
                parsed, error = parse_meter_csv(uploaded)
            if error:
                st.markdown(f"<p style='color:#b91c1c; font-weight:600;'>{error}</p>", unsafe_allow_html=True)
            else:
                st.session_state["uploaded_data"] = parsed
                st.success("CSV loaded successfully. Uploaded CSV is now the active data source.")
            
            csv_buffer = parsed.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Processed CSV",
                data=csv_buffer,
                file_name="processed_smart_meter_data.csv",
                mime="text/csv",
            )
        elif not st.session_state["demo_mode"] and st.session_state["uploaded_data"] is None:
            st.info("No CSV uploaded yet. Demo data is currently being used.")

    if st.session_state["uploaded_data"] is not None:
        active_data = st.session_state["uploaded_data"].copy()
        source_label = "Uploaded CSV"
    elif st.session_state["demo_mode"]:
        active_data = generate_demo_data()
        source_label = "Demo Mode"
    else:
        active_data = generate_demo_data()
        source_label = "Demo Fallback (no CSV uploaded)"

    with st.spinner("Loading energy insights..."):
        data = preprocess_data(active_data)
        data, appliance_cols = ensure_appliances(data)

        now = pd.Timestamp.now()
        
        # Determine display date (handle historical data)
        max_ts = data["timestamp"].max()
        if pd.notna(max_ts) and max_ts.date() < (now.date() - pd.Timedelta(days=1)):
            display_date = max_ts.date()
            date_label = f"Latest ({display_date:%b %d})"
        else:
            display_date = now.date()
            date_label = "Today"

        today_usage = float(data[data["timestamp"].dt.date == display_date]["total_kwh"].sum())
        month_usage = float(
            data[
                (data["timestamp"].dt.month == now.month)
                & (data["timestamp"].dt.year == now.year)
            ]["total_kwh"].sum()
        )

        base_rate = float(st.session_state["electricity_rate"])
        predicted_units, predicted_bill, units_so_far, trend_df = predict_bill(data, base_rate)
        score, category, _ = compute_energy_score(data, predicted_units)
        alerts, potential_savings = compute_alerts(data, base_rate)
        usage_suggestions, usage_savings = generate_usage_suggestions(data, appliance_cols, base_rate)

    report_text = build_report(
        source=source_label,
        rate=base_rate,
        today_usage=today_usage,
        month_usage=month_usage,
        predicted_units=predicted_units,
        predicted_bill=predicted_bill,
        score=score,
        category=category,
        savings=potential_savings,
    )

    st.sidebar.caption(f"Data Source: {source_label}")
    st.sidebar.download_button(
        label="Download Report",
        data=report_text,
        file_name="smartenergy_report.txt",
        mime="text/plain",
    )

    if page == "\U0001F3E0 Dashboard":
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric(f"{date_label} (kWh)", f"{today_usage:.1f}")
        kpi2.metric("Monthly (kWh)", f"{month_usage:.1f}")
        kpi3.metric("Predicted Bill", f"${predicted_bill:.2f}")
        kpi4.metric("Energy Score", f"{score}/100", category)

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            hourly = data.set_index("timestamp").resample("h")["total_kwh"].sum().tail(72).reset_index()
            fig_hourly = px.line(
                hourly,
                x="timestamp",
                y="total_kwh",
                title="Hourly Usage (Last 72 Hours)",
                markers=True,
            )
            fig_hourly.update_layout(template="plotly_white", xaxis_title="Time", yaxis_title="kWh", height=360)
            st.plotly_chart(fig_hourly, use_container_width=True)

        with chart_col2:
            dist = data[appliance_cols].sum().reset_index()
            dist.columns = ["Appliance", "kWh"]
            fig_pie = px.pie(
                dist,
                names="Appliance",
                values="kWh",
                hole=0.45,
                title="Appliance Consumption Distribution",
            )
            fig_pie.update_layout(template="plotly_white", height=360)
            st.plotly_chart(fig_pie, use_container_width=True)

    elif page == "\U0001F4CA Smart Meter Analysis":
        st.subheader("Data Preview")
        st.caption(f"Current source: {source_label}")
        st.dataframe(data.head(20), use_container_width=True)

        hourly = data.set_index("timestamp").resample("h")["total_kwh"].sum().reset_index()
        daily = data.set_index("timestamp").resample("D")["total_kwh"].sum().reset_index()

        line_col, bar_col = st.columns(2)

        with line_col:
            fig_line = px.line(hourly, x="timestamp", y="total_kwh", title="Hourly Usage", markers=True)
            fig_line.update_layout(template="plotly_white", xaxis_title="Time", yaxis_title="kWh", height=340)
            st.plotly_chart(fig_line, use_container_width=True)

        with bar_col:
            fig_bar = px.bar(daily, x="timestamp", y="total_kwh", title="Daily Consumption", color="total_kwh")
            fig_bar.update_layout(template="plotly_white", xaxis_title="Date", yaxis_title="kWh", height=340)
            st.plotly_chart(fig_bar, use_container_width=True)

        heat = data.copy()
        heat["Date"] = heat["timestamp"].dt.date
        heat["Hour"] = heat["timestamp"].dt.hour
        pivot = heat.pivot_table(index="Date", columns="Hour", values="total_kwh", aggfunc="sum", fill_value=0).tail(30)
        fig_heat = px.imshow(
            pivot,
            labels={"x": "Hour of Day", "y": "Date", "color": "kWh"},
            title="Usage Heatmap",
            color_continuous_scale="YlOrRd",
            aspect="auto",
        )
        fig_heat.update_layout(template="plotly_white", height=380)
        st.plotly_chart(fig_heat, use_container_width=True)

        anomaly_mask = detect_anomalies(hourly["total_kwh"])
        anomaly_count = int(anomaly_mask.sum())

        fig_anomaly = go.Figure()
        fig_anomaly.add_trace(
            go.Scatter(
                x=hourly["timestamp"],
                y=hourly["total_kwh"],
                mode="lines",
                name="Usage",
                line=dict(color="#1f77b4", width=2),
            )
        )
        fig_anomaly.add_trace(
            go.Scatter(
                x=hourly.loc[anomaly_mask, "timestamp"],
                y=hourly.loc[anomaly_mask, "total_kwh"],
                mode="markers",
                name="Anomaly",
                marker=dict(color="red", size=9),
            )
        )
        fig_anomaly.update_layout(
            title="Anomaly Detection (Red Markers)",
            template="plotly_white",
            xaxis_title="Time",
            yaxis_title="kWh",
            height=350,
        )
        st.plotly_chart(fig_anomaly, use_container_width=True)

        if anomaly_count > 0:
            st.markdown(
                f"<p style='color:#b91c1c; font-weight:600;'>Warning: {anomaly_count} anomalies detected in hourly usage.</p>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<p style='color:#166534; font-weight:600;'>No significant anomalies detected.</p>",
                unsafe_allow_html=True,
            )

        st.subheader("AI Suggestions to Reduce High Electricity Use")
        if source_label != "Uploaded CSV":
            st.info("Upload your own CSV and disable Demo Mode to get personalized recommendations from your meter history.")

        for suggestion in usage_suggestions:
            title = (
                f"{suggestion['time_window']} | Avg {suggestion['avg_usage']:.2f} kWh/hr | "
                f"Likely drivers: {suggestion['dominant']}"
            )
            message = (
                f"<strong>Possible reason:</strong> {suggestion['reason']}<br>"
                f"<strong>How to reduce:</strong> {suggestion['action']}<br>"
                f"<strong>Estimated monthly savings:</strong> ${suggestion['savings']:.2f}"
            )
            render_alert(suggestion["level"], title, message)

        st.metric("Time-Based Optimization Potential", f"${usage_savings:.2f} / month")

    elif page == "\U0001F9EE Appliance Calculator":
        specs_kw = {
            "TV": 0.12,
            "Geyser": 2.00,
            "Refrigerator": 0.15,
            "AC": 1.50,
            "Washing Machine": 0.50,
            "Fans": 0.075,
        }
        defaults = {
            "TV": 4.0,
            "Geyser": 1.0,
            "Refrigerator": 24.0,
            "AC": 6.0,
            "Washing Machine": 1.0,
            "Fans": 10.0,
        }

        input_cols = st.columns(3)
        usage_hours = {}
        for idx, name in enumerate(specs_kw):
            with input_cols[idx % 3]:
                usage_hours[name] = st.number_input(
                    f"{name} Hours/Day",
                    min_value=0.0,
                    max_value=24.0,
                    value=float(defaults[name]),
                    step=0.5,
                )

        rate = float(st.session_state["electricity_rate"])
        calc_rows = []
        for name, power in specs_kw.items():
            daily_units = power * usage_hours[name]
            monthly_units = daily_units * 30
            monthly_cost = monthly_units * rate
            calc_rows.append(
                {
                    "Appliance": name,
                    "Power (kW)": power,
                    "Hours/Day": usage_hours[name],
                    "Daily Units (kWh)": daily_units,
                    "Monthly Units (kWh)": monthly_units,
                    "Monthly Cost ($)": monthly_cost,
                }
            )

        calc_df = pd.DataFrame(calc_rows)

        m1, m2, m3 = st.columns(3)
        m1.metric("Daily Units", f"{calc_df['Daily Units (kWh)'].sum():.2f} kWh")
        m2.metric("Monthly Units", f"{calc_df['Monthly Units (kWh)'].sum():.2f} kWh")
        m3.metric("Estimated Monthly Cost", f"${calc_df['Monthly Cost ($)'].sum():.2f}")

        st.dataframe(
            calc_df.style.format(
                {
                    "Power (kW)": "{:.2f}",
                    "Hours/Day": "{:.1f}",
                    "Daily Units (kWh)": "{:.2f}",
                    "Monthly Units (kWh)": "{:.2f}",
                    "Monthly Cost ($)": "${:.2f}",
                }
            ),
            use_container_width=True,
        )

        fig_calc = px.bar(
            calc_df,
            x="Appliance",
            y="Monthly Units (kWh)",
            color="Monthly Units (kWh)",
            title="Monthly Consumption by Appliance",
            text_auto=".1f",
        )
        fig_calc.update_layout(template="plotly_white", height=360)
        st.plotly_chart(fig_calc, use_container_width=True)

    elif page == "\u26A0 Insights & Alerts":
        for alert in alerts:
            render_alert(alert["level"], alert["title"], alert["message"])

        st.metric("Potential Monthly Savings", f"${potential_savings:.2f}")
        st.caption("Savings estimate assumes operational adjustments and reduced standby losses.")

    elif page == "\U0001F4B0 Bill Prediction":
        rate_input = st.number_input(
            "Electricity Rate ($/kWh)",
            min_value=0.01,
            max_value=5.00,
            value=float(st.session_state["electricity_rate"]),
            step=0.01,
        )
        st.session_state["electricity_rate"] = rate_input

        pred_units, pred_bill, mtd_units, bill_trend = predict_bill(data, rate_input)

        c1, c2, c3 = st.columns(3)
        c1.metric("Month-to-Date Units", f"{mtd_units:.2f} kWh")
        c2.metric("Predicted Monthly Units", f"{pred_units:.2f} kWh")
        c3.metric("Predicted Bill", f"${pred_bill:.2f}")

        fig_trend = px.line(
            bill_trend,
            x="Date",
            y="Units (kWh)",
            color="Type",
            markers=True,
            title="Daily Consumption Trend and Forecast",
            color_discrete_map={"Actual": "#1f77b4", "Forecast": "#ff7f0e"},
        )
        fig_trend.update_layout(template="plotly_white", height=380)
        st.plotly_chart(fig_trend, use_container_width=True)

    elif page == "\U0001F331 Energy Score":
        score_value, score_category, _ = compute_energy_score(data, predicted_units)

        g1, g2 = st.columns([1, 1])

        with g1:
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=score_value,
                    number={"suffix": "/100"},
                    title={"text": "Energy Efficiency Score"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#0f766e"},
                        "steps": [
                            {"range": [0, 60], "color": "#fee2e2"},
                            {"range": [60, 80], "color": "#fef3c7"},
                            {"range": [80, 100], "color": "#dcfce7"},
                        ],
                    },
                )
            )
            fig_gauge.update_layout(template="plotly_white", height=360)
            st.plotly_chart(fig_gauge, use_container_width=True)

        with g2:
            st.metric("Category", score_category)
            st.metric("Score", f"{score_value}/100")

            emission_factor = 0.82
            co2_kg = predicted_units * emission_factor

            if score_value >= 80:
                target_factor = 0.90
            elif score_value >= 60:
                target_factor = 0.80
            else:
                target_factor = 0.65

            reduced_units = predicted_units * target_factor
            reduction_kg = max((predicted_units - reduced_units) * emission_factor, 0.0)

            st.metric("Estimated Monthly CO2", f"{co2_kg:.1f} kg")
            st.metric("Potential CO2 Reduction", f"{reduction_kg:.1f} kg/month")

    st.markdown(
        '<div class="footer">Built for Smart City Sustainable Energy Management</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
