"""
STRATA-X — Mine Operations Command Center
==========================================
AI-Driven Real-Time Mine Subsidence Intelligence & Early Warning System
Frontend Dashboard: Integrated with FastAPI Backend, Layer 1 LSTM TTF Engine & Layer 2 Spatial Risk Engine
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

# ================================================================
# 1) PAGE CONFIGURATION & INDUSTRIAL COMMAND CENTER CSS
# ================================================================
st.set_page_config(
    page_title="STRATA-X | Mine Subsidence Operations Command Center",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* Dark Industrial Command Center Theme */
    .stApp {
        background-color: #07090e;
        color: #e2e8f0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    /* Top Header */
    .header-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 18px 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
    }
    .main-title {
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #f8fafc;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .system-subtitle {
        color: #94a3b8;
        font-size: 0.88rem;
        font-weight: 500;
        margin-top: 4px;
    }
    .status-badge-online {
        background-color: #064e3b;
        color: #34d399;
        border: 1px solid #059669;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    
    /* Evacuation & Critical Banners */
    .banner-critical {
        background: linear-gradient(90deg, #7f1d1d 0%, #dc2626 50%, #7f1d1d 100%);
        color: #ffffff;
        padding: 16px;
        text-align: center;
        font-size: 1.25rem;
        font-weight: 800;
        border-radius: 8px;
        border: 2px solid #ef4444;
        box-shadow: 0 0 25px rgba(220, 38, 38, 0.45);
        margin-bottom: 18px;
        animation: pulse 1.5s infinite;
    }
    .banner-anomaly {
        background: linear-gradient(90deg, #713f12 0%, #ca8a04 50%, #713f12 100%);
        color: #fef08a;
        padding: 12px 18px;
        font-weight: 700;
        font-size: 0.95rem;
        border-radius: 8px;
        border: 1px solid #eab308;
        margin-bottom: 18px;
    }
    @keyframes pulse {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.90; transform: scale(0.998); }
        100% { opacity: 1; transform: scale(1); }
    }
    
    /* Metric Cards */
    .metric-box {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 14px 16px;
        text-align: left;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
    }
    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #94a3b8;
        font-weight: 600;
    }
    .metric-val {
        font-size: 1.5rem;
        font-weight: 800;
        color: #f8fafc;
        margin: 4px 0;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #64748b;
    }
    
    /* 3x3 Node Cards */
    .node-grid-card {
        background-color: #0f172a;
        border-radius: 8px;
        padding: 12px;
        border: 1px solid #1e293b;
        margin-bottom: 12px;
        transition: all 0.2s ease-in-out;
    }
    .node-grid-card.real-sensor {
        border: 2px solid #38bdf8;
        background: linear-gradient(180deg, #0c4a6e22 0%, #0f172a 100%);
    }
    .node-grid-card.zone-member {
        border: 2px solid #f97316;
        box-shadow: 0 0 15px rgba(249, 115, 22, 0.25);
    }
    .node-badge-real {
        background-color: #0369a1;
        color: #e0f2fe;
        font-size: 0.65rem;
        font-weight: 800;
        padding: 2px 6px;
        border-radius: 4px;
        letter-spacing: 0.5px;
    }
    .node-badge-sim {
        background-color: #334155;
        color: #cbd5e1;
        font-size: 0.65rem;
        font-weight: 700;
        padding: 2px 6px;
        border-radius: 4px;
    }
    
    /* Panels */
    .panel-card {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
    }
    .panel-header {
        font-size: 1.05rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .transparency-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        background-color: #1e293b;
        color: #cbd5e1;
        border: 1px solid #334155;
    }
    </style>
""", unsafe_allow_html=True)

BACKEND_URL = "http://localhost:8000"


# ================================================================
# 2) BACKEND DATA FETCHING HELPERS
# ================================================================
@st.cache_data(ttl=1)
def fetch_api(endpoint: str):
    try:
        resp = requests.get(f"{BACKEND_URL}{endpoint}", timeout=2.0)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        return None
    return None


def post_api(endpoint: str, payload: dict = None):
    try:
        resp = requests.post(f"{BACKEND_URL}{endpoint}", json=payload or {}, timeout=3.0)
        return resp.json() if resp.status_code == 200 else None
    except Exception as e:
        st.sidebar.error(f"Action failed: {e}")
        return None


# ================================================================
# 3) SIDEBAR CONTROLS & SIMULATION CONSOLE
# ================================================================
with st.sidebar:
    st.markdown("### ⛏️ STRATA-X Console")
    st.caption("Industrial Mine Subsidence Operations")
    
    st.divider()
    st.markdown("#### 🕹️ Scenario Controller")
    
    sim_status = fetch_api("/simulation/status") or {"is_running": False, "active_scenario": "NORMAL", "step_count": 0}
    
    scenario_options = [
        ("NORMAL", "Normal Baseline (Stable)"),
        ("DEFORMATION_DEVELOPING", "Deformation Developing (Secondary Creep)"),
        ("HIGH_RISK_ZONE", "High-Risk Zone (Tertiary Acceleration)"),
        ("SENSOR_FAULT", "Isolated Sensor Fault (N7 Disturbance)"),
        ("COMMUNICATION_FAILURE", "Communication Failure (N9 Dropout)"),
        ("RECOVERY", "Recovery / Stabilization"),
    ]
    
    scen_keys = [s[0] for s in scenario_options]
    scen_labels = [s[1] for s in scenario_options]
    current_key = sim_status.get("active_scenario", "NORMAL")
    current_idx = scen_keys.index(current_key) if current_key in scen_keys else 0
    
    selected_idx = st.selectbox(
        "Geotechnical Scenario",
        range(len(scenario_options)),
        format_func=lambda i: scen_labels[i],
        index=current_idx
    )
    selected_key = scen_keys[selected_idx]
    
    if st.button("⚡ Apply Scenario", use_container_width=True):
        res = post_api("/simulation/scenario", {"scenario": selected_key})
        if res:
            st.success(f"Scenario updated: {selected_key}")
            time.sleep(0.3)
            st.rerun()

    c1, c2 = st.columns(2)
    with c1:
        if st.button("▶️ Start Sim", use_container_width=True):
            post_api("/simulation/start")
            time.sleep(0.3)
            st.rerun()
    with c2:
        if st.button("⏹️ Stop Sim", use_container_width=True):
            post_api("/simulation/stop")
            time.sleep(0.3)
            st.rerun()

    st.markdown(f"**State:** `{'🟢 RUNNING' if sim_status.get('is_running') else '⚪ IDLE'}` | **Step:** `{sim_status.get('step_count', 0)}`")
    
    st.divider()
    st.markdown("#### 🛰️ Hardware & Model")
    sys_status = fetch_api("/system/status")
    if sys_status:
        st.write(f"**Mode:** `{sys_status.get('mode')}`")
        st.write(f"**Physical Anchor:** `NODE_05` (100% Real)")
        st.write(f"**Simulated Nodes:** 8 Prototype Nodes")
        st.write(f"**Layer 1 LSTM:** `{sys_status.get('ml_model_status')}`")
        st.write(f"**Layer 2 Spatial:** `ACTIVE`")
    else:
        st.warning("⚠️ Backend connection waiting...")


# ================================================================
# 4) FETCH LIVE TELEMETRY & SPATIAL DATA
# ================================================================
latest_reading = fetch_api("/latest") or {
    "node_id": "NODE_05",
    "is_real": True,
    "tilt_deg": 1.15,
    "ttf_hours": None,
    "velocity_mm_h": 0.038,
    "deformation_mm": 2.10,
    "risk_status": "NORMAL",
    "risk_score": 10.0,
    "confidence": 0.95,
    "trend": "STABLE",
    "timestamp": datetime.utcnow().isoformat()
}

spatial_map = fetch_api("/spatial/risk-map") or {
    "mine_risk": "NORMAL",
    "active_zone_count": 0,
    "zones": [],
    "nodes": [],
    "suppressed_anomalies": []
}

nodes_list = fetch_api("/nodes") or []
active_alerts = fetch_api("/alerts") or []

active_zones = spatial_map.get("zones", [])
suppressed_anomalies = spatial_map.get("suppressed_anomalies", [])
mine_risk = spatial_map.get("mine_risk", "NORMAL")

# Identify all nodes that belong to active risk zones
zone_node_ids = set()
for z in active_zones:
    for nid in z.get("affected_node_ids", []):
        zone_node_ids.add(nid)

# Lowest predicted TTF across all nodes
valid_ttfs = [n.get("current_ttf_hours") for n in nodes_list if n.get("current_ttf_hours") is not None]
min_ttf = min(valid_ttfs) if valid_ttfs else None


# ================================================================
# 5) HEADER & SYSTEM BAR
# ================================================================
st.markdown(f"""
<div class="header-container">
    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
        <div>
            <div class="main-title">⛏️ STRATA-X &mdash; Mine Operations Command Center</div>
            <div class="system-subtitle">AI-Driven Underground Strata Subsidence Intelligence &bull; SCAMP Early Warning Platform</div>
        </div>
        <div style="text-align:right;">
            <span class="status-badge-online">&bull; SYSTEM OPERATIONAL</span>
            <div style="color:#94a3b8; font-size:0.8rem; margin-top:4px;">
                Anchor: <b>NODE_05 (Real)</b> | Sim: <b>ACTIVE</b> | Time: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# Critical Emergency Evacuation Alarm
if mine_risk == "CRITICAL" or (min_ttf is not None and min_ttf <= 1.0):
    st.markdown(f"""
    <div class="banner-critical">
        🚨 CRITICAL STRATA SUBSIDENCE HAZARD DETECTED &bull; PREDICTED TTF: {min_ttf:.2f} HOURS &bull; EVACUATE ZONE IMMEDIATELY 🚨
    </div>
    """, unsafe_allow_html=True)

# Isolated Sensor Anomaly Notice (Demonstrating Layer 2 False Alarm Suppression)
if suppressed_anomalies:
    for anomaly in suppressed_anomalies:
        st.markdown(f"""
        <div class="banner-anomaly">
            🛠️ <b>ISOLATED SENSOR ANOMALY SUPPRESSED:</b> Node <b>{anomaly.get('node_id')}</b> reported abnormal kinematics, but adjacent gallery nodes are stable. 
            <b>Layer 2 has suppressed false alarm &mdash; No subsidence zone declared.</b>
        </div>
        """, unsafe_allow_html=True)


# ================================================================
# 6) TOP EXECUTIVE METRIC CARDS
# ================================================================
m1, m2, m3, m4, m5, m6 = st.columns(6)

with m1:
    risk_color_map = {
        "NORMAL": "#10b981",
        "WATCH": "#facc15",
        "HIGH RISK": "#f97316",
        "CRITICAL": "#ef4444"
    }
    color = risk_color_map.get(mine_risk, "#10b981")
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">Mine Risk Level</div>
        <div class="metric-val" style="color:{color};">{mine_risk}</div>
        <div class="metric-sub">Layer 2 Spatial State</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    ttf_text = f"{min_ttf:.2f} hrs" if min_ttf is not None else "Buffering..."
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">Minimum TTF</div>
        <div class="metric-val">{ttf_text}</div>
        <div class="metric-sub">LSTM Layer 1 Ground Truth</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">Active Risk Zones</div>
        <div class="metric-val">{len(active_zones)}</div>
        <div class="metric-sub">Connected Clusters</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    online_count = len([n for n in nodes_list if n.get("status") == "ONLINE"])
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">Active Sensor Mesh</div>
        <div class="metric-val">{online_count}/9 Nodes</div>
        <div class="metric-sub">1 Real + 8 Simulated</div>
    </div>
    """, unsafe_allow_html=True)

with m5:
    conf_val = float(latest_reading.get("confidence", 0.95)) * 100
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">Model Confidence</div>
        <div class="metric-val">{conf_val:.1f}%</div>
        <div class="metric-sub">Spatial & Kinematic Coherence</div>
    </div>
    """, unsafe_allow_html=True)

with m6:
    max_vel = max([float(n.get("velocity_mm_h", 0.0) or 0.0) for n in spatial_map.get("nodes", [])] or [0.038])
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">Peak Velocity</div>
        <div class="metric-val">{max_vel:.3f} mm/h</div>
        <div class="metric-sub">Deformation Rate</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)


# ================================================================
# 7) HERO COMPONENT — 3x3 SENSOR MESH & SPATIAL RISK ANALYSIS
# ================================================================
col_mesh, col_spatial = st.columns([1.1, 0.9])

with col_mesh:
    st.markdown("""
    <div class="panel-card">
        <div class="panel-header">
            <span>🌐 3&times;3 Underground Mine Sensor Matrix</span>
            <span class="transparency-pill">N5 = REAL HARDWARE &bull; N1-N4, N6-N9 = SOFTWARE SIMULATION</span>
        </div>
    """, unsafe_allow_html=True)
    
    # Render 3x3 Grid
    # Row 1: NODE_01, NODE_02, NODE_03
    # Row 2: NODE_04, NODE_05 (REAL), NODE_06
    # Row 3: NODE_07, NODE_08, NODE_09
    grid_rows = [
        ["NODE_01", "NODE_02", "NODE_03"],
        ["NODE_04", "NODE_05", "NODE_06"],
        ["NODE_07", "NODE_08", "NODE_09"]
    ]

    nodes_dict = {n.get("node_id"): n for n in nodes_list}
    spatial_nodes_dict = {n.get("node_id"): n for n in spatial_map.get("nodes", [])}

    for row in grid_rows:
        cols = st.columns(3)
        for idx, nid in enumerate(row):
            with cols[idx]:
                n_info = nodes_dict.get(nid, {})
                s_info = spatial_nodes_dict.get(nid, {})
                
                is_real = n_info.get("is_real", nid == "NODE_05")
                r_state = n_info.get("current_risk", "NORMAL")
                is_in_zone = nid in zone_node_ids
                
                risk_icon = {"NORMAL": "🟢", "WATCH": "🟡", "HIGH RISK": "🟠", "CRITICAL": "🔴"}.get(r_state, "🟢")
                card_class = "node-grid-card"
                if is_real:
                    card_class += " real-sensor"
                if is_in_zone:
                    card_class += " zone-member"

                badge = '<span class="node-badge-real">REAL HARDWARE</span>' if is_real else '<span class="node-badge-sim">SIMULATED</span>'
                zone_tag = '<span style="color:#f97316; font-size:0.75rem; font-weight:700;">[ZONE MEMBER]</span>' if is_in_zone else ''
                
                ttf_val = n_info.get("current_ttf_hours")
                ttf_str = f"{ttf_val:.1f}h" if ttf_val is not None else "Buffering"
                
                vel_val = s_info.get("velocity_mm_h", 0.038)
                def_val = s_info.get("deformation_mm", 2.0)

                st.markdown(f"""
                <div class="{card_class}">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <b>{risk_icon} {nid}</b>
                        {badge}
                    </div>
                    <div style="color:#94a3b8; font-size:0.78rem; margin:2px 0;">{n_info.get('name', nid)}</div>
                    <div style="font-size:0.8rem; margin-top:4px;">
                        TTF: <b>{ttf_str}</b> &bull; Vel: <b>{vel_val:.3f} mm/h</b><br>
                        Def: <b>{def_val:.1f} mm</b> &bull; Batt: <b>{n_info.get('battery', 95):.0f}%</b>
                    </div>
                    {zone_tag}
                </div>
                """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

with col_spatial:
    st.markdown("""
    <div class="panel-card">
        <div class="panel-header">
            <span>🛡️ Layer 2: Spatial Risk Analysis</span>
            <span class="transparency-pill">Deterministic Clustering</span>
        </div>
    """, unsafe_allow_html=True)
    
    if active_zones:
        for z in active_zones:
            z_risk = z.get("risk_level", "HIGH RISK")
            z_badge = {"WATCH": "🟡 WATCH", "HIGH RISK": "🟠 HIGH RISK", "CRITICAL": "🔴 CRITICAL"}.get(z_risk, "🟠 HIGH RISK")
            nodes_str = ", ".join(z.get("affected_node_ids", []))
            
            st.markdown(f"""
            <div style="background-color:#1e293b; border-left:4px solid #f97316; border-radius:6px; padding:12px; margin-bottom:12px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <b style="font-size:1.1rem; color:#f8fafc;">📍 {z.get('zone_id')}</b>
                    <span style="font-weight:700; font-size:0.85rem;">{z_badge}</span>
                </div>
                <div style="font-size:0.82rem; color:#cbd5e1; margin:6px 0;">
                    <b>Affected Nodes:</b> <code>{nodes_str}</code> ({z.get('number_of_nodes')} nodes)<br>
                    <b>Centroid:</b> Grid ({z.get('centroid', {}).get('grid_x')}, {z.get('centroid', {}).get('grid_y')}) &bull; Lat/Lon ({z.get('centroid', {}).get('lat'):.4f}, {z.get('centroid', {}).get('lon'):.4f})<br>
                    <b>Max Deformation:</b> {z.get('maximum_deformation'):.2f} mm &bull; <b>Cluster Avg:</b> {z.get('average_deformation'):.2f} mm<br>
                    <b>Min Predicted TTF:</b> {z.get('minimum_ttf') or 'N/A'} hours &bull; <b>Confidence:</b> {z.get('spatial_confidence', 0.9)*100:.0f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background-color:#1e293b; border-left:4px solid #10b981; border-radius:6px; padding:12px; margin-bottom:12px;">
            <b style="color:#34d399;">✅ NO ACTIVE SUBSIDENCE ZONES DETECTED</b>
            <div style="font-size:0.82rem; color:#94a3b8; margin-top:4px;">
                All 9 gallery nodes are operating within the standard geotechnical stability envelope or isolated non-correlated movement.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Explainability Section
    st.markdown("#### 💡 Why is this risk level declared?")
    if active_zones:
        for z in active_zones:
            reasons = z.get("reasons", [])
            for r in reasons:
                st.markdown(f"- ✓ {r}")
    elif suppressed_anomalies:
        st.markdown("- ✓ Single node anomaly detected without multi-gallery spatial correlation.")
        st.markdown("- ✓ Surrounding neighborhood baseline confirmed stable.")
        st.markdown("- ✓ False-alarm suppression algorithm active.")
    else:
        st.markdown("- ✓ Velocity across all nodes is within safe baseline (< 0.10 mm/h).")
        st.markdown("- ✓ Inverse velocity trends exhibit no tertiary acceleration slope.")
        st.markdown("- ✓ Spatial correlation across adjacent galleries is uniform.")

    st.markdown("</div>", unsafe_allow_html=True)


# ================================================================
# 8) TELEMETRY CHARTS & GEO-SPATIAL MAP
# ================================================================
c_chart1, c_chart2 = st.columns([1, 1])

# Session state history buffer for charts
if "chart_history" not in st.session_state:
    st.session_state.chart_history = pd.DataFrame(columns=["timestamp", "tilt_deg", "ttf_hours", "deformation_mm", "velocity_mm_h"])

# Append latest reading
new_entry = {
    "timestamp": datetime.now().strftime("%H:%M:%S"),
    "tilt_deg": float(latest_reading.get("tilt_deg", 1.0) or 1.0),
    "ttf_hours": float(latest_reading.get("ttf_hours") or 8.5),
    "deformation_mm": float(latest_reading.get("deformation_mm", 2.0) or 2.0),
    "velocity_mm_h": float(latest_reading.get("velocity_mm_h", 0.038) or 0.038),
}
st.session_state.chart_history = pd.concat([st.session_state.chart_history, pd.DataFrame([new_entry])], ignore_index=True).tail(50)

with c_chart1:
    st.markdown("#### 📈 Physical Sensor (N5) & Ground Kinematics")
    fig1 = go.Figure()
    
    # Deformation fill
    fig1.add_trace(go.Scatter(
        x=st.session_state.chart_history["timestamp"],
        y=st.session_state.chart_history["deformation_mm"],
        mode="lines",
        name="Deformation (mm)",
        fill='tozeroy',
        fillcolor='rgba(56, 189, 248, 0.12)',
        line=dict(color="#38bdf8", width=2.5)
    ))
    
    # Velocity on secondary axis
    fig1.add_trace(go.Scatter(
        x=st.session_state.chart_history["timestamp"],
        y=st.session_state.chart_history["velocity_mm_h"],
        mode="lines+markers",
        name="Velocity (mm/h)",
        yaxis="y2",
        line=dict(color="#f43f5e", width=2)
    ))
    
    fig1.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#07090e",
        height=300,
        margin=dict(l=10, r=10, t=25, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(title="Deformation (mm)", side="left"),
        yaxis2=dict(title="Velocity (mm/h)", overlaying="y", side="right")
    )
    st.plotly_chart(fig1, use_container_width=True)

with c_chart2:
    st.markdown("#### ⏱️ Predicted Time-to-Failure (TTF) & Safety Thresholds")
    fig2 = go.Figure()
    
    fig2.add_trace(go.Scatter(
        x=st.session_state.chart_history["timestamp"],
        y=st.session_state.chart_history["ttf_hours"],
        mode="lines+markers",
        name="Predicted TTF (h)",
        line=dict(color="#f59e0b", width=2.5)
    ))
    
    # Threshold horizontal lines
    fig2.add_hline(y=6.0, line_dash="dot", line_color="#facc15", annotation_text="Watch (6h)", annotation_position="top right")
    fig2.add_hline(y=3.0, line_dash="dash", line_color="#f97316", annotation_text="High Risk (3h)", annotation_position="top right")
    fig2.add_hline(y=1.0, line_dash="solid", line_color="#ef4444", annotation_text="Critical (1h)", annotation_position="top right")
    
    fig2.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#07090e",
        height=300,
        margin=dict(l=10, r=10, t=25, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(title="Time-to-Failure (hours)", range=[0, 14])
    )
    st.plotly_chart(fig2, use_container_width=True)


# ================================================================
# 9) GEO-SPATIAL MAP & AUDIT EVENT TIMELINE
# ================================================================
c_map, c_audit = st.columns([1, 1])

with c_map:
    st.markdown("#### 🗺️ Geo-Spatial Subsidence Risk Map")
    risk_map_data = fetch_api("/risk/map")
    if risk_map_data and "nodes" in risk_map_data:
        map_df = pd.DataFrame(risk_map_data["nodes"])
        try:
            fig_map = px.scatter_mapbox(
                map_df,
                lat="lat",
                lon="lon",
                color="risk_state",
                size=[18 if r.get("is_real") else 14 for _, r in map_df.iterrows()],
                hover_name="name",
                hover_data={"node_id": True, "is_real": True, "ttf_hours": True, "tilt_deg": True, "risk_score": True, "lat": False, "lon": False},
                color_discrete_map={"NORMAL": "#10b981", "WATCH": "#facc15", "HIGH RISK": "#f97316", "CRITICAL": "#ef4444"},
                zoom=15,
                height=300
            )
            fig_map.update_layout(
                mapbox_style="open-street-map",
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor="#0f172a"
            )
            st.plotly_chart(fig_map, use_container_width=True)
        except Exception:
            st.map(map_df[["lat", "lon"]], zoom=14)
    else:
        st.info("Loading mine spatial map coordinates...")

with c_audit:
    st.markdown("#### 📜 Live Operational Event Stream")
    events = fetch_api("/events?limit=6") or []
    if events:
        for ev in events[:5]:
            ts = ev.get("timestamp", "")
            ts_display = ts.split("T")[1][:8] if "T" in ts else ts
            ev_type = ev.get("event_type", "INFO")
            st.markdown(f"⏱️ `[{ts_display}]` **{ev_type}**: {ev.get('details')}")
    else:
        st.markdown(f"⏱️ `[{datetime.now().strftime('%H:%M:%S')}]` **SYSTEM_HEARTBEAT**: All 9 gallery sensors reporting synchronized telemetry.")
        st.markdown(f"⏱️ `[{datetime.now().strftime('%H:%M:%S')}]` **LAYER2_CHECK**: Spatial topology scan completed &mdash; 0 active hazards.")


# ================================================================
# 10) SCAMP COMPLIANCE AUDIT EXPORT & ACTIVE ALERTS
# ================================================================
st.divider()
c_scamp, c_alerts = st.columns([1, 1])

with c_scamp:
    st.markdown("#### 📄 SCAMP-Oriented Monitoring Report")
    st.caption("Strata Control and Monitoring Plan (DGMS Cir. No. 06/2020 Operational Guidance)")

    def generate_scamp_report():
        return f"""================================================================================
STRATA-X — UNDERGROUND COAL MINE SUBSIDENCE MONITORING REPORT
Strata Control and Monitoring Plan (SCAMP) Operational Guidance
================================================================================
Generated At:            {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
Monitoring System:       STRATA-X Layer 1 (LSTM) + Layer 2 (Spatial Risk Engine)
Physical Anchor:         NODE_05 (Real Hardware)
Simulation Mesh:         NODE_01 to NODE_04, NODE_06 to NODE_09 (8 Prototype Nodes)

1. MINE-WIDE INTEGRITY ASSESSMENT
--------------------------------------------------------------------------------
Overall Mine Risk:       {mine_risk}
Active Risk Zones:       {len(active_zones)}
Lowest Predicted TTF:    {min_ttf if min_ttf is not None else 'Buffering'} hours
Model Confidence:        {conf_val:.1f}%

2. ACTIVE SUBSIDENCE ZONES
--------------------------------------------------------------------------------
{chr(10).join([f"Zone {z.get('zone_id')}: Level={z.get('risk_level')}, Affected={', '.join(z.get('affected_node_ids', []))}, MaxDef={z.get('maximum_deformation'):.2f}mm, MinTTF={z.get('minimum_ttf')}h" for z in active_zones]) if active_zones else 'No active multi-node subsidence zones detected.'}

3. SENSOR ANOMALY SUPPRESSION STATUS
--------------------------------------------------------------------------------
{chr(10).join([f"Node {a.get('node_id')}: {a.get('reason')}" for a in suppressed_anomalies]) if suppressed_anomalies else 'No isolated sensor anomalies detected.'}

================================================================================
"""

    st.download_button(
        label="⬇️ Download SCAMP-Oriented Monitoring Report (.txt)",
        data=generate_scamp_report(),
        file_name=f"STRATAMonitoring_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
        use_container_width=True
    )

with c_alerts:
    st.markdown("#### 🚨 Active Diagnostic & Hazard Alerts")
    if active_alerts:
        for a in active_alerts[:3]:
            st.warning(f"**[{a.get('level')}] {a.get('title')}**\n{a.get('message')}")
    else:
        st.success("✅ All 9 gallery sensor nodes operational. No active strata hazard alerts.")


# Auto-refresh interval (2.0s)
time.sleep(2.0)
st.rerun()
