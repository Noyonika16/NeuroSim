"""
NEURO-ELEGANS: Connectome-inspired C. elegans model and research dashboard.
The browser game uses simplified neuron-state dynamics, synthetic sensory encoding,
and hand-designed action decoders; its behavior is experimental and not biologically validated.
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import json
import os
import sys

# Ensure local imports work
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from connectome.data_loader import Connectome
from connectome.biophysical_engine import BiophysicalConnectomeEngine
from connectome.graph_analytics import ConnectomeGraphAnalytics
from connectome.ablation_optogenetics import ExperimentalLab
from benchmark.comparative_study import ComparativeStudyBenchmark
from web.embed_helper import get_inlined_html

# Page configuration
st.set_page_config(
    page_title="NEURO-ELEGANS | Connectome Engine",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    :root {
        color-scheme: dark;
        --ne-bg: #0b1220;
        --ne-panel: #111c30;
        --ne-panel-2: #15233a;
        --ne-border: #293b58;
        --ne-text: #e6edf7;
        --ne-muted: #a5b4cc;
        --ne-cyan: #67e8f9;
        --ne-blue: #7aaeff;
        --ne-green: #6ee7b7;
        --primary-color: #22d3ee;
        --text-color: #e6edf7;
        --background-color: #0b1220;
        --secondary-background-color: #111c30;
    }
    html, body, .stApp, [data-testid="stAppViewContainer"],
    [data-testid="stHeader"], [data-testid="stMain"] {
        background-color: var(--ne-bg);
        color: var(--ne-text);
    }
    /* Streamlit widgets must keep readable labels on the navy background. */
    .stApp p, .stApp li, .stApp label, .stApp legend,
    .stApp [data-testid="stMarkdownContainer"],
    .stApp [data-testid="stWidgetLabel"] p,
    .stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stRadio"] label,
    .stApp [data-testid="stCheckbox"] label,
    .stApp [data-testid="stExpander"] summary,
    .stApp [data-testid="stTabs"] button,
    .stApp [data-testid="stDataFrame"] th,
    .stApp [data-testid="stTable"] th {
        color: #e6edf7 !important;
    }
    .stApp [data-testid="stCaptionContainer"] { color: #b8c6dc !important; }
    .stApp [data-testid="stWidgetLabel"] { color: #e6edf7 !important; }
    .stApp [data-testid="stSidebar"] { background-color: #0e192b; }
    .stApp [data-testid="stSidebar"] * { color: #e6edf7; }
    /* Keep form fields light with dark text for familiar, high-contrast controls. */
    .stApp input, .stApp textarea,
    .stApp [data-baseweb="select"] input,
    .stApp [data-baseweb="select"] [role="combobox"] {
        color: #172033 !important;
        -webkit-text-fill-color: #172033 !important;
    }
    .stApp [data-baseweb="select"] > div { border-color: #94a3b8 !important; }
    .stApp [data-baseweb="tag"] { background: #2563eb !important; color: #ffffff !important; }
    .stApp [data-baseweb="tag"] *, .stApp [data-baseweb="tag"] svg { color: #ffffff !important; fill: #ffffff !important; }
    .stApp button[kind="primary"], .stApp [data-testid="stFormSubmitButton"] button {
        background: #0891b2 !important; color: #ffffff !important;
        border: 1px solid #67e8f9 !important; font-weight: 800 !important;
    }
    .stApp button[kind="secondary"] { color: #e6edf7 !important; border-color: #475569 !important; }
    .stApp [data-testid="stSlider"] [data-testid="stTickBarMin"],
    .stApp [data-testid="stSlider"] [data-testid="stTickBarMax"] { color: #cbd5e1 !important; }
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(21,35,58,.94), rgba(15,27,46,.96));
        border: 1px solid var(--ne-border);
        padding: 12px 14px;
        border-radius: 12px;
    }
    .metric-card {
        background: linear-gradient(145deg, rgba(21,35,58,.96), rgba(15,27,46,.96));
        border: 1px solid #293b58;
        padding: 16px 20px;
        border-radius: 10px;
        box-shadow: 0 8px 24px rgba(2, 8, 23, 0.22);
    }
    .stApp [data-testid="stMarkdownContainer"] p,
    .stApp [data-testid="stMarkdownContainer"] li {
        font-size: 15px;
        line-height: 1.62;
    }
    .metric-val {
        font-size: clamp(28px, 2.1vw, 38px);
        line-height: 1.12;
        font-weight: 900;
        color: #67e8f9 !important;
        -webkit-text-fill-color: #67e8f9 !important;
        font-variant-numeric: tabular-nums;
        text-shadow: 0 1px 10px rgba(34,211,238,.12);
    }
    .metric-label {
        font-size: 12px;
        line-height: 1.45;
        font-weight: 750;
        text-transform: uppercase;
        letter-spacing: .55px;
        color: #cbd5e1 !important;
        -webkit-text-fill-color: #cbd5e1 !important;
        margin-top: 7px;
    }
    .stApp [data-testid="stMetricValue"] { color: #67e8f9 !important; font-weight: 900 !important; }
    .stApp [data-testid="stMetricLabel"] { color: #cbd5e1 !important; font-weight: 700 !important; }
    .stApp [data-testid="stThumbValue"] { color: #67e8f9 !important; font-weight: 800 !important; }
    .stApp [data-testid="stSlider"] label { color: #e6edf7 !important; font-weight: 700 !important; }
    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_connectome_and_analytics():
    c = Connectome()
    analytics = ConnectomeGraphAnalytics(c)
    return c, analytics

connectome, analytics = load_connectome_and_analytics()

# Compute the small-world statistic once and reuse the exact result in both KPI
# locations. A fixed seed makes the ER null baseline reproducible across reruns.
summary_stats = analytics.get_summary_stats()

@st.cache_data(show_spinner=False)
def cached_small_world_stats(graph_signature, n_random_samples=20, seed=20261027):
    # graph_signature is part of the cache key, so changing connectome topology
    # invalidates the cached metric instead of reusing a stale KPI.
    return analytics.compute_small_worldness(n_random_samples=n_random_samples, seed=seed)

_graph_signature = (
    tuple(sorted(connectome.neurons.keys())),
    tuple(sorted(tuple(sorted((edge["n1"], edge["n2"]))) for edge in connectome.gap_junctions)),
    tuple(sorted(tuple(sorted((edge["pre"], edge["post"]))) for edge in connectome.chemical_synapses)),
)
try:
    small_world_stats = cached_small_world_stats(_graph_signature, 20, 20261027)
    small_world_display = f"{small_world_stats['sigma (small_worldness_index)']:.2f}"
    small_world_error = None
except (ValueError, RuntimeError, nx.NetworkXException) as exc:
    # Keep the dashboard usable if a graph or null-model calculation is invalid;
    # never replace a failed calculation with a hard-coded value.
    small_world_stats = None
    small_world_display = "N/A"
    small_world_error = str(exc)

# Header & Hero Banner
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("<h1 style='margin-bottom:4px;background:linear-gradient(90deg,#00f2fe,#4facfe,#00ff87);-webkit-background-clip:text;-webkit-text-fill-color:transparent;'>🧬 NEURO-ELEGANS: Synaptic Connectome Neuro-Engine</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:15px;color:#b8c6dc;margin-bottom:12px;'><i>Exploring closed-loop control with a connectome-inspired model of C. elegans.</i> Neuron-only and trained-hybrid game scores are computational outcomes, not evidence that a living worm plays Flappy Bird.</p>", unsafe_allow_html=True)
with col_h2:
    st.markdown("""
    <div style='text-align:right;padding-top:10px;'>
        <span class='badge-pill' style='background:rgba(0,242,254,0.15);color:#00f2fe;border:1px solid #00f2fe;'>HERMAPHRODITE</span>
        <span class='badge-pill' style='background:rgba(0,255,135,0.15);color:#00ff87;border:1px solid #00ff87;'>WHITE ET AL. 1986</span>
    </div>
    """, unsafe_allow_html=True)

# Global Telemetry Bar
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.markdown(f"<div class='metric-card'><div class='metric-val'>{len(connectome.neurons)}</div><div class='metric-label'>Modeled Neuron States</div></div>", unsafe_allow_html=True)
with m2:
    st.markdown(f"<div class='metric-card'><div class='metric-val'>{len(connectome.chemical_synapses):,}</div><div class='metric-label'>Chemical Synapses</div></div>", unsafe_allow_html=True)
with m3:
    st.markdown(f"<div class='metric-card'><div class='metric-val'>{len(connectome.gap_junctions):,}</div><div class='metric-label'>Electrical Gap Junctions</div></div>", unsafe_allow_html=True)
with m4:
    st.markdown("<div class='metric-card'><div class='metric-val'>0</div><div class='metric-label'>Backprop-Trained Connectome Weights</div></div>", unsafe_allow_html=True)
with m5:
    st.markdown(f"<div class='metric-card'><div class='metric-val'>{small_world_display}</div><div class='metric-label'>Small-World Index (σ)</div></div>", unsafe_allow_html=True)

st.caption("Connectome synaptic weights are frozen and not backpropagation-trained. Separate supervised action readouts are trained offline to imitate an assisted policy; this is not a zero-shot controller. The small-world KPI uses the documented, seeded graph calculation shown in Graph Theory.")
if small_world_error:
    st.warning(f"Small-world index unavailable: {small_world_error}. No fallback value is displayed.")

st.write("")

# Navigation Tabs
tabs = st.tabs([
    "🕹️ 3D Connectome & Retro Games",
    "📊 Graph Theory & Topology",
    "⚡ Biophysical Electrophysiology",
    "🔬 Laser Ablation & Optogenetics",
    "🧪 Model vs. Deep RL (Caveats)",
    "📖 Biological Connectome Atlas"
])

# ==========================================
# TAB 1: 3D CONNECTOME & RETRO GAMES
# ==========================================
with tabs[0]:
    st.markdown("### 🎮 Interactive 3D WebGL Connectome & Closed-Loop Game Arena")
    st.markdown("The browser runs a closed-loop, connectome-inspired sensorimotor model. Engineered sensory encoders inject obstacle- or concentration-related currents; the neural state propagates through frozen connectome weights; a selected motor readout then chooses the game action. The calibrated readouts were trained offline to imitate an assisted policy, so this mode is not zero-shot biological behavior.")
    
    # Embed the browser-based Three.js simulation
    inlined_html = get_inlined_html()
    components.html(inlined_html, height=750, scrolling=False)
    
    st.info("💡 **Controls:** Orbit the 3D brain with Left-Click drag. Zoom with scroll. Click any 3D neuron sphere to Laser-Ablate it live during gameplay. Toggle between **Flappy Bird** and **Chemotaxis** retro tasks in the sidebar panel!")

# ==========================================
# TAB 2: GRAPH THEORY & TOPOLOGY
# ==========================================
with tabs[1]:
    st.markdown("### 📊 Network Topology & Graph Statistics")
    st.markdown("How does the undirected union of chemical connections and electrical gap junctions compare with a random-graph baseline under a documented small-world statistic?")

    stats = summary_stats
    sw = small_world_stats
    if sw is None:
        st.error("Small-worldness could not be computed from the current graph and null model. Other graph metrics remain available.")
    else:
        st.caption(
            f"Definition: σ = (C/C₀)/(L/L₀), calculated on the largest connected component "
            f"({sw['largest_component_nodes']} nodes, {sw['largest_component_edges']} edges). "
            f"Baseline: mean of {sw['random_samples']} connected Erdős–Rényi G(n,p) graphs "
            f"matched to its edge density; seed {sw['random_seed']}. The value is structural, not a gameplay or biological-fidelity metric."
        )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Graph Density", f"{stats['density'] * 100:.2f}%")
        st.metric("Avg Path Length (L)", f"{stats['avg_shortest_path_length']:.2f}")
    with c2:
        st.metric("Clustering Coeff (C)", f"{stats['avg_clustering_coefficient']:.3f}")
        st.metric("Graph Diameter", f"{stats['diameter']} hops")
    with c3:
        st.metric("Clustering Ratio (γ)", f"{sw['gamma (clustering_ratio)']:.2f}x" if sw else "N/A")
        st.metric("Path Ratio (λ)", f"{sw['lambda (path_ratio)']:.2f}x" if sw else "N/A")
    with c4:
        st.metric("Small-World Index (σ)", f"{sw['sigma (small_worldness_index)']:.2f}" if sw else "N/A")
        if sw:
            st.info("σ > 1 indicates greater clustering relative to path length under this specific random-graph baseline; it does not establish functional advantage.")

    st.write("---")
    
    # 3D Anatomical Spatial Distribution Plot
    st.markdown("#### 🧬 3D Anatomical Morphological Coordinates of 302 Neurons")
    neuron_rows = []
    for name, data in connectome.neurons.items():
        neuron_rows.append({
            "Neuron": name,
            "Type": data["type"].capitalize(),
            "Transmitter": data["transmitter"],
            "X (L-R um)": data["x"],
            "Y (D-V um)": data["y"],
            "Z (A-P um)": data["z"]
        })
    df_coords = pd.DataFrame(neuron_rows)

    fig_3d = px.scatter_3d(
        df_coords,
        x="Z (A-P um)",
        y="X (L-R um)",
        z="Y (D-V um)",
        color="Type",
        hover_name="Neuron",
        hover_data=["Transmitter"],
        color_discrete_map={
            "Sensory": "#00f2fe",
            "Command": "#f857a6",
            "Motor": "#00ff87",
            "Interneuron": "#ffb300"
        },
        title="3D Morphological Mapping of C. elegans Hermaphrodite Nervous System"
    )
    fig_3d.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b1220",
        plot_bgcolor="#111c30",
        font_color="#e6edf7",
        scene=dict(
            xaxis_title="Anterior-Posterior (z um)",
            yaxis_title="Left-Right (x um)",
            zaxis_title="Dorsal-Ventral (y um)",
            aspectmode='manual',
            aspectratio=dict(x=3.5, y=0.8, z=0.8)
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        height=520
    )
    st.plotly_chart(fig_3d, use_container_width=True)

    # Centrality Hub Analysis
    st.markdown("#### 👑 Centrality Hubs: Uncovering the Master Command Neurons")
    centrality = analytics.compute_centrality_rankings(top_k=12)
    
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        df_btw = pd.DataFrame(centrality["top_betweenness"], columns=["Neuron", "Betweenness"])
        fig_btw = px.bar(df_btw, x="Neuron", y="Betweenness", color="Betweenness", color_continuous_scale="Purples", title="Top Betweenness Centrality (Information Bottlenecks)")
        fig_btw.update_layout(template="plotly_dark", paper_bgcolor="#0b1220", plot_bgcolor="#111c30", font_color="#e6edf7", height=340)
        st.plotly_chart(fig_btw, use_container_width=True)
        st.caption("Neurons with highest betweenness (`AVA`, `AVB`, `AIB`, `RIBL`) act as critical switches coordinating motor behavior.")

    with col_g2:
        df_deg = pd.DataFrame(centrality["top_degree"], columns=["Neuron", "Degree Centrality"])
        fig_deg = px.bar(df_deg, x="Neuron", y="Degree Centrality", color="Degree Centrality", color_continuous_scale="Teal", title="Top Degree Centrality (Most Connected Hubs)")
        fig_deg.update_layout(template="plotly_dark", paper_bgcolor="#0b1220", plot_bgcolor="#111c30", font_color="#e6edf7", height=340)
        st.plotly_chart(fig_deg, use_container_width=True)
        st.caption("Command interneurons connect directly to large pools of ventral cord motor neurons.")

# ==========================================
# TAB 3: BIOPHYSICAL ELECTROPHYSIOLOGY
# ==========================================
with tabs[2]:
    st.markdown("### ⚡ Membrane-Potential Numerical Solver")
    st.markdown(r"""
    This solver uses **graded membrane potentials**, a conductance-like chemical synapse approximation, gap-junction coupling, and a phenomenological voltage-dependent calcium-like inward current. The current is a smooth sigmoid of voltage: the implementation does **not** include an intracellular calcium state, explicit channel gating kinetics, or validated plateau hysteresis/duration. Treat it as an exploratory approximation, not a validated calcium-plateau model. The equation is:
    $$\tau \frac{dV_i}{dt} = -(V_i-E_L) + \sum_j g^{gap}_{ij}(V_j-V_i) + \sum_j g^{chem}_{ji}s_j(E^{syn}_{ji}-V_i) + I_{Ca,i}+I_{sensory,i}+I_{opto,i}+\eta_i$$

    Here the code uses a **2 ms midpoint RK2 step**, a **30 ms time constant**, and membrane voltage in **mV**. Sensory/opto/noise terms are model-scaled voltage-equivalent inputs, not calibrated physical currents. The calcium-like term is `g_Ca * sigmoid((V - theta_Ca)/sigma_Ca) * (E_Ca - V)`; it has no explicit intracellular calcium state. Voltages are clipped to the model bounds [-80, 25] mV.
    """)

    st.write("---")
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        target_sensory = st.selectbox("Sensory Neuron to Stimulate", ["ALML (Anterior Touch)", "ASHL (Nociceptive Avoidance)", "PLML (Posterior Touch)", "ASEL (Chemotaxis ON)", "AWCL (Olfactory Attractant)"])
        sensory_name = target_sensory.split(" ")[0]
    with col_p2:
        current_stim = st.slider("Injected current (model mV-equivalent units)", min_value=0.0, max_value=40.0, value=25.0, step=1.0)
    with col_p3:
        sim_duration_ms = st.slider("Simulation Duration (ms)", min_value=100, max_value=600, value=300, step=50)

    if st.button("🚀 Run Biophysical Simulation Protocol"):
        engine = BiophysicalConnectomeEngine(connectome, seed=20261027)
        dt_ms = engine.dt * 1000.0
        n_steps = int(sim_duration_ms / dt_ms)
        
        # Track traces
        trace_neurons = [sensory_name, "AVAL", "AVAR", "AVBL", "AVBR", "DB1", "VB1"]
        time_points = []
        voltage_history = {n: [] for n in trace_neurons}

        for step in range(n_steps):
            t_ms = step * dt_ms
            time_points.append(t_ms)
            
            # Pulse stimulus between 40ms and 180ms
            if 40.0 <= t_ms <= 180.0:
                engine.set_sensory_input(sensory_name, current_stim)
            else:
                engine.set_sensory_input(sensory_name, 0.0)
                
            engine.step()
            
            for n in trace_neurons:
                voltage_history[n].append(engine.get_neuron_voltage(n))

        # Plot multi-channel traces
        fig_traces = go.Figure()
        colors = {"AVAL": "#ff3366", "AVAR": "#ff5588", "AVBL": "#00ff87", "AVBR": "#55ffaa", "DB1": "#ffb300", "VB1": "#9b51e0"}
        colors[sensory_name] = "#00f2fe"

        for n in trace_neurons:
            fig_traces.add_trace(go.Scatter(
                x=time_points,
                y=voltage_history[n],
                mode="lines",
                name=f"{n} ({connectome.neurons[n]['type'].capitalize()})",
                line=dict(color=colors.get(n, "#ffffff"), width=2)
            ))

        # Stimulus window highlight
        fig_traces.add_vrect(x0=40, x1=180, fillcolor="rgba(0, 242, 254, 0.12)", layer="below", line_width=0, annotation_text=f"Stimulus: {sensory_name}", annotation_position="top left")

        fig_traces.update_layout(
            title=f"Electrophysiological Voltage Traces $V_m(t)$ Under {sensory_name} Stimulation",
            xaxis_title="Time (milliseconds)",
            yaxis_title="Membrane Potential (mV)",
            template="plotly_dark",
            height=460
        )
        st.plotly_chart(fig_traces, use_container_width=True)

        # Quantitative Motor Summary
        ava_peak = max(voltage_history["AVAL"])
        avb_peak = max(voltage_history["AVBL"])
        st.info(f"**Observed simulated voltage output:** peak AVAL = `{ava_peak:.2f} mV`; peak AVBL = `{avb_peak:.2f} mV`. These are voltages from this numerical model; a game action or biological escape reflex is not implied by the voltage peaks alone.")

# ==========================================
# TAB 4: LASER ABLATION & OPTOGENETICS
# ==========================================
with tabs[3]:
    st.markdown("### 🔬 In-Silico Laser Ablation & Optogenetic Studio")
    st.markdown("Test functional neurobiology hypotheses by silencing specific neurons with a virtual laser or activating them with blue-light Channelrhodopsin (ChR2).")

    col_ab1, col_ab2 = st.columns([1, 2])
    with col_ab1:
        st.markdown("#### Select Circuit Perturbation")
        ablation_choice = st.radio(
            "Target Circuit to Invalidate:",
            [
                "Control (Intact 302 Connectome)",
                "Ablate Reversal Command Pair (AVA)",
                "Ablate Forward Command Pair (AVB)",
                "Ablate GABAergic Cross-Inhibition (DD/VD)",
                "Ablate Chemosensory Amphids (ASEL/ASER)"
            ]
        )

        st.markdown("#### Optogenetic Stimulation")
        opto_choice = st.selectbox(
            "Shine Light Pulse on Target:",
            ["None", "Blue Light 470nm (ChR2 on AVA)", "Blue Light 470nm (ChR2 on AVB)", "Yellow Light 590nm (NpHR on ASH)"]
        )

    with col_ab2:
        st.markdown("#### Model intervention interpretation")
        st.caption("Qualitative circuit hypothesis only; this panel does not execute or validate the browser-game behavior for the selected intervention.")
        if ablation_choice == "Control (Intact 302 Connectome)":
            st.success("**Control condition:** no neurons are intentionally silenced. Use this as the intact model reference condition.")
        elif ablation_choice == "Ablate Reversal Command Pair (AVA)":
            st.warning("**Hypothesis:** silencing the AVA command pair may alter reversal-related dynamics. The direction and magnitude of any game effect must be measured in matched neural-only runs.")
        elif ablation_choice == "Ablate Forward Command Pair (AVB)":
            st.warning("**Hypothesis:** silencing the AVB command pair may alter forward-state dynamics. Validate the effect with repeated, matched runs.")
        elif ablation_choice == "Ablate GABAergic Cross-Inhibition (DD/VD)":
            st.warning("**Hypothesis:** silencing DD/VD neurons may alter modeled motor coordination. The browser model does not simulate detailed body-wall muscle mechanics.")
        elif ablation_choice == "Ablate Chemosensory Amphids (ASEL/ASER)":
            st.warning("**Hypothesis:** silencing ASEL/ASER removes the model's primary concentration-change inputs. Measure the resulting navigation change rather than assuming a specific phenotype.")

        if opto_choice == "Blue Light 470nm (ChR2 on AVA)":
            st.info("**Modeled intervention:** a depolarizing current is intended to perturb AVA voltage. A downstream motor or behavioral response is not guaranteed and must be recorded.")
        elif opto_choice == "Blue Light 470nm (ChR2 on AVB)":
            st.info("**Modeled intervention:** a depolarizing current is intended to perturb AVB voltage. Measure motor and behavioral outcomes to determine the effect.")
        elif opto_choice == "Yellow Light 590nm (NpHR on ASH)":
            st.info("**Modeled intervention:** this option describes a proposed hyperpolarizing ASH perturbation; do not interpret it as a validated photophysical opsin simulation or guaranteed behavioral phenotype.")

# ==========================================
# TAB 5: MODEL COMPONENTS & TRAINING STATUS
# ==========================================
with tabs[4]:
    st.markdown("### 🧪 Model components, training status & fair comparison")
    st.warning("The number of neurons is not comparable to a machine-learning parameter count. This project has a frozen connectome core and two separately trained output readouts; it does not establish superiority over deep RL or large language models.")

    component_data = {
        "Component": [
            "Neuron-state model",
            "Chemical connectome",
            "Electrical gap-junction graph",
            "Flappy action readout",
            "Chemotaxis turn readout",
            "Neuron-only Flappy decoder",
            "Neuron-only Chemotaxis decoder",
        ],
        "Size / representation": [
            f"{len(connectome.neurons)} modeled voltage states",
            f"{len(connectome.chemical_synapses):,} chemical synapse records",
            f"{len(connectome.gap_junctions):,} gap-junction records",
            "99 voltage features",
            "105 voltage features",
            "DB − VB population contrast",
            "Lateral command / motor asymmetry",
        ],
        "Training status": [
            "Numerical dynamics; no gradient training",
            "Frozen imported weights",
            "Frozen imported connections",
            "Supervised logistic readout; imitates assisted labels",
            "Ridge regression; imitates assisted turn labels",
            "Untrained hand-specified threshold",
            "Untrained hand-specified readout",
        ],
        "Uses game target at runtime?": [
            "No direct target action",
            "No",
            "No",
            "No; uses simulated neuron voltages",
            "No; uses simulated neuron voltages",
            "No; uses simulated motor output",
            "No; uses simulated neuron voltages",
        ],
    }
    st.dataframe(pd.DataFrame(component_data), use_container_width=True, hide_index=True)

    st.markdown("#### What this does—and does not—show")
    st.markdown("""
    - **Supported:** the default Neuron-only browser mode derives actions from simulated neuron-state readouts without trained decoders or game-aware action overrides; the trained hybrid is separate.
    - **Not established:** that connectome topology itself causes better game performance. That requires matched, multi-seed tests against shuffled-weight and random-topology controls.
    - **Not measured:** biological metabolic power, GPU inference energy, or comparable training FLOPs. No energy-efficiency claim is made.
    - **Interpretation:** the assisted controller is a direct-game-state reference policy; the trained hybrid uses imitation-trained neural readouts; Neuron-only mode is an untrained baseline and may fail; its sensory transduction and motor-to-game mapping remain engineered abstractions. Failure is reported, not hidden by silently switching policies.
    """)

# ==========================================
# TAB 6: CONNECTOME ATLAS
# ==========================================
with tabs[5]:
    st.markdown("### 📖 Biological Connectome Atlas & Searchable Directory")
    st.markdown("Explore all 302 hermaphrodite neurons mapped by White et al. (1986), Varshney et al. (2011), and Cook et al. (2019).")

    f1, f2 = st.columns(2)
    with f1:
        type_filter = st.multiselect("Filter by Functional Class:", ["Sensory", "Command", "Motor", "Interneuron"], default=["Sensory", "Command", "Motor", "Interneuron"])
    with f2:
        search_query = st.text_input("Search Neuron by Name (e.g. AVA, ASH, ASEL, DB, DD):", "")

    filtered_df = df_coords[df_coords["Type"].isin(type_filter)]
    if search_query:
        filtered_df = filtered_df[filtered_df["Neuron"].str.contains(search_query.upper())]

    st.dataframe(filtered_df, use_container_width=True, height=450)
    st.caption(f"Showing {len(filtered_df)} of 302 neurons.")
