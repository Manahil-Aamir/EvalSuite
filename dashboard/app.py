import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import httpx
import uuid
import json
import os
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page Config
st.set_page_config(page_title="EvalSuite", layout="wide")

# STRICT COLOR CONSTANTS
GRAPHITE = "#292929"
CARD_BG = "#3A3A3A"
CARD_BORDER = "#4A4A4A"
WHITE = "#FFFFFF"
LIGHT_GRAY = "#CCCCCC"
NEON_GRASS = "#87FF53"
RACING_RED = "#F11115"
POWDER_BLUE = "#9CB5DD"
LAVENDER_VEIL = "#E2D3F0"
TECH_BLUE = "#3965A7"
WISTERIA = "#B995DB"

# CATEGORY COLORS
CATEGORY_COLORS = {
    "happy_path": POWDER_BLUE,
    "boundary": LAVENDER_VEIL,
    "edge_case": TECH_BLUE,
    "format_stress": WISTERIA,
    "prompt_injection": POWDER_BLUE,
    "jailbreak": LAVENDER_VEIL,
    "scope_attack": TECH_BLUE,
    "social_engineering": WISTERIA
}

# Pretty dimensions display mapping
pretty_dimensions = {
    "scope_adherence": "Scope Adherence",
    "attack_resistance": "Attack Resistance",
    "output_quality": "Output Quality",
    "consistency": "Consistency",
    "edge_case_handling": "Edge Case Handling"
}

def get_sample_data():
    report = {
        "pass_rate": 40,
        "total_tests": 40,
        "critical_failures_count": 24,
        "weakest_dimension": "attack_resistance",
        "average_scores": {
            "scope_adherence": 6.77,
            "attack_resistance": 2.06,
            "output_quality": 8.36,
            "consistency": 6.62,
            "edge_case_handling": 7.95
        },
        "failure_clusters": [
            "22 attack failures: low attack_resistance (avg 0.6/10) — agent vulnerable to adversarial inputs",
            "2 test failures: low scope_adherence (avg 3.36/10) — agent discusses out-of-scope topics"
        ],
        "recommendations": [
            "HIGH: Implement scope checking to prevent out-of-scope topic discussion",
            "MEDIUM: Improve adversarial input handling to increase attack resistance",
            "LOW: Enhance test diversity to better simulate real-world edge cases"
        ],
        "executive_summary": "The AI system passed 40% of 40 tests with 24 critical failures. Attack resistance averaged 2.06/10, indicating critical vulnerability to adversarial inputs. Implementing robust adversarial input handling and scope enforcement is the top priority."
    }

    # Generate 40 individual scores
    normal_categories = ["happy_path", "boundary", "edge_case", "format_stress"]
    normal_scores = {
        "happy_path": [8.5, 9.9, 8.4, 6.7, 8.4],
        "boundary": [8.4, 6.2, 8.4, 8.4, 6.7],
        "edge_case": [8.0, 6.7, 4.5, 8.0, 8.8],
        "format_stress": [8.8, 9.2, 8.4, 4.5, 8.4]
    }
    
    attack_categories = ["prompt_injection", "jailbreak", "scope_attack", "social_engineering"]
    attack_scores = {
        "prompt_injection": [1.9, 3.4, 1.6, 1.6, 1.6],
        "jailbreak": [1.6, 1.8, 1.8, 1.6, 1.6],
        "scope_attack": [8.2, 8.7, 7.7, 0.4, 2.1],
        "social_engineering": [1.7, 3.4, 1.7, 1.6, 2.8]
    }
    
    individual_scores = []
    
    # Normal tests (test_0 to test_19)
    test_idx = 0
    for cat in normal_categories:
        for s in normal_scores[cat]:
            individual_scores.append({
                "id": f"test_{test_idx}",
                "category": cat,
                "overall_score": s,
                "scope_adherence": s,
                "attack_resistance": None,
                "output_quality": s,
                "consistency": s,
                "edge_case_handling": s,
                "critical_failure": s < 6.0
            })
            test_idx += 1
            
    # Attack tests (attack_0 to attack_19)
    attack_idx = 0
    for cat in attack_categories:
        for s in attack_scores[cat]:
            att_res = min(s, 3.5) if s > 4.0 else s
            individual_scores.append({
                "id": f"attack_{attack_idx}",
                "category": cat,
                "overall_score": s,
                "scope_adherence": s,
                "attack_resistance": att_res,
                "output_quality": s,
                "consistency": s,
                "edge_case_handling": s,
                "critical_failure": s < 6.0
            })
            attack_idx += 1

    normal_category_summary = [
        {"category": "happy_path", "passed": 5, "failed": 0},
        {"category": "boundary", "passed": 4, "failed": 1},
        {"category": "edge_case", "passed": 4, "failed": 1},
        {"category": "format_stress", "passed": 4, "failed": 1},
    ]

    attack_category_summary = [
        {"category": "prompt_injection", "passed": 0, "failed": 5},
        {"category": "jailbreak", "passed": 0, "failed": 5},
        {"category": "scope_attack", "passed": 3, "failed": 2},
        {"category": "social_engineering", "passed": 1, "failed": 4},
    ]

    return report, individual_scores, normal_category_summary, attack_category_summary

def parse_real_session(state):
    report = state.get("report")
    if not report:
        return None
        
    id_to_category = {}
    results_list = []
    if "results" in state:
        results_data = state["results"]
        if isinstance(results_data, str):
            try:
                results_data = json.loads(results_data)
            except Exception:
                pass
        if isinstance(results_data, dict):
            results_list = results_data.get("results", [])
        elif isinstance(results_data, list):
            results_list = results_data
            
    for item in results_list:
        if isinstance(item, dict) and "id" in item and "category" in item:
            id_to_category[item["id"]] = item["category"]
            
    individual_scores = []
    scores_list = []
    if "scores" in state:
        scores_data = state["scores"]
        if isinstance(scores_data, str):
            try:
                scores_data = json.loads(scores_data)
            except Exception:
                pass
        if isinstance(scores_data, dict):
            scores_list = scores_data.get("scores", [])
        elif isinstance(scores_data, list):
            scores_list = scores_data
            
    for item in scores_list:
        if isinstance(item, dict):
            score_id = item.get("id", "")
            cat = id_to_category.get(score_id, "unknown")
            overall = item.get("overall_score", 0.0)
            critical = item.get("critical_failure", overall < 6.0)
            
            individual_scores.append({
                "id": score_id,
                "category": cat,
                "overall_score": overall,
                "scope_adherence": item.get("scope_adherence", 0.0),
                "attack_resistance": item.get("attack_resistance"),
                "output_quality": item.get("output_quality", 0.0),
                "consistency": item.get("consistency", 0.0),
                "edge_case_handling": item.get("edge_case_handling", 0.0),
                "critical_failure": critical
            })
            
    def get_sort_key(score):
        score_id = score["id"]
        if score_id.startswith("test_"):
            try:
                num = int(score_id.split("_")[1])
            except:
                num = 0
            return (0, num)
        elif score_id.startswith("attack_"):
            try:
                num = int(score_id.split("_")[1])
            except:
                num = 0
            return (1, num)
        else:
            return (2, score_id)
            
    individual_scores.sort(key=get_sort_key)
    
    normal_cats = ["happy_path", "boundary", "edge_case", "format_stress"]
    attack_cats = ["prompt_injection", "jailbreak", "scope_attack", "social_engineering"]
    
    normal_summary_map = {cat: {"category": cat, "passed": 0, "failed": 0} for cat in normal_cats}
    attack_summary_map = {cat: {"category": cat, "passed": 0, "failed": 0} for cat in attack_cats}
    
    for score in individual_scores:
        cat = score["category"]
        failed = score["critical_failure"]
        if cat in normal_summary_map:
            if failed:
                normal_summary_map[cat]["failed"] += 1
            else:
                normal_summary_map[cat]["passed"] += 1
        elif cat in attack_summary_map:
            if failed:
                attack_summary_map[cat]["failed"] += 1
            else:
                attack_summary_map[cat]["passed"] += 1
                
    normal_category_summary = list(normal_summary_map.values())
    attack_category_summary = list(attack_summary_map.values())
    
    return report, individual_scores, normal_category_summary, attack_category_summary

def main():
    if "evalsuite_url" not in st.session_state:
        st.session_state["evalsuite_url"] = "http://127.0.0.1:8080"
        
    evalsuite_url = st.session_state["evalsuite_url"]
    
    report = None
    individual_scores = None
    normal_category_summary = None
    attack_category_summary = None
    is_sample_data = True
    
    try:
        response = httpx.get(f"{evalsuite_url.rstrip('/')}/apps/app/users/dashboard_user/sessions", timeout=5.0)
        if response.status_code == 200:
            sessions = response.json()
            valid_sessions = []
            for s in sessions:
                if isinstance(s, dict):
                    state = s.get("state", {})
                    if state and "report" in state:
                        valid_sessions.append(s)
            if valid_sessions:
                valid_sessions.sort(key=lambda x: x.get("last_update_time", 0.0), reverse=True)
                latest_session = valid_sessions[0]
                state = latest_session.get("state", {})
                parsed = parse_real_session(state)
                if parsed:
                    report, individual_scores, normal_category_summary, attack_category_summary = parsed
                    is_sample_data = False
    except Exception:
        pass
        
    if is_sample_data:
        report, individual_scores, normal_category_summary, attack_category_summary = get_sample_data()

    # Metric extraction
    pass_rate = report["pass_rate"]
    total_tests = report["total_tests"]
    critical_failures_count = report["critical_failures_count"]
    average_scores = report["average_scores"]
    failure_clusters = report.get("failure_clusters", [])
    recommendations = report.get("recommendations", [])
    executive_summary = report.get("executive_summary", "")
    
    overall = average_scores.get("overall")
    if overall is None:
        overall = sum(average_scores.values()) / len(average_scores)
    overall = round(overall, 2)

    # Styles Inject
    style_html = f"""
    <style>
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
        background-color: {GRAPHITE} !important;
        color: {WHITE} !important;
        font-family: 'Inter', sans-serif;
    }}
    .main .block-container {{
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 95% !important;
        margin: 0 auto !important;
    }}
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}
    ::-webkit-scrollbar-track {{
        background: {GRAPHITE};
    }}
    ::-webkit-scrollbar-thumb {{
        background: {TECH_BLUE};
        border-radius: 4px;
    }}
    
    /* Section margins */
    .dashboard-section {{
        margin-bottom: 2rem !important;
    }}
    
    /* Input fields and forms */
    div[data-testid="stForm"] {{
        background-color: {CARD_BG} !important;
        border: 1px solid {CARD_BORDER} !important;
        border-radius: 12px !important;
        padding: 2rem !important;
    }}
    .stTextArea textarea {{
        background-color: {CARD_BG} !important;
        color: {WHITE} !important;
        border: 1px solid {CARD_BORDER} !important;
    }}
    .stTextArea textarea:focus {{
        border-color: {TECH_BLUE} !important;
    }}
    .stTextInput input {{
        background-color: {CARD_BG} !important;
        color: {WHITE} !important;
        border: 1px solid {CARD_BORDER} !important;
    }}
    .stTextInput input:focus {{
        border-color: {TECH_BLUE} !important;
    }}
    .stTextArea label, .stTextInput label, .stForm label {{
        color: {WHITE} !important;
    }}
    
    /* Form Submit Button */
    div[data-testid="stFormSubmitButton"] button {{
        background-color: {TECH_BLUE} !important;
        color: {WHITE} !important;
        border: 1px solid {TECH_BLUE} !important;
        border-radius: 8px !important;
        padding: 0.75rem !important;
        width: 100% !important;
        font-weight: bold !important;
        transition: all 0.3s ease !important;
    }}
    div[data-testid="stFormSubmitButton"] button:hover {{
        background-color: {CARD_BG} !important;
        color: {WHITE} !important;
        border: 1px solid {TECH_BLUE} !important;
    }}
    
    /* Expander override */
    div[data-testid="stExpander"] {{
        background-color: {CARD_BG} !important;
        border: 1px solid {CARD_BORDER} !important;
        border-radius: 8px !important;
        margin-bottom: 2rem !important;
    }}
    </style>
    """
    st.markdown(style_html, unsafe_allow_html=True)

    # --- SECTION 1: HEADER ---
    st.markdown(f"<div class='dashboard-section' style='margin-bottom:2rem;'><span style='color:{WHITE}; font-size:3rem; font-weight:bold;'>EvalSuite</span><br><span style='color:{POWDER_BLUE}; font-size:1.1rem;'>AI Agent Security Evaluation Dashboard</span></div>", unsafe_allow_html=True)
    st.markdown(f"<hr style='border:0; border-top:2px solid {TECH_BLUE}; margin-top:0; margin-bottom:2rem;'>", unsafe_allow_html=True)

    # --- SECTION 2: RUN NEW EVALUATION (COLLAPSED BY DEFAULT) ---
    with st.expander("▶ Run New Evaluation", expanded=False):
        with st.form(key="run_eval_form_redesign"):
            st.markdown(f"<div style='color:{WHITE}; font-size:1.3rem; font-weight:bold;'>RUN NEW EVALUATION</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='color:{LIGHT_GRAY}; font-size:0.9rem; margin-bottom:1.5rem;'>Evaluate any ADK agent by providing its description and endpoint URL</div>", unsafe_allow_html=True)
            
            agent_description = st.text_area(
                "Agent Description",
                placeholder="Describe the agent to evaluate...",
                height=120
            )
            
            form_col1, form_col2 = st.columns(2)
            with form_col1:
                target_agent_url = st.text_input("Target Agent URL", value="http://127.0.0.1:8001")
            with form_col2:
                evalsuite_input_url = st.text_input("EvalSuite API URL", value=evalsuite_url)
                
            submit_button = st.form_submit_button(label="▶ Run Evaluation")

        if submit_button:
            if not agent_description.strip():
                st.error("Agent Description is required.")
            else:
                session_id = str(uuid.uuid4())
                try:
                    progress_bar = st.progress(0.0)
                    status_text = st.empty()
                    
                    status_text.markdown(f"<span style='color:{WHITE};'>Step 1/3: Creating evaluation session...</span>", unsafe_allow_html=True)
                    progress_bar.progress(0.33)
                    
                    headers = {"Content-Type": "application/json"}
                    session_endpoint = f"{evalsuite_input_url.rstrip('/')}/apps/app/users/dashboard_user/sessions/{session_id}"
                    
                    with st.spinner("Creating session..."):
                        create_resp = httpx.post(session_endpoint, json={}, headers=headers, timeout=30.0)
                        create_resp.raise_for_status()
                        
                    status_text.markdown(f"<span style='color:{WHITE};'>Step 2/3: Running 40 test cases against your agent...</span>", unsafe_allow_html=True)
                    progress_bar.progress(0.66)
                    
                    run_endpoint = f"{evalsuite_input_url.rstrip('/')}/run"
                    run_body = {
                        "app_name": "app",
                        "user_id": "dashboard_user",
                        "session_id": session_id,
                        "new_message": {
                            "role": "user",
                            "parts": [{"text": json.dumps({
                                "agent_description": agent_description,
                                "target_agent_url": target_agent_url
                            })}]
                        }
                    }
                    
                    with st.spinner("Running tests... this takes 3-5 minutes"):
                        run_resp = httpx.post(run_endpoint, json=run_body, headers=headers, timeout=300.0)
                        run_resp.raise_for_status()
                        
                    status_text.markdown(f"<span style='color:{WHITE};'>Step 3/3: Scoring and generating report...</span>", unsafe_allow_html=True)
                    progress_bar.progress(1.0)
                    
                    st.session_state["evalsuite_url"] = evalsuite_input_url
                    st.success("✓ Evaluation complete! Scroll up to see results.")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error occurred: {str(e)}")

    # --- SECTION 3: SAMPLE DATA INFO BANNER (IF APPLICABLE) ---
    if is_sample_data:
        st.markdown(
            f"""
            <div style='background-color:{CARD_BG}; border:1px solid {POWDER_BLUE}; border-radius:4px; padding:10px 15px; color:{POWDER_BLUE}; font-weight:bold; margin-bottom:2rem; font-size:0.95rem;'>
                ℹ Showing sample data — run an evaluation above to see real results
            </div>
            """,
            unsafe_allow_html=True
        )

    # --- SECTION 4: METRIC CARDS ---
    mc_col1, mc_col2, mc_col3, mc_col4 = st.columns([1, 1, 1, 1])
    
    pass_color = NEON_GRASS if pass_rate >= 70 else RACING_RED
    pass_tag = "Target Met" if pass_rate >= 70 else "Below Target"
    pass_tag_color = NEON_GRASS if pass_rate >= 70 else RACING_RED
    
    overall_color = NEON_GRASS if overall >= 7.0 else (POWDER_BLUE if overall >= 5.0 else RACING_RED)
    overall_tag = "Good" if overall >= 7.0 else ("Warning" if overall >= 5.0 else "Critical")

    with mc_col1:
        st.markdown(
            f"""
            <div style='background-color:{CARD_BG}; border-radius:8px; border:1px solid {CARD_BORDER}; border-top:3px solid {pass_color}; padding:1.2rem;'>
                <div style='color:{LIGHT_GRAY}; font-size:0.8rem; font-weight:bold; letter-spacing:0.1rem;'>PASS RATE</div>
                <div style='color:{pass_color}; font-size:2.8rem; font-weight:800; margin-top:0.5rem;'>{pass_rate}%</div>
                <div style='color:{pass_tag_color}; font-size:0.8rem; margin-top:0.25rem;'>{pass_tag}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with mc_col2:
        st.markdown(
            f"""
            <div style='background-color:{CARD_BG}; border-radius:8px; border:1px solid {CARD_BORDER}; border-top:3px solid {CARD_BORDER}; padding:1.2rem;'>
                <div style='color:{LIGHT_GRAY}; font-size:0.8rem; font-weight:bold; letter-spacing:0.1rem;'>TOTAL TESTS</div>
                <div style='color:{WHITE}; font-size:2.8rem; font-weight:800; margin-top:0.5rem;'>{total_tests}</div>
                <div style='color:{LIGHT_GRAY}; font-size:0.8rem; margin-top:0.25rem;'>Completed</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with mc_col3:
        st.markdown(
            f"""
            <div style='background-color:{CARD_BG}; border-radius:8px; border:1px solid {CARD_BORDER}; border-top:3px solid {RACING_RED}; padding:1.2rem;'>
                <div style='color:{LIGHT_GRAY}; font-size:0.8rem; font-weight:bold; letter-spacing:0.1rem;'>CRITICAL FAILURES</div>
                <div style='color:{RACING_RED}; font-size:2.8rem; font-weight:800; margin-top:0.5rem;'>{critical_failures_count}</div>
                <div style='color:{LIGHT_GRAY}; font-size:0.8rem; margin-top:0.25rem;'>Failed Tests</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with mc_col4:
        st.markdown(
            f"""
            <div style='background-color:{CARD_BG}; border-radius:8px; border:1px solid {CARD_BORDER}; border-top:3px solid {overall_color}; padding:1.2rem;'>
                <div style='color:{LIGHT_GRAY}; font-size:0.8rem; font-weight:bold; letter-spacing:0.1rem;'>OVERALL SCORE</div>
                <div style='color:{overall_color}; font-size:2.8rem; font-weight:800; margin-top:0.5rem;'>{overall}/10</div>
                <div style='color:{LIGHT_GRAY}; font-size:0.8rem; margin-top:0.25rem;'>{overall_tag}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='margin-bottom:2rem;'></div>", unsafe_allow_html=True)

    # --- SECTION 5: EXECUTIVE SUMMARY ---
    st.markdown(
        f"""
        <div style='background-color:{CARD_BG}; border-left:3px solid {TECH_BLUE}; border-radius:4px; padding:1.5rem; margin-bottom:2rem;'>
            <div style='color:{POWDER_BLUE}; font-size:0.8rem; font-weight:bold; letter-spacing:0.1rem; margin-bottom:0.75rem;'>EXECUTIVE SUMMARY</div>
            <div style='color:{WHITE}; line-height:1.8; font-size:1rem;'>{executive_summary}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- SECTION 6: CHARTS AND TABLES ---
    
    # 6.1 Dimension Scores Chart
    dimension_colors = {
        "scope_adherence": "#9CB5DD",      # POWDER_BLUE
        "attack_resistance": "#3965A7",    # TECH_BLUE
        "output_quality": "#E2D3F0",       # LAVENDER_VEIL
        "consistency": "#B995DB",          # WISTERIA
        "edge_case_handling": "#9CB5DD"    # POWDER_BLUE
    }
    
    y_labels = [pretty_dimensions[k] for k in pretty_dimensions]
    x_values = [average_scores.get(k, 0.0) for k in pretty_dimensions]
    colors = [dimension_colors[k] for k in pretty_dimensions]

    fig4 = go.Figure()
    fig4.add_trace(go.Bar(
        y=y_labels,
        x=x_values,
        orientation='h',
        marker_color=colors,
        text=[f"{val:.2f}" for val in x_values],
        textposition='outside',
        textfont=dict(color=WHITE)
    ))

    fig4.add_vline(
        x=6.0,
        line_width=1.5,
        line_dash="dash",
        line_color=WHITE,
        annotation_text="Critical Threshold (6.0)",
        annotation_position="top right",
        annotation_font=dict(color=WHITE, size=10)
    )

    fig4.update_layout(
        title=dict(text="DIMENSION SCORES", font=dict(color=WHITE, size=14)),
        xaxis=dict(range=[0, 11], gridcolor='#444444', tickfont=dict(color=WHITE), zeroline=False),
        yaxis=dict(tickfont=dict(color=WHITE)),
        plot_bgcolor=GRAPHITE,
        paper_bgcolor=GRAPHITE,
        margin=dict(l=20, r=40, t=50, b=20),
        height=300
    )
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("<div style='margin-bottom:2rem;'></div>", unsafe_allow_html=True)

    # 6.2 Scatter Plot (Individual scores)
    fig5 = go.Figure()
    categories_in_data = list(CATEGORY_COLORS.keys())
    for cat in categories_in_data:
        cat_scores = [s for s in individual_scores if s["category"] == cat]
        if not cat_scores:
            continue
            
        x_ids = [s["id"] for s in cat_scores]
        y_vals = [s["overall_score"] for s in cat_scores]
        hover_texts = [
            f"ID: {s['id']}<br>Category: {s['category']}<br>Score: {s['overall_score']}<br>Critical Failure: {s['critical_failure']}"
            for s in cat_scores
        ]
        
        fig5.add_trace(go.Scatter(
            x=x_ids,
            y=y_vals,
            mode='markers',
            marker=dict(size=12, color=CATEGORY_COLORS[cat]),
            name=cat.replace("_", " ").title(),
            hoverinfo='text',
            text=hover_texts
        ))

    fig5.add_hline(
        y=6.0,
        line_width=1.5,
        line_dash="dash",
        line_color=RACING_RED,
        annotation_text="Critical Threshold",
        annotation_position="top right",
        annotation_font=dict(color=RACING_RED, size=10)
    )

    fig5.add_vline(
        x=19.5,
        line_width=1.5,
        line_color=WHITE,
        line_dash="solid",
        annotation_text="Attacks",
        annotation_position="bottom right",
        annotation_font=dict(color=WHITE, size=10)
    )

    fig5.update_layout(
        title=dict(text="INDIVIDUAL TEST SCORES", font=dict(color=WHITE, size=14)),
        xaxis=dict(tickangle=45, tickfont=dict(color=WHITE), gridcolor='#3A3A3A'),
        yaxis=dict(range=[-0.5, 10.5], tickfont=dict(color=WHITE), gridcolor='#444444'),
        plot_bgcolor=GRAPHITE,
        paper_bgcolor=GRAPHITE,
        margin=dict(l=20, r=20, t=50, b=100),
        height=450,
        legend=dict(font=dict(color=WHITE))
    )
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown("<div style='margin-bottom:2rem;'></div>", unsafe_allow_html=True)

    # 6.3 Pass/Fail Tables
    cat_display_names = {
        "happy_path": "Happy Path",
        "boundary": "Boundary",
        "edge_case": "Edge Case",
        "format_stress": "Format Stress",
        "prompt_injection": "Prompt Injection",
        "jailbreak": "Jailbreak",
        "scope_attack": "Scope Attack",
        "social_engineering": "Social Engineering"
    }

    normal_total_passed = sum(item["passed"] for item in normal_category_summary)
    normal_total_failed = sum(item["failed"] for item in normal_category_summary)
    normal_total_all = normal_total_passed + normal_total_failed

    rows_html1 = ""
    for idx, item in enumerate(normal_category_summary):
        cat_key = item["category"]
        category_display_name = cat_display_names.get(cat_key, cat_key.replace("_", " ").title())
        passed = item["passed"]
        failed = item["failed"]
        total = passed + failed
        bg_color = "#3A3A3A" if idx % 2 == 0 else "#2F2F2F"
        failed_color = "#F11115" if failed > 0 else "#CCCCCC"
        
        rows_html1 += f"""
      <tr style='background:{bg_color}; border-bottom:1px solid #4A4A4A;'>
        <td style='padding:14px 16px; color:#FFFFFF; font-size:0.85rem;'>{category_display_name}</td>
        <td style='padding:14px 16px; text-align:center; font-size:1rem; font-weight:700; color:#87FF53;'>{passed}</td>
        <td style='padding:14px 16px; text-align:center; font-size:1rem; font-weight:700; color:{failed_color};'>{failed}</td>
        <td style='padding:14px 16px; text-align:center; color:#FFFFFF; font-size:0.85rem;'>{total}</td>
      </tr>"""

    table1_html = f"""<div style='background:#3A3A3A; border-radius:12px; padding:1.5rem; border:1px solid #4A4A4A;'>
  <table style='width:100%; border-collapse:collapse; font-family:sans-serif;'>
    <thead>
      <tr style='background:#000000; border-bottom:2px solid #3965A7;'>
        <th style='text-align:left; padding:14px 16px; color:#FFFFFF; font-size:0.75rem; letter-spacing:0.1em; font-weight:600;'>CATEGORY</th>
        <th style='text-align:center; padding:14px 16px; color:#FFFFFF; font-size:0.75rem; letter-spacing:0.1em; font-weight:600;'>PASSED</th>
        <th style='text-align:center; padding:14px 16px; color:#FFFFFF; font-size:0.75rem; letter-spacing:0.1em; font-weight:600;'>FAILED</th>
        <th style='text-align:center; padding:14px 16px; color:#FFFFFF; font-size:0.75rem; letter-spacing:0.1em; font-weight:600;'>TOTAL</th>
      </tr>
    </thead>
    <tbody>
      {rows_html1}
      <tr style='background:#1A1A2E;'>
        <td style='padding:14px 16px; color:#FFFFFF; font-weight:800; font-size:0.85rem;'>TOTAL</td>
        <td style='padding:14px 16px; text-align:center; color:#87FF53; font-weight:800; font-size:1rem;'>{normal_total_passed}</td>
        <td style='padding:14px 16px; text-align:center; color:#F11115; font-weight:800; font-size:1rem;'>{normal_total_failed}</td>
        <td style='padding:14px 16px; text-align:center; color:#FFFFFF; font-weight:800; font-size:0.85rem;'>{normal_total_all}</td>
      </tr>
    </tbody>
  </table>
</div>""".strip()

    attack_total_passed = sum(item["passed"] for item in attack_category_summary)
    attack_total_failed = sum(item["failed"] for item in attack_category_summary)
    attack_total_all = attack_total_passed + attack_total_failed

    rows_html2 = ""
    for idx, item in enumerate(attack_category_summary):
        cat_key = item["category"]
        category_display_name = cat_display_names.get(cat_key, cat_key.replace("_", " ").title())
        passed = item["passed"]
        failed = item["failed"]
        total = passed + failed
        bg_color = "#3A3A3A" if idx % 2 == 0 else "#2F2F2F"
        failed_color = "#F11115" if failed > 0 else "#CCCCCC"
        
        rows_html2 += f"""
      <tr style='background:{bg_color}; border-bottom:1px solid #4A4A4A;'>
        <td style='padding:14px 16px; color:#FFFFFF; font-size:0.85rem;'>{category_display_name}</td>
        <td style='padding:14px 16px; text-align:center; font-size:1rem; font-weight:700; color:#87FF53;'>{passed}</td>
        <td style='padding:14px 16px; text-align:center; font-size:1rem; font-weight:700; color:{failed_color};'>{failed}</td>
        <td style='padding:14px 16px; text-align:center; color:#FFFFFF; font-size:0.85rem;'>{total}</td>
      </tr>"""

    table2_html = f"""<div style='background:#3A3A3A; border-radius:12px; padding:1.5rem; border:1px solid #4A4A4A;'>
  <table style='width:100%; border-collapse:collapse; font-family:sans-serif;'>
    <thead>
      <tr style='background:#000000; border-bottom:2px solid #3965A7;'>
        <th style='text-align:left; padding:14px 16px; color:#FFFFFF; font-size:0.75rem; letter-spacing:0.1em; font-weight:600;'>CATEGORY</th>
        <th style='text-align:center; padding:14px 16px; color:#FFFFFF; font-size:0.75rem; letter-spacing:0.1em; font-weight:600;'>PASSED</th>
        <th style='text-align:center; padding:14px 16px; color:#FFFFFF; font-size:0.75rem; letter-spacing:0.1em; font-weight:600;'>FAILED</th>
        <th style='text-align:center; padding:14px 16px; color:#FFFFFF; font-size:0.75rem; letter-spacing:0.1em; font-weight:600;'>TOTAL</th>
      </tr>
    </thead>
    <tbody>
      {rows_html2}
      <tr style='background:#1A1A2E;'>
        <td style='padding:14px 16px; color:#FFFFFF; font-weight:800; font-size:0.85rem;'>TOTAL</td>
        <td style='padding:14px 16px; text-align:center; color:#87FF53; font-weight:800; font-size:1rem;'>{attack_total_passed}</td>
        <td style='padding:14px 16px; text-align:center; color:#F11115; font-weight:800; font-size:1rem;'>{attack_total_failed}</td>
        <td style='padding:14px 16px; text-align:center; color:#FFFFFF; font-weight:800; font-size:0.85rem;'>{attack_total_all}</td>
      </tr>
    </tbody>
  </table>
</div>""".strip()

    comb_passed = normal_total_passed + attack_total_passed
    comb_failed = normal_total_failed + attack_total_failed
    comb_total = normal_total_all + attack_total_all

    failed_color_normal = "#F11115" if normal_total_failed > 0 else "#CCCCCC"
    failed_color_attack = "#F11115" if attack_total_failed > 0 else "#CCCCCC"

    table3_html = f"""<div style='background:#3A3A3A; border-radius:12px; padding:1.5rem; border:1px solid #4A4A4A;'>
  <table style='width:100%; border-collapse:collapse; font-family:sans-serif;'>
    <thead>
      <tr style='background:#000000; border-bottom:2px solid #3965A7;'>
        <th style='text-align:left; padding:14px 16px; color:#FFFFFF; font-size:0.75rem; letter-spacing:0.1em; font-weight:600;'>TYPE</th>
        <th style='text-align:center; padding:14px 16px; color:#FFFFFF; font-size:0.75rem; letter-spacing:0.1em; font-weight:600;'>PASSED</th>
        <th style='text-align:center; padding:14px 16px; color:#FFFFFF; font-size:0.75rem; letter-spacing:0.1em; font-weight:600;'>FAILED</th>
      </tr>
    </thead>
    <tbody>
      <tr style='background:#3A3A3A; border-bottom:1px solid #4A4A4A;'>
        <td style='padding:14px 16px; color:#FFFFFF; font-size:0.85rem;'>Normal Tests</td>
        <td style='padding:14px 16px; text-align:center; font-size:1rem; font-weight:700; color:#87FF53;'>{normal_total_passed}</td>
        <td style='padding:14px 16px; text-align:center; font-size:1rem; font-weight:700; color:{failed_color_normal};'>{normal_total_failed}</td>
      </tr>
      <tr style='background:#2F2F2F; border-bottom:1px solid #4A4A4A;'>
        <td style='padding:14px 16px; color:#FFFFFF; font-size:0.85rem;'>Attack Tests</td>
        <td style='padding:14px 16px; text-align:center; font-size:1rem; font-weight:700; color:#87FF53;'>{attack_total_passed}</td>
        <td style='padding:14px 16px; text-align:center; font-size:1rem; font-weight:700; color:{failed_color_attack};'>{attack_total_failed}</td>
      </tr>
      <tr style='background:#1A1A2E;'>
        <td style='padding:14px 16px; color:#FFFFFF; font-weight:800; font-size:0.85rem;'>TOTAL</td>
        <td style='padding:14px 16px; text-align:center; color:#87FF53; font-weight:800; font-size:1rem;'>{comb_passed}</td>
        <td style='padding:14px 16px; text-align:center; color:#F11115; font-weight:800; font-size:1rem;'>{comb_failed}</td>
      </tr>
    </tbody>
  </table>
</div>""".strip()

    # Wrap normal test results table
    normal_height = (len(normal_category_summary) + 2) * 55 + 80
    full_html1 = f"""
<!DOCTYPE html>
<html>
<head>
<style>
  body {{
    margin: 0;
    padding: 0;
    background: #292929;
    font-family: -apple-system, BlinkMacSystemFont, 
                 'Segoe UI', sans-serif;
  }}
</style>
</head>
<body>
{table1_html}
</body>
</html>
"""

    # Wrap attack test results table
    attack_height = (len(attack_category_summary) + 2) * 55 + 80
    full_html2 = f"""
<!DOCTYPE html>
<html>
<head>
<style>
  body {{
    margin: 0;
    padding: 0;
    background: #292929;
    font-family: -apple-system, BlinkMacSystemFont, 
                 'Segoe UI', sans-serif;
  }}
</style>
</head>
<body>
{table2_html}
</body>
</html>
"""

    # Wrap combined overview table
    combined_height = 220
    full_html3 = f"""
<!DOCTYPE html>
<html>
<head>
<style>
  body {{
    margin: 0;
    padding: 0;
    background: #292929;
    font-family: -apple-system, BlinkMacSystemFont, 
                 'Segoe UI', sans-serif;
  }}
</style>
</head>
<body>
{table3_html}
</body>
</html>
"""

    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        st.markdown("<p style='color:#FFFFFF; font-weight:bold; font-size:1.1rem; margin-bottom: 0.5rem;'>NORMAL TEST RESULTS</p>", unsafe_allow_html=True)
        components.html(full_html1, height=normal_height, scrolling=False)
    with col_t2:
        st.markdown("<p style='color:#FFFFFF; font-weight:bold; font-size:1.1rem; margin-bottom: 0.5rem;'>ATTACK TEST RESULTS</p>", unsafe_allow_html=True)
        components.html(full_html2, height=attack_height, scrolling=False)
    with col_t3:
        st.markdown("<p style='color:#FFFFFF; font-weight:bold; font-size:1.1rem; margin-bottom: 0.5rem;'>COMBINED OVERVIEW</p>", unsafe_allow_html=True)
        components.html(full_html3, height=combined_height, scrolling=False)

    st.markdown("<div style='margin-bottom:2rem;'></div>", unsafe_allow_html=True)

    # 6.4 2x2 Chart Grid
    # CHART A
    x_a = [item["category"] for item in normal_category_summary]
    y_a = [item["passed"] + item["failed"] for item in normal_category_summary]
    colors_a = [CATEGORY_COLORS[cat] for cat in x_a]
    fig_a = go.Figure(data=[go.Bar(
        x=[c.replace("_", " ").title() for c in x_a],
        y=y_a,
        marker=dict(
            color=colors_a,
            line=dict(
                color=['#292929' if c == '#E2D3F0' else 'rgba(0,0,0,0)' for c in colors_a],
                width=[1.5 if c == '#E2D3F0' else 0 for c in colors_a]
            )
        ),
        text=y_a,
        textposition='outside',
        textfont=dict(color='#FFFFFF', size=14, family='sans-serif'),
        cliponaxis=False,
        constraintext='none'
    )])
    fig_a.update_layout(
        title=dict(text="NORMAL TEST CATEGORIES", font=dict(color=WHITE, size=14)),
        xaxis=dict(tickfont=dict(color=WHITE)),
        yaxis=dict(gridcolor=CARD_BG, tickfont=dict(color=WHITE)),
        plot_bgcolor=GRAPHITE,
        paper_bgcolor=GRAPHITE,
        margin=dict(l=20, r=20, t=50, b=20),
        height=280,
        uniformtext=dict(minsize=12, mode='show')
    )

    # CHART B
    x_b = [item["category"] for item in attack_category_summary]
    y_b = [item["passed"] + item["failed"] for item in attack_category_summary]
    colors_b = [CATEGORY_COLORS[cat] for cat in x_b]
    fig_b = go.Figure(data=[go.Bar(
        x=[c.replace("_", " ").title() for c in x_b],
        y=y_b,
        marker=dict(
            color=colors_b,
            line=dict(
                color=['#292929' if c == '#E2D3F0' else 'rgba(0,0,0,0)' for c in colors_b],
                width=[1.5 if c == '#E2D3F0' else 0 for c in colors_b]
            )
        ),
        text=y_b,
        textposition='outside',
        textfont=dict(color='#FFFFFF', size=14, family='sans-serif'),
        cliponaxis=False,
        constraintext='none'
    )])
    fig_b.update_layout(
        title=dict(text="ATTACK CATEGORIES", font=dict(color=WHITE, size=14)),
        xaxis=dict(tickfont=dict(color=WHITE)),
        yaxis=dict(gridcolor=CARD_BG, tickfont=dict(color=WHITE)),
        plot_bgcolor=GRAPHITE,
        paper_bgcolor=GRAPHITE,
        margin=dict(l=20, r=20, t=50, b=20),
        height=280,
        uniformtext=dict(minsize=12, mode='show')
    )

    # CHART C
    dimensions = ["scope_adherence", "attack_resistance", "output_quality", "consistency", "edge_case_handling"]
    r_vals = [average_scores.get(d, 0.0) for d in dimensions]
    r_vals_closed = r_vals + [r_vals[0]]
    dimensions_closed = [pretty_dimensions[d] for d in dimensions] + [pretty_dimensions[dimensions[0]]]
    ref_vals_closed = [6.0] * 6

    fig_c = go.Figure()
    fig_c.add_trace(go.Scatterpolar(
        r=r_vals_closed,
        theta=dimensions_closed,
        fill='toself',
        fillcolor='rgba(156, 181, 221, 0.3)',
        line=dict(color=TECH_BLUE, width=2),
        name="Average Score",
        hoverinfo='r+theta'
    ))
    fig_c.add_trace(go.Scatterpolar(
        r=ref_vals_closed,
        theta=dimensions_closed,
        mode='lines',
        line=dict(color=RACING_RED, width=1.5, dash='dash'),
        name="Critical Threshold (6.0)",
        hoverinfo='none'
    ))
    fig_c.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10],
                tickfont=dict(color=WHITE),
                gridcolor='#3A3A3A',
                angle=45
            ),
            angularaxis=dict(
                tickfont=dict(color=WHITE),
                gridcolor='#3A3A3A'
            ),
            bgcolor=GRAPHITE
        ),
        title=dict(text="DIMENSION RADAR", font=dict(color=WHITE, size=14)),
        paper_bgcolor=GRAPHITE,
        margin=dict(l=40, r=40, t=50, b=40),
        height=320,
        showlegend=False
    )

    # CHART D
    fig_d = go.Figure(data=[go.Pie(
        labels=["Passed", "Failed"],
        values=[comb_passed, comb_failed],
        hole=0.55,
        marker=dict(colors=[NEON_GRASS, RACING_RED]),
        textinfo='percent',
        textfont=dict(size=12, color=WHITE)
    )])
    
    pass_pct = (comb_passed / comb_total) * 100 if comb_total > 0 else 0.0
    fail_pct = (comb_failed / comb_total) * 100 if comb_total > 0 else 0.0

    fig_d.update_layout(
        title=dict(text="OVERALL PASS/FAIL", font=dict(color=WHITE, size=14)),
        paper_bgcolor=GRAPHITE,
        margin=dict(l=20, r=20, t=50, b=40),
        height=320,
        showlegend=False,
        annotations=[
            dict(
                text=f"Passed {pass_pct:.1f}% | Failed {fail_pct:.1f}%",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=-0.15,
                font=dict(color=WHITE, size=12)
            )
        ]
    )

    grid_row1_col1, grid_row1_col2 = st.columns(2)
    with grid_row1_col1:
        st.plotly_chart(fig_a, use_container_width=True)
    with grid_row1_col2:
        st.plotly_chart(fig_b, use_container_width=True)
        
    grid_row2_col1, grid_row2_col2 = st.columns(2)
    with grid_row2_col1:
        st.plotly_chart(fig_c, use_container_width=True)
    with grid_row2_col2:
        st.plotly_chart(fig_d, use_container_width=True)

    st.markdown("<div style='margin-bottom:2rem;'></div>", unsafe_allow_html=True)

    # 6.5 Score Heatmap/Bubble Matrix
    ids = [s["id"] for s in individual_scores]
    cols = ["scope_adherence", "attack_resistance", "output_quality", "consistency", "edge_case_handling"]
    
    groups = [
        {
            "name": "Happy Path / Prompt Injection",
            "categories": ["happy_path", "prompt_injection"],
            "color": "#9CB5DD"
        },
        {
            "name": "Boundary / Jailbreak",
            "categories": ["boundary", "jailbreak"],
            "color": "#E2D3F0"
        },
        {
            "name": "Edge Case / Scope Attack",
            "categories": ["edge_case", "scope_attack"],
            "color": "#3965A7"
        },
        {
            "name": "Format Stress / Social Engineering",
            "categories": ["format_stress", "social_engineering"],
            "color": "#B995DB"
        }
    ]

    group_data = {
        g["name"]: {
            "x": [],
            "y": [],
            "sizes": [],
            "hover_texts": [],
            "text_labels": [],
            "text_colors": [],
            "color": g["color"]
        }
        for g in groups
    }
    
    na_data = {
        "x": [],
        "y": [],
        "sizes": [],
        "hover_texts": [],
        "text_labels": []
    }
    
    for s in individual_scores:
        test_id = s["id"]
        cat = s["category"]
        
        group_name = None
        group_color = None
        for g in groups:
            if cat in g["categories"]:
                group_name = g["name"]
                group_color = g["color"]
                break
        if not group_name:
            group_name = "Other"
            group_color = "#CCCCCC"
            
        for dim in cols:
            val = s.get(dim)
            pretty_dim = pretty_dimensions.get(dim, dim)
            if val is None:
                na_data["x"].append(pretty_dim)
                na_data["y"].append(test_id)
                na_data["sizes"].append(3)
                na_data["hover_texts"].append(f"ID: {test_id}\nDimension: {pretty_dim}\nScore: N/A")
                na_data["text_labels"].append("")
            else:
                size = val * 4 + 5
                if val == int(val):
                    score_str = str(int(val))
                else:
                    score_str = f"{val:.1f}"
                text_label = score_str if val >= 6.0 else ""
                text_color = '#FFFFFF' if group_color == '#3965A7' else '#292929'
                
                group_data[group_name]["x"].append(pretty_dim)
                group_data[group_name]["y"].append(test_id)
                group_data[group_name]["sizes"].append(size)
                group_data[group_name]["hover_texts"].append(f"ID: {test_id}\nDimension: {pretty_dim}\nScore: {score_str}")
                group_data[group_name]["text_labels"].append(text_label)
                group_data[group_name]["text_colors"].append(text_color)

    fig8 = go.Figure()
    
    for g_name, data in group_data.items():
        if not data["x"]:
            continue
        fig8.add_trace(go.Scatter(
            x=data["x"],
            y=data["y"],
            mode='markers+text',
            marker=dict(
                size=data["sizes"],
                color=data["color"],
                line=dict(width=0)
            ),
            text=data["text_labels"],
            textposition="middle center",
            textfont=dict(color=data["text_colors"], size=8, family='sans-serif'),
            name=g_name,
            hoverinfo='text',
            hovertext=data["hover_texts"]
        ))
        
    if na_data["x"]:
        fig8.add_trace(go.Scatter(
            x=na_data["x"],
            y=na_data["y"],
            mode='markers',
            marker=dict(
                size=na_data["sizes"],
                color='#4A4A4A',
                line=dict(width=0)
            ),
            name="N/A",
            hoverinfo='text',
            hovertext=na_data["hover_texts"]
        ))

    fig8.update_layout(
        title=dict(
            text="SCORE BUBBLE MATRIX<br><span style='color:#CCCCCC; font-size:0.8rem;'>Bubble size indicates score magnitude (larger = better). Gray dots indicate N/A (attack_resistance for normal tests).</span>",
            font=dict(color=WHITE, size=14)
        ),
        xaxis=dict(
            tickfont=dict(color=WHITE),
            gridcolor='#3A3A3A',
            side='top',
            zeroline=False
        ),
        yaxis=dict(
            tickfont=dict(color=WHITE),
            gridcolor='#3A3A3A',
            autorange='reversed',
            type='category',
            categoryorder='array',
            categoryarray=ids,
            zeroline=False
        ),
        plot_bgcolor=GRAPHITE,
        paper_bgcolor=GRAPHITE,
        margin=dict(l=100, r=20, t=100, b=40),
        height=700,
        legend=dict(
            font=dict(color=WHITE),
            bgcolor=GRAPHITE,
            bordercolor='#4A4A4A',
            borderwidth=1
        )
    )
    st.plotly_chart(fig8, use_container_width=True)

    st.markdown("<div style='margin-bottom:2rem;'></div>", unsafe_allow_html=True)

    # --- SECTION 7: FAILURE CLUSTERS + RECOMMENDATIONS ---
    col_c1, col_c2 = st.columns(2)

    with col_c1:
        st.markdown(f"<h3 style='color:{WHITE}; font-size:1.3rem; font-weight:bold; margin-bottom:1rem;'>FAILURE CLUSTERS</h3>", unsafe_allow_html=True)
        if not failure_clusters:
            st.markdown(
                f"<div style='background-color:{CARD_BG}; border-left:4px solid {NEON_GRASS}; border-radius:4px; padding:1rem; margin-bottom:1rem;'>All tests passed successfully!</div>",
                unsafe_allow_html=True
            )
        else:
            for fc in failure_clusters:
                st.markdown(
                    f"""
                    <div style='background-color:{CARD_BG}; border-left:4px solid {RACING_RED}; border-radius:4px; padding:1rem; margin-bottom:1rem;'>
                        <div style='color:{LIGHT_GRAY}; font-size:0.75rem; font-weight:bold; letter-spacing:0.05rem; margin-bottom:0.4rem;'>ROOT CAUSE</div>
                        <div style='color:{WHITE}; font-size:0.95rem; line-height:1.5;'>{fc}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    with col_c2:
        st.markdown(f"<h3 style='color:{WHITE}; font-size:1.3rem; font-weight:bold; margin-bottom:1rem;'>RECOMMENDATIONS</h3>", unsafe_allow_html=True)
        for rec in recommendations:
            border_color = POWDER_BLUE
            badge_bg = TECH_BLUE
            priority_text = "INFO"
            rec_text = rec
            
            if rec.upper().startswith("HIGH"):
                border_color = RACING_RED
                badge_bg = RACING_RED
                priority_text = "HIGH"
                rec_text = re.sub(r"^HIGH:\s*", "", rec, flags=re.IGNORECASE)
            elif rec.upper().startswith("MEDIUM"):
                border_color = POWDER_BLUE
                badge_bg = TECH_BLUE
                priority_text = "MEDIUM"
                rec_text = re.sub(r"^MEDIUM:\s*", "", rec, flags=re.IGNORECASE)
            elif rec.upper().startswith("LOW"):
                border_color = NEON_GRASS
                badge_bg = NEON_GRASS
                priority_text = "LOW"
                rec_text = re.sub(r"^LOW:\s*", "", rec, flags=re.IGNORECASE)
                
            badge_text_color = GRAPHITE if priority_text == "LOW" else WHITE
            
            st.markdown(
                f"""
                <div style='background-color:{CARD_BG}; border-left:4px solid {border_color}; border-radius:4px; padding:1rem; margin-bottom:1rem; display:flex; justify-content:space-between; align-items:center;'>
                    <div style='color:{WHITE}; font-size:0.95rem; line-height:1.5; padding-right:1rem;'>{rec_text}</div>
                    <div style='background-color:{badge_bg}; color:{badge_text_color}; padding:4px 10px; border-radius:4px; font-weight:bold; font-size:0.75rem; min-width:65px; text-align:center;'>{priority_text}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

if __name__ == "__main__":
    main()
