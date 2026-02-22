"""
Streamlit frontend — AI Recruiting Agent v2
"""
import os
import uuid
import requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# URL for the FastAPI backend
API = os.getenv("API_URL", "http://localhost:8000/api/v1")

st.set_page_config(
    page_title="AI Recruiting Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🤖 AI Recruiting")
    st.caption("v2.0")
    st.markdown("---")
    page = st.radio("", [
        "🔍 Find Candidates",
        "📄 Upload Resume",
        "💼 Vacancies",
        "📊 Analytics",
    ])
    st.markdown("---")
    st.markdown("""
**Methods**
- 🔵 **BM25** — ES full-text baseline  
- 🟣 **Semantic** — Cosine similarity  
- 🟢 **LLM** — BM25→GPT   
- ⚡ **Hybrid** — RRF+rerank+LLM  
""")


def api(method, path, **kwargs):
    try:
        fn = getattr(requests, method)
        r = fn(f"{API}{path}", timeout=90, **kwargs)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("⚠️ API not reachable")
        return None
    except requests.exceptions.HTTPError as e:
        # Try to extract human-readable detail from FastAPI JSON error response
        detail = ""
        try:
            body = e.response.json()
            detail = body.get("detail", "")
        except Exception:
            pass
        if detail:
            st.error(f"⚠️ {detail}")
        else:
            st.error(f"API error: {e}")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


# ── FIND CANDIDATES ───────────────────────────────────────────────────────────
if "🔍 Find Candidates" in page:
    st.header("🔍 Find Top Candidates")

    vacancies = api("get", "/vacancies") or []
    if not vacancies:
        st.info("Create a vacancy first in the 💼 Vacancies tab.")
        st.stop()

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        opts = {f"{v['title']}  (id: {v['id'][:8]}...)": v["id"] for v in vacancies}
        sel = st.selectbox("Select Vacancy", list(opts.keys()))
        job_id = opts[sel]
    with col2:
        method = st.selectbox("Method", ["hybrid", "bm25", "semantic", "llm"])
    with col3:
        top_k = st.slider("Top K", 1, 20, 5)

    if st.button("🚀 Find", type="primary", use_container_width=True):
        with st.spinner("Matching..."):
            res = api("get", "/recommendations", params={"job_id": job_id, "method": method, "top_k": top_k})

        if res:
            candidates = res.get("top_candidates", [])
            st.success(
                f"✅ {len(candidates)} candidates found in **{res['processing_time_seconds']:.2f}s** "
                f"({res['total_candidates_evaluated']} evaluated)"
            )

            if candidates:
                # Score chart
                names = [c["candidate_name"] for c in candidates]
                fig = go.Figure()

                def add_bar(key, label, color):
                    vals = [c.get(key) or 0 for c in candidates]
                    if any(v > 0 for v in vals):
                        fig.add_trace(go.Bar(name=label, x=names, y=vals, marker_color=color))

                add_bar("score", "Final Score", "#6366f1")
                add_bar("bm25_score", "BM25", "#f59e0b")
                add_bar("semantic_score", "Semantic", "#3b82f6")
                add_bar("llm_score", "LLM", "#10b981")

                fig.update_layout(
                    barmode="group", yaxis_range=[0, 1],
                    title=f"Scores — {res['job_title']} | method: {method}",
                    height=340,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig, use_container_width=True)

                # Cards
                for c in candidates:
                    from_cache = "💾 cached" if c.get("from_cache") else ""
                    with st.expander(
                        f"#{c['rank']}  {c['candidate_name']} — **{c['score']:.2%}** {from_cache}",
                        expanded=c["rank"] == 1,
                    ):
                        cols = st.columns(4)
                        for col, (label, key) in zip(cols, [
                            ("Final", "score"), ("BM25", "bm25_score"),
                            ("Semantic", "semantic_score"), ("LLM", "llm_score"),
                        ]):
                            v = c.get(key)
                            col.metric(label, f"{v:.2%}" if v is not None else "—")

                        if c.get("matched_skills"):
                            st.markdown("✅ **Matched:** " + " ".join(f"`{s}`" for s in c["matched_skills"]))
                        if c.get("missing_skills"):
                            st.markdown("❌ **Missing:** " + " ".join(f"`{s}`" for s in c["missing_skills"]))
                        if c.get("explanation"):
                            st.info(f"💬 {c['explanation']}")


# ── UPLOAD RESUME ─────────────────────────────────────────────────────────────
elif "📄 Upload Resume" in page:
    st.header("📄 Upload & Parse Resume")
    st.info("LangChain + OpenAI parses the resume into structured fields, then stores with embeddings in Elasticsearch.")

    tab_file, tab_manual = st.tabs(["Upload File", "Enter Manually"])

    with tab_file:
        f = st.file_uploader("PDF / DOCX / TXT", type=["pdf", "docx", "txt"])
        if f and st.button("📤 Parse & Save", type="primary"):
            with st.spinner("Parsing with LLM..."):
                res = api("post", "/candidates/upload",
                          files={"file": (f.name, f.getvalue(), f.type)})
            if res:
                st.success(f"✅ Saved: **{res['name']}**")
                st.json({k: v for k, v in res.items() if k not in ("raw_text",)})

    with tab_manual:
        with st.form("manual_candidate"):
            name = st.text_input("Name*")
            email = st.text_input("Email")
            role = st.text_input("Role", placeholder="Senior Python Developer")
            skills_raw = st.text_input("Skills (comma-separated)", placeholder="python, fastapi, docker")
            education = st.text_input("Education (specialty)", placeholder="Computer Science")
            experience = st.text_area("Experience", height=120,
                placeholder="5 years backend. TechCorp 2020-2024: built microservices in FastAPI...")
            submitted = st.form_submit_button("Save Candidate", type="primary")

            if submitted and name:
                res = api("post", "/candidates", json={
                    "id": str(uuid.uuid4()),
                    "name": name,
                    "email": email or None,
                    "role": role or None,
                    "skills": [s.strip().lower() for s in skills_raw.split(",") if s.strip()],
                    "education": education or None,
                    "experience": experience or None,
                    "raw_text": f"{name} {role} {skills_raw} {education} {experience}",
                })
                if res:
                    st.success(f"✅ Saved: {name}")

    st.markdown("---")
    st.subheader("All Candidates")
    candidates = api("get", "/candidates") or []
    if candidates:
        df = pd.DataFrame([{
            "Name": c["name"],
            "Role": c.get("role", "—"),
            "Skills": ", ".join((c.get("skills") or [])[:6]),
            "Education": c.get("education", "—"),
        } for c in candidates])
        st.dataframe(df, use_container_width=True, height=300)
    else:
        st.info("No candidates yet.")


# ── VACANCIES ─────────────────────────────────────────────────────────────────
elif "💼 Vacancies" in page:
    st.header("💼 Manage Vacancies")

    with st.expander("➕ Create Vacancy", expanded=True):
        with st.form("vac_form"):
            title = st.text_input("Title*", placeholder="Senior Python Developer")
            role = st.text_input("Role", placeholder="Backend Developer")
            req_skills = st.text_input("Required skills (comma-separated)", placeholder="python, fastapi, docker")
            req_edu = st.text_input("Required education (specialty)", placeholder="Computer Science")
            description = st.text_area("Description*", height=160)

            if st.form_submit_button("Create", type="primary"):
                if title and description:
                    res = api("post", "/vacancies", json={
                        "id": str(uuid.uuid4()),
                        "title": title,
                        "role": role or None,
                        "required_skills": [s.strip().lower() for s in req_skills.split(",") if s.strip()],
                        "required_education": req_edu or None,
                        "description": description,
                    })
                    if res:
                        st.success(f"✅ Created: {title}")
                        st.rerun()
                else:
                    st.error("Title and description required")

    st.markdown("---")
    st.subheader("Existing Vacancies")
    vacancies = api("get", "/vacancies") or []
    for v in vacancies:
        with st.expander(f"**{v['title']}** — role: {v.get('role','—')}"):
            st.write(v["description"])
            if v.get("required_skills"):
                st.markdown("**Required:** " + ", ".join(f"`{s}`" for s in v["required_skills"]))
            if v.get("required_education"):
                st.markdown(f"**Education:** {v['required_education']}")


# ── ANALYTICS ─────────────────────────────────────────────────────────────────
elif "📊 Analytics" in page:
    st.header("📊 Analytics")

    candidates = api("get", "/candidates") or []
    vacancies = api("get", "/vacancies") or []

    m1, m2, m3 = st.columns(3)
    m1.metric("Candidates", len(candidates))
    m2.metric("Vacancies", len(vacancies))
    avg_skills = sum(len(c.get("skills") or []) for c in candidates) / max(len(candidates), 1)
    m3.metric("Avg Skills / Candidate", f"{avg_skills:.1f}")

    if candidates:
        from collections import Counter
        all_skills = [s for c in candidates for s in (c.get("skills") or [])]
        if all_skills:
            top = Counter(all_skills).most_common(20)
            df_s = pd.DataFrame(top, columns=["Skill", "Count"])
            fig = px.bar(df_s, x="Count", y="Skill", orientation="h",
                         title="Top Skills", color="Count",
                         color_continuous_scale="Blues")
            fig.update_layout(height=500, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

        roles = [c.get("role") for c in candidates if c.get("role")]
        if roles:
            fig2 = px.pie(values=list(Counter(roles).values()),
                          names=list(Counter(roles).keys()),
                          title="Candidate Roles")
            st.plotly_chart(fig2, use_container_width=True)
