import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Resume ↔ Job Matcher", page_icon="📄", layout="centered")

st.title("📄 AI Resume ↔ Job Description Matcher")
st.write(
    "Paste your resume and a job description below. The app uses **TF-IDF + cosine "
    "similarity** to calculate a match score, and highlights important keywords from "
    "the job posting that are missing from your resume."
)

st.divider()

col1, col2 = st.columns(2)
with col1:
    resume_text = st.text_area(
        "📝 Your Resume Text",
        height=250,
        placeholder="Paste your resume content here (skills, experience, projects, etc.)",
    )
with col2:
    job_text = st.text_area(
        "💼 Job Description",
        height=250,
        placeholder="Paste the job description you're applying for here.",
    )

top_n = st.slider("Number of key skills to check for:", 5, 20, 10)


def compute_match(resume, job, n_keywords=10):
    documents = [resume, job]

    # 1. Overall match score using TF-IDF + cosine similarity
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(documents)
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    # 2. Find important keywords in the job description specifically
    job_vectorizer = TfidfVectorizer(stop_words="english")
    job_matrix = job_vectorizer.fit_transform([job])
    feature_names = np.array(job_vectorizer.get_feature_names_out())
    scores = job_matrix.toarray()[0]

    top_indices = scores.argsort()[::-1][:n_keywords]
    top_keywords = feature_names[top_indices]

    resume_lower = resume.lower()
    matched = [kw for kw in top_keywords if kw in resume_lower]
    missing = [kw for kw in top_keywords if kw not in resume_lower]

    return similarity, matched, missing


if st.button("🔍 Analyze Match", type="primary"):
    if not resume_text.strip() or not job_text.strip():
        st.warning("Please paste both your resume and the job description.")
    else:
        similarity, matched, missing = compute_match(resume_text, job_text, top_n)
        match_pct = round(similarity * 100, 1)

        st.subheader(f"Match Score: {match_pct}%")
        st.progress(min(int(match_pct), 100))

        if match_pct >= 60:
            st.success("Strong match! Your resume aligns well with this job.")
        elif match_pct >= 35:
            st.warning("Moderate match. Consider adding some missing keywords below.")
        else:
            st.error("Low match. Your resume may need significant tailoring for this role.")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("### ✅ Keywords Found")
            if matched:
                for kw in matched:
                    st.markdown(f"- {kw}")
            else:
                st.write("None of the top keywords were found.")

        with col_b:
            st.markdown("### ❌ Missing Keywords")
            if missing:
                for kw in missing:
                    st.markdown(f"- {kw}")
            else:
                st.write("Great! No major keywords missing.")

        st.divider()
        st.caption(
            "Tip: Missing keywords aren't always required verbatim, but adding relevant "
            "ones (if truthful to your experience) can improve ATS (Applicant Tracking "
            "System) matching."
        )

st.divider()
st.caption("Built with Streamlit + scikit-learn (TF-IDF + Cosine Similarity)")
