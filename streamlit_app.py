import streamlit as st
import requests

st.title("📄 Resume Matcher - V1 (TF-IDF + Cosine Similarity)")

st.write("Upload a resume and paste a job description to see the match score.")

# Upload resume
uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

# Job description input
job_description = st.text_area("Paste Job Description")

if st.button("Analyze"):

    if uploaded_file is None:
        st.error("Please upload a resume.")
    else:
        with st.spinner("Analyzing..."):

            files = {
                "resume": (uploaded_file.name, uploaded_file, "application/pdf")
            }

            data = {
                "job_description": job_description
            }

            response = requests.post(
                "http://127.0.0.1:5000/parse-resume",
                files=files,
                data=data
            )

            if response.status_code == 200:
                result = response.json()

                st.success("Analysis Complete!")

                st.write("### Results:")
                st.json(result)

            else:
                st.error(f"Error: {response.text}")
