import streamlit as st
import pandas as pd
import google.generativeai as genai   

# Configure page
st.set_page_config(
    page_title="Medical Results Interpreter",
    page_icon="🏥",
    layout="wide"
)

# Title and description
st.title("AI Medical Test Analyzer")
st.markdown("""
Upload patient test results in CSV format to get AI-powered interpretation and recommendations.
""")

# File upload section
uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
    help="Format: Test_Name,Result,Unit,Reference_Range"
)

if uploaded_file:
    try:
        # Read uploaded CSV
        df = pd.read_csv(uploaded_file)
        st.success("✅ File successfully uploaded!")

        with st.expander("📊 View Raw Data"):
            st.dataframe(df)

        # Build the prompt
        analysis_text = (
            "You are a medical assistant. Analyze the following test results:\n"
            "1. Identify abnormal values.\n"
            "2. Explain possible health implications.\n"
            "3. Provide patient-friendly clinical suggestions.\n\n"
        )
        analysis_text += "\n".join(
            f"{row['Test_Name']}: {row['Result']} {row['Unit']} (Normal: {row['Reference_Range']})"
            for _, row in df.iterrows()
        )

        # BONUS: Show AI prompt (debug)
        st.subheader("🧪 Prompt Sent to Gemini AI")
        st.code(analysis_text, language='text')

        # Trigger AI analysis
        if st.button("Generate Medical Analysis", type="primary"):
            with st.spinner("🧠 Analyzing with Gemini AI..."):
                try:
                    # ✅ Configure Gemini
                    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

                    # Load Gemini model and generate
                    model = genai.GenerativeModel("models/gemini-2.5-pro")  
                    response = model.generate_content(analysis_text)

                    # Extract text
                    analysis = response.text

                    # Display result
                    st.subheader("📄 AI Analysis Report")
                    st.markdown(
                        f"<div style='background:#f0f2f6;padding:20px;border-radius:10px'>{analysis}</div>",
                        unsafe_allow_html=True
                    )

                    # Option to download
                    st.download_button(
                        label="⬇️ Download Full Report",
                        data=analysis,
                        file_name="medical_analysis.txt",
                        mime="text/plain"
                    )

                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    st.info("🔑 Please check your Gemini API key in Streamlit secrets.")

    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")
        st.info("📄 Make sure your CSV has: Test_Name, Result, Unit, Reference_Range")
