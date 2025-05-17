import streamlit as st
import pandas as pd
import joblib
import altair as alt
import io
from fpdf import FPDF
import base64
import tempfile
import os
import matplotlib.pyplot as plt

# === Load model and metadata ===
st.set_page_config(page_title="Hospital Crowding Predictor", layout="wide")
model_data = joblib.load("crowding_model.pkl")
model = model_data["model"]
expected_columns = model_data["columns"]

# === Title ===
st.title("🏥 Hospital Crowding Predictor")
st.markdown("Forecast crowding in hospital sections and plan ahead effectively")

# === Sidebar Configuration ===
st.sidebar.header("🛠️ Settings")
selected_section = st.sidebar.selectbox("🏷️ Section", ["ICU", "Emergency", "General Ward"], key="section_select")
forecast_period = st.sidebar.selectbox("🗓️ Forecast Period", ["3 months", "6 months"], key="forecast_select")

# === Input Form ===
st.subheader("📝 Section-Specific Inputs")
with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        total_hospital_beds = st.number_input("Total Hospital Beds", min_value=0)
        available_hospital_beds = st.number_input("Available Hospital Beds", min_value=0)
        potentially_available_hospital_beds = st.number_input("Potentially Available Hospital Beds", min_value=0)
        projected_infected_individuals = st.number_input("Projected Infected Individuals", min_value=0)
    with col2:
        total_unit_beds = st.number_input(f"Total {selected_section} Beds", min_value=0)
        available_unit_beds = st.number_input(f"Available {selected_section} Beds", min_value=0)
        adult_population = st.number_input("Adult Population", min_value=0)
        population_65_plus = st.number_input("Population 65+", min_value=0)

    st.subheader("🧑‍⚕️ Staff & Patient Load")
    staff_count = st.number_input("Staff Count in this Section", min_value=0)
    patients_last_2_months = st.number_input("Patients in the Last 2 Months", min_value=0)
    submitted = st.form_submit_button("🔍 Predict Crowding")

# === Processing & Output ===
if submitted:
    available_total = available_hospital_beds + potentially_available_hospital_beds
    crowding_ratio = projected_infected_individuals / available_total if available_total > 0 else 0

    raw_input = {
        "crowding_ratio": crowding_ratio
    }
    model_input = pd.DataFrame([raw_input])[expected_columns]

    st.write("📈 Model Input Preview:", model_input)
    st.info(f"🔍 Calculated Crowding Ratio: {crowding_ratio:.2f}")

    prediction = model.predict(model_input)[0]
    result_label = "CROWDED" if prediction == 1 else "NOT Crowded"

    if prediction == 1:
        st.error("🚨 This section is likely to be **CROWDED**.")
    else:
        st.success("✅ This section is **NOT** likely to be crowded.")

    # === Summary Table ===
    st.subheader("📋 Prediction Summary")
    input_summary = pd.DataFrame([{
        "total_hospital_beds": total_hospital_beds,
        "available_hospital_beds": available_hospital_beds,
        "potentially_available_hospital_beds": potentially_available_hospital_beds,
        "total_icu_beds": total_unit_beds,
        "available_icu_beds": available_unit_beds,
        "projected_infected_individuals": projected_infected_individuals,
        "adult_population": adult_population,
        "population_65+": population_65_plus,
        "crowding_ratio": f"{crowding_ratio:.2f}",
        "Section": selected_section,
        "Forecast Period": forecast_period,
        "Staff Count": staff_count,
        "Patients Last 2 Months": patients_last_2_months,
        "Prediction Result": result_label
    }])
    st.dataframe(input_summary)

    csv_buffer = io.StringIO()
    input_summary.to_csv(csv_buffer, index=False)
    st.download_button("🗅️ Download as CSV", data=csv_buffer.getvalue(), file_name="prediction_summary.csv", mime="text/csv")

    # === Bed Demand Chart ===
    demand_vs_supply_df = pd.DataFrame({
        "Type": ["Available Beds", "Projected Demand"],
        "Beds": [available_total, projected_infected_individuals]
    })

    chart_path = os.path.join(tempfile.gettempdir(), "bed_pressure_chart.png")
    fig, ax = plt.subplots()
    labels = ['Available Beds', 'Projected Demand']
    values = [available_total, projected_infected_individuals]
    colors = ['green', 'red']
    ax.barh(labels, values, color=colors)
    ax.set_xlabel('Beds')
    ax.set_title('Hospital Bed Demand vs Availability')
    for i, v in enumerate(values):
        ax.text(v + 2, i, str(v), color='black', va='center')
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()

    # === PDF Report Generation ===
    class PDF(FPDF):
        def header(self):
            self.set_font("Arial", 'B', 14)
            self.cell(0, 10, "Hospital Crowding Prediction Report", ln=True, align='C')
            self.ln(10)

        def add_section(self, title, content):
            self.set_font("Arial", 'B', 12)
            self.cell(0, 10, title, ln=True)
            self.set_font("Arial", '', 11)
            for line in content:
                safe_line = line.encode('latin-1', 'replace').decode('latin-1')
                self.cell(0, 8, f"- {safe_line}", ln=True)
            self.ln(4)

    pdf = PDF()
    pdf.add_page()
    details = [f"{col}: {val}" for col, val in input_summary.iloc[0].items() if col != 'Prediction Result']
    pdf.add_section("Inputs Provided", details)
    pdf.add_section("Prediction", [f"This section is likely to be: {result_label}"])

    if patients_last_2_months > 0:
        ratio = staff_count / patients_last_2_months
        if ratio < 0.05:
            alert = "Very Low Staff Ratio - Consider increasing staff!"
        elif ratio < 0.1:
            alert = "Low Staff Ratio - May struggle under load."
        else:
            alert = "Healthy Staff Ratio"
    else:
        alert = "No patient data to evaluate staff ratio."

    pdf.add_section("Staff-to-Patient Ratio", [
        f"Staff Count: {staff_count}",
        f"Patient Count: {patients_last_2_months}",
        f"Ratio: {staff_count / patients_last_2_months:.2f}" if patients_last_2_months > 0 else "N/A",
        alert
    ])
    pdf.image(chart_path, x=10, w=190)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        pdf.output(tmp_file.name)
        tmp_file_path = tmp_file.name

    with open(tmp_file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
        pdf_download_link = f'<a href="data:application/pdf;base64,{base64_pdf}" download="crowding_prediction_report.pdf">📄 Download PDF Report</a>'
        st.markdown(pdf_download_link, unsafe_allow_html=True)

    os.unlink(tmp_file_path)

    # === Visualization: Bed Demand ===
    st.subheader("📉 Bed Demand vs Availability")
    demand_chart = alt.Chart(demand_vs_supply_df).mark_bar().encode(
        x=alt.X("Beds", title="Number of Beds"),
        y=alt.Y("Type", sort='-x'),
        color=alt.Color("Type", scale=alt.Scale(range=['green', 'red'])),
        tooltip=['Type', 'Beds']
    ).properties(height=200)
    st.altair_chart(demand_chart, use_container_width=True)

    # === Visualization: Patient Load ===
    st.subheader("📈 Patient Load Trend (Simulated)")
    weeks = list(range(1, 9))
    patients_per_week = [int(patients_last_2_months / 8 + (i * 2)) for i in range(8)]
    trend_df = pd.DataFrame({"Week": weeks, "Patients": patients_per_week})
    line_chart = alt.Chart(trend_df).mark_line(point=True).encode(
        x='Week', y='Patients', tooltip=['Week', 'Patients']
    ).properties(height=300)
    st.altair_chart(line_chart, use_container_width=True)

# === Footer ===
st.markdown("""
---
<center style='color: grey;'>Ali Project 21510563 | April 2025</center>
""", unsafe_allow_html=True)