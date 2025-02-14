import pandas as pd
import joblib
import gradio as gr

# Load saved model, label encoder, and feature names
salary_model = joblib.load('salary_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
feature_names = joblib.load('feature_names.pkl')

def model_predict(user_inputs):
    features = {
        "job_title": user_inputs[0],
        "experience_level": user_inputs[1],
        "employment_type": user_inputs[2],
        "work_model": user_inputs[3],
        "employee_residence": user_inputs[4],
        "company_location": user_inputs[5],
        "company_size": user_inputs[6]
    }

    input_df = pd.DataFrame([features])
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=feature_names, fill_value=0)

    prediction = salary_model.predict(input_encoded)
    predicted_range = label_encoder.inverse_transform(prediction)[0]

    return f"Predicted Salary Range: {predicted_range}"

# Load your data
file_location = "./Resources/data_science_salaries.csv"
salary_df = pd.read_csv(file_location)

# Verify the DataFrame is loaded correctly
print(salary_df.head())

# Ensure columns are present and contain data
required_columns = ['job_title', 'experience_level', 'employment_type', 'work_models', 'employee_residence', 'company_location', 'company_size']
for col in required_columns:
    if col not in salary_df.columns:
        raise ValueError(f"Column {col} is missing from the DataFrame.")
    if salary_df[col].isnull().all():
        raise ValueError(f"Column {col} contains all null values.")

# Convert columns to string if necessary
for col in required_columns:
    salary_df[col] = salary_df[col].astype(str)

# Gradio UI
with gr.Blocks() as demo:
    with gr.Row():
        job_title_input = gr.Dropdown(choices=salary_df['job_title'].unique().tolist(), label="Job Title")
        experience_level_input = gr.Dropdown(choices=salary_df['experience_level'].unique().tolist(), label="Experience Level")
        employment_type_input = gr.Dropdown(choices=salary_df['employment_type'].unique().tolist(), label="Employment Type")
        work_models_input = gr.Dropdown(choices=salary_df['work_models'].unique().tolist(), label="Work Model")

    with gr.Row():
        employee_residence_input = gr.Dropdown(choices=salary_df['employee_residence'].unique().tolist(), label="Employee Residence")
        company_location_input = gr.Dropdown(choices=salary_df['company_location'].unique().tolist(), label="Company Location")
        company_size_input = gr.Dropdown(choices=salary_df['company_size'].unique().tolist(), label="Company Size")

    predict_button = gr.Button("Predict Salary")
    output_label = gr.Textbox(label="Predicted Salary")

    predict_button.click(
        fn=model_predict,
        inputs=[job_title_input, experience_level_input, employment_type_input, work_models_input,
                employee_residence_input, company_location_input, company_size_input],
        outputs=output_label
    )

demo.launch()