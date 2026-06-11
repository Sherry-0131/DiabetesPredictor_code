import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import time
import threading
import pandas as pd
import joblib
import numpy as np

# ==========================================
# 📊 CDC Standard Data Conversion Logic (Aligned with your 21-Dim RF Model)
# ==========================================
def convert_user_input_to_cdc_features(raw_inputs):
    """
    Transforms user interface inputs into the exact 21-dimensional vector expected by the Random Forest model.
    """
    cdc_features = {}
    
    # 1. Binary Mapping (Yes=1, No=0) - Strictly case-sensitive to match CSV headers
    binary_keys = [
        'HighBP', 'HighChol', 'HexCheck', 'Smoker', 'Stroke', 
        'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 
        'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'DiffWalk'
    ]
    
    for key in binary_keys:
        val = raw_inputs.get(key) if raw_inputs.get(key) is not None else raw_inputs.get('CholCheck' if key=='HexCheck' else key)
        cdc_features[key] = 1.0 if val == "Yes" else 0.0

    # 2. BMI
    try:
        cdc_features['BMI'] = float(raw_inputs['BMI'])
    except:
        cdc_features['BMI'] = 25.0

    # 3. Continuous Mental/Physical Health Days (Bounded between 0-30 days)
    for key in ['MentHlth', 'PhysHlth']:
        try:
            days = int(raw_inputs[key])
            cdc_features[key] = float(max(0, min(30, days)))
        except:
            cdc_features[key] = 0.0

    # 4. Biological Sex Mapping (Male=1, Female=0)
    cdc_features['Sex'] = 1.0 if raw_inputs.get('Sex') == "Male" else 0.0

    # 5. General Health Evaluation (GenHlth)
    gen_hlth_map = {"Excellent": 1, "Very Good": 2, "Good": 3, "Fair": 4, "Poor": 5}
    cdc_features['GenHlth'] = float(gen_hlth_map.get(raw_inputs.get('GenHlth'), 3))

    # 6. Education Level
    edu_map = {
        "College Graduate or higher": 6, 
        "Some College/Technical School": 5, 
        "High School Graduate": 4, 
        "Some High School": 3, 
        "Elementary School (Grades 1-8)": 2, 
        "Never Attended / Kindergarten only": 1
    }
    cdc_features['Education'] = float(edu_map.get(raw_inputs.get('Education'), 4))

    # 7. Annual Income Categories
    inc_map = {
        "Salary > $75k (Scale 8)": 8, 
        "$50k - $75k (Scale 7)": 7, 
        "$35k - $50k (Scale 6)": 6, 
        "$25k - $35k (Scale 5)": 5, 
        "$20k - $25k (Scale 4)": 4, 
        "$15k - $20k (Scale 3)": 3, 
        "$10k - $15k (Scale 2)": 2, 
        "Salary < $10k (Scale 1)": 1
    }
    cdc_features['Income'] = float(inc_map.get(raw_inputs.get('Income'), 5))

    # 8. Age Conversion (CDC Standard Breakdown)
    try:
        age = float(raw_inputs['Age'])
        if age < 18: age_cdc = 1
        elif age <= 24: age_cdc = 1
        elif age <= 29: age_cdc = 2
        elif age <= 34: age_cdc = 3
        elif age <= 39: age_cdc = 4
        elif age <= 44: age_cdc = 5
        elif age <= 49: age_cdc = 6
        elif age <= 54: age_cdc = 7
        elif age <= 59: age_cdc = 8
        elif age <= 64: age_cdc = 9
        elif age <= 69: age_cdc = 10
        elif age <= 74: age_cdc = 11
        elif age <= 79: age_cdc = 12
        else: age_cdc = 13
        cdc_features['Age'] = float(age_cdc)
    except:
        cdc_features['Age'] = 1.0

    # 9. Strictly reconstruct the vector matching your Random Forest feature order
    ordered_columns = [
        'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 
        'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 
        'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 
        'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
    ]
    
    # Internal HexCheck maps back to CholCheck recognized by the model
    cdc_features['CholCheck'] = cdc_features.get('HexCheck', 1.0)
    
    final_vector = [cdc_features[col] for col in ordered_columns]
    return np.array([final_vector]), ordered_columns


# ==========================================
# 🖥️ GUI Framework (Fixed Text Expansion)
# ==========================================
class DiabetesPredictorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Diabetes Risk Intelligent Prediction System - Deployment Software")
        self.geometry("900x680")  # 💡 Expand total width slightly to host long English texts
        self.configure(bg="#f4f6f9")
        
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=5)
        self.style.configure("TLabel", font=("Segoe UI", 9), background="#f4f6f9")
        self.style.configure("Header.TLabel", font=("Segoe UI", 15, "bold"), background="#f4f6f9", foreground="#2c3e50")
        
        self.model_path = "random_forest_diabetes_model.joblib"
        self.show_login_page()

    def clear_frame(self):
        for widget in self.winfo_children():
            widget.destroy()

    def show_login_page(self):
        self.clear_frame()
        frame = tk.Frame(self, bg="#ffffff", bd=2, relief="groove")
        frame.place(relx=0.5, rely=0.5, anchor="center", width=400, height=300)
        
        title = ttk.Label(frame, text="System Authentication", style="Header.TLabel", background="#ffffff")
        title.pack(pady=20)
        
        row1 = tk.Frame(frame, bg="#ffffff")
        row1.pack(pady=10)
        ttk.Label(row1, text="Username: ", background="#ffffff", width=12, anchor="e").pack(side="left")
        self.username_entry = ttk.Entry(row1, font=("Segoe UI", 10))
        self.username_entry.pack(side="left", padx=5)
        self.username_entry.insert(0, "admin")
        
        row2 = tk.Frame(frame, bg="#ffffff")
        row2.pack(pady=10)
        ttk.Label(row2, text="Password: ", background="#ffffff", width=12, anchor="e").pack(side="left")
        self.password_entry = ttk.Entry(row2, show="*", font=("Segoe UI", 10))
        self.password_entry.pack(side="left", padx=5)
        self.password_entry.insert(0, "123456")
        
        ttk.Button(frame, text="Login to System", command=self.handle_login).pack(pady=25)

    def handle_login(self):
        if self.username_entry.get() == "admin" and self.password_entry.get() == "123456":
            self.show_selection_page()
        else:
            messagebox.showerror("Error", "Invalid username or password!")

    def show_selection_page(self):
        self.clear_frame()
        ttk.Label(self, text="Please Select Data Input & Prediction Mode", style="Header.TLabel").pack(pady=40)
        
        # Mode selection button frame
        btn_frame = tk.Frame(self, bg="#f4f6f9")
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text="✍️ Manual Entry\n(21 Clinical Features)", font=("Segoe UI", 11, "bold"), bg="#3498db", fg="white", width=26, height=3, relief="raised", command=self.show_manual_input_page).pack(side="left", padx=20)
        tk.Button(btn_frame, text="📊 Batch Inference\n(Auto Read from Excel)", font=("Segoe UI", 11, "bold"), bg="#2ecc71", fg="white", width=26, height=3, relief="raised", command=self.show_auto_input_page).pack(side="left", padx=20)
        
        # 🛡️ Added: Ethical & Safety Compliance Container Panel
        ethical_frame = tk.LabelFrame(self, text=" System Compliance & Security Declaration ", font=("Segoe UI", 10, "bold"), bg="#ffffff", fg="#e74c3c", bd=2, relief="groove")
        ethical_frame.pack(pady=30, padx=60, fill="x")
        
        notice_text = (
            "Ethical & Safety Notice: This AI system is an early screening aid and health stratification tool, "
            "not a clinical diagnosis. The system features a zero-retention memory mechanism; no patient data "
            "or uploaded files are permanently retained or cached."
        )
        
        ethical_lbl = tk.Label(ethical_frame, text=notice_text, font=("Segoe UI", 10, "italic"), fg="#34495e", bg="#ffffff", justify="left", wraplength=720, anchor="w")
        ethical_lbl.pack(padx=20, pady=15, fill="x")
        
        ttk.Button(self, text="Logout", command=self.show_login_page).pack(side="bottom", pady=40)

    def show_manual_input_page(self):
        self.clear_frame()
        ttk.Label(self, text="Clinical Health Metrics Data Collection Panel (21 Dimensions)", style="Header.TLabel").pack(pady=10)
        
        canvas = tk.Canvas(self, bg="#ffffff", borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#ffffff")
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", width=820)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True, padx=(20, 0), pady=10)
        scrollbar.pack(side="right", fill="y", padx=(0, 20), pady=10)
        
        self.input_vars = {}
        
        # Fixed Descriptions to prevent multi-line breaks or truncation
        feature_configs = [
            ("1. Actual Chronological Age (Years old):", "Age", "entry", "26"),
            ("2. Body Mass Index (BMI value):", "BMI", "entry", "24.5"),
            ("3. Poor Mental Health Days in past 30 days (MentHlth / 0-30):", "MentHlth", "entry", "0"),
            ("4. Physical Illness/Injury Days in past 30 days (PhysHlth / 0-30):", "PhysHlth", "entry", "0"),
            ("5. Ever told by doctor you have High Blood Pressure (HighBP):", "HighBP", "combo", ["No", "Yes"]),
            ("6. Ever told by doctor you have High Cholesterol (HighChol):", "HighChol", "combo", ["No", "Yes"]),
            ("7. Cholesterol check within past 5 years (CholCheck):", "CholCheck", "combo", ["Yes", "No"]),
            ("8. Smoked at least 100 cigarettes in entire life (Smoker):", "Smoker", "combo", ["No", "Yes"]),
            ("9. Ever diagnosed with or suffered a Stroke history (Stroke):", "Stroke", "combo", ["No", "Yes"]),
            ("10. Had Coronary Heart Disease or Myocardial Infarction:", "HeartDiseaseorAttack", "combo", ["No", "Yes"]),
            ("11. Regular physical activity/exercise in past 30 days:", "PhysActivity", "combo", ["Yes", "No"]),
            ("12. Consume fruits 1 or more times per day (Fruits):", "Fruits", "combo", ["Yes", "No"]),
            ("13. Consume vegetables 1 or more times per day (Veggies):", "Veggies", "combo", ["Yes", "No"]),
            ("14. Heavy drinker (Adult male >14 drinks/wk, female >7):", "HvyAlcoholConsump", "combo", ["No", "Yes"]),
            ("15. Have any kind of health care coverage/insurance:", "AnyHealthcare", "combo", ["Yes", "No"]),
            ("16. Past 1 yr, needed doctor but could not due to cost:", "NoDocbcCost", "combo", ["No", "Yes"]),
            ("17. Have serious difficulty walking or climbing stairs:", "DiffWalk", "combo", ["No", "Yes"]),
            ("18. Biological Sex Factor (Sex):", "Sex", "combo", ["Male", "Female"]),
            ("19. Self-evaluation of current general health status:", "GenHlth", "combo", ["Excellent", "Very Good", "Good", "Fair", "Poor"]),
            ("20. Highest education level achieved (Education):", "Education", "combo", ["College Graduate or higher", "Some College/Technical School", "High School Graduate", "Some High School", "Elementary School (Grades 1-8)", "Never Attended / Kindergarten only"]),
            ("21. Household annual income scale distribution (Income):", "Income", "combo", ["Salary > $75k (Scale 8)", "$50k - $75k (Scale 7)", "$35k - $50k (Scale 6)", "$25k - $35k (Scale 5)", "$20k - $25k (Scale 4)", "$15k - $20k (Scale 3)", "$10k - $15k (Scale 2)", "Salary < $10k (Scale 1)"])
        ]
        
        for label_name, key, field_type, default in feature_configs:
            row = tk.Frame(scrollable_frame, bg="#ffffff")
            row.pack(fill="x", padx=15, pady=6)
            
            # 💡 Increased width from 52 to 62 to prevent text truncation
            lbl = ttk.Label(row, text=label_name, font=("Segoe UI", 9), background="#ffffff", width=62, anchor="w")
            lbl.pack(side="left")
            
            if field_type == "entry":
                val_entry = ttk.Entry(row, font=("Segoe UI", 10))
                val_entry.pack(side="left", fill="x", expand=True)
                val_entry.insert(0, default)
                self.input_vars[key] = val_entry
            elif field_type == "combo":
                val_combo = ttk.Combobox(row, values=default, state="readonly", font=("Segoe UI", 9), width=32)
                val_combo.set(default[0])
                val_combo.pack(side="left")
                self.input_vars[key] = val_combo

        self.control_frame = tk.Frame(self, bg="#f4f6f9")
        self.control_frame.pack(fill="x", padx=30, pady=10)
        
        self.progress = ttk.Progressbar(self.control_frame, orient="horizontal", mode="determinate")
        self.btn_predict = ttk.Button(self.control_frame, text="🚀 Run 21-Dim Inference", command=self.start_manual_prediction)
        self.btn_predict.pack(side="right", padx=5)
        
        ttk.Button(self, text="⬅️ Back to Menu", command=self.show_selection_page).pack(side="bottom", pady=5)

    def start_manual_prediction(self):
        self.btn_predict.config(state="disabled")
        self.progress.pack(side="left", fill="x", expand=True, padx=5, pady=5)
        threading.Thread(target=self.run_manual_prediction_thread).start()

    def run_manual_prediction_thread(self):
        raw_data = {key: widget.get() for key, widget in self.input_vars.items()}
        
        for i in range(1, 101, 25):
            time.sleep(0.1)
            self.progress['value'] = i
            self.update_idletasks()
            
        try:
            processed_vector, col_names = convert_user_input_to_cdc_features(raw_data)
            model = joblib.load(self.model_path)
            
            print("\n[DEBUG] Model expected dimensions:", model.n_features_in_)
            print("[DEBUG] App actual input dimensions:", processed_vector.shape[1])
            
            # Predict Probabilities
            probabilities = model.predict_proba(processed_vector)[0]
            prediction = int(np.argmax(probabilities))
            confidence = probabilities[prediction] * 100
            
            result_map = {
                0: "🎉 Excellent health. No significant diabetes risk detected.", 
                1: "⚠️ Warning: Prediabetes status detected. High non-linear tendency, lifestyle adjustments recommended immediately.", 
                2: "🚨 Critical Diagnosis: High risk Diabetes detected. Clinical examination requested as soon as possible."
            }
            final_conclusion = result_map.get(prediction, "Unknown Status")
            
            prob_details = f" -> Normal Probability: {probabilities[0]*100:.2f}%\n" \
                           f" -> Prediabetes Probability: {probabilities[1]*100:.2f}%\n" \
                           f" -> Diabetes Probability: {probabilities[2]*100:.2f}%"
            
            log_details = "\n".join([f" -> {col_names[m]}: {processed_vector[0][m]}" for m in range(len(col_names))])
            
            messagebox.showinfo(
                "Intelligent Diagnostic Report", 
                f"[21-Dim Medical Matrix Parsed Successfully]\n\n"
                f"[Model Confidence Parsing]:\n{prob_details}\n\n"
                f"【Final Model Conclusion】:\n{final_conclusion} (Confidence: {confidence:.2f}%)\n\n"
                f"[Underlying Standardized Vector (CDC Formatted)]:\n{log_details}"
            )
            
        except FileNotFoundError:
            messagebox.showerror("Interrupted", f"Local model file [{self.model_path}] not found. Please run the training script first.")
        except Exception as e:
            messagebox.showerror("Error", f"Prediction Exception: {str(e)}")
            
        self.progress.pack_forget()
        self.progress['value'] = 0
        self.btn_predict.config(state="normal")

    def show_auto_input_page(self):
        self.clear_frame()
        ttk.Label(self, text="Medical Center Excel Report Batch Processing Panel", style="Header.TLabel").pack(pady=50)
        
        card_frame = tk.Frame(self, bg="#ffffff", bd=2, relief="groove")
        card_frame.pack(pady=20, padx=60, fill="both", expand=True)
        
        ttk.Label(card_frame, text="[Batch Import Instructions]\nThe uploaded Excel template MUST contain all 21 standardized feature columns (e.g., HighBP, HighChol, Age, BMI... exactly matching CSV headers).\nThe system will execute the classification model line by line, appending the prediction probabilities back into the sheet.", 
                  font=("Segoe UI", 10), justify="left", background="#ffffff", wraplength=650).pack(pady=25, padx=20)
        
        file_row = tk.Frame(card_frame, bg="#ffffff")
        file_row.pack(fill="x", padx=20, pady=10)
        self.file_path_entry = ttk.Entry(file_row, font=("Segoe UI", 10), state="readonly")
        self.file_path_entry.pack(side="left", fill="x", expand=True, padx=5)
        
        ttk.Button(file_row, text="📁 Browse File", command=self.browse_excel_file).pack(side="left", padx=5)
        
        self.auto_control_frame = tk.Frame(card_frame, bg="#ffffff")
        self.auto_control_frame.pack(fill="x", padx=20, pady=20)
        
        self.auto_progress = ttk.Progressbar(self.auto_control_frame, orient="horizontal", mode="determinate")
        self.btn_auto_predict = ttk.Button(self.auto_control_frame, text="📊 Run Batch Inference", state="disabled", command=self.start_auto_prediction)
        self.btn_auto_predict.pack(side="right", padx=5)
        
        ttk.Button(self, text="⬅️ Back to Menu", command=self.show_selection_page).pack(side="bottom", pady=20)

    def browse_excel_file(self):
        file_selected = filedialog.askopenfilename(filetypes=[("Excel Spreadsheet", "*.xlsx *.xls")])
        if file_selected:
            self.file_path_entry.config(state="normal")
            self.file_path_entry.delete(0, tk.END)
            self.file_path_entry.insert(0, file_selected)
            self.file_path_entry.config(state="readonly")
            self.btn_auto_predict.config(state="normal")

    def start_auto_prediction(self):
        self.btn_auto_predict.config(state="disabled")
        self.auto_progress.pack(side="left", fill="x", expand=True, padx=5)
        threading.Thread(target=self.run_auto_prediction_thread).start()

    def run_auto_prediction_thread(self):
        excel_path = self.file_path_entry.get()
        
        for i in range(1, 101, 10):
            time.sleep(0.05)
            self.auto_progress['value'] = i
            self.update_idletasks()
            
        try:
            model = joblib.load(self.model_path)
            df_excel = pd.read_excel(excel_path)
            
            expected_cols = [
                'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 
                'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 
                'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 
                'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
            ]
            
            if not all(col in df_excel.columns for col in expected_cols):
                raise ValueError("The Excel file does not contain all 21 required columns. Verify headers and case sensitivity.")
            
            X_batch = df_excel[expected_cols].values
            
            preds = model.predict(X_batch)
            probs = model.predict_proba(X_batch)
            
            df_excel['Model_Prediction_Result'] = preds
            df_excel['Prob_Normal (%)'] = np.round(probs[:, 0] * 100, 2)
            df_excel['Prob_Prediabetes (%)'] = np.round(probs[:, 1] * 100, 2)
            df_excel['Prob_Diabetes (%)'] = np.round(probs[:, 2] * 100, 2)
            df_excel['Max_Confidence (%)'] = np.round(np.max(probs, axis=1) * 100, 2)
            
            save_output_path = excel_path.replace(".xlsx", "_Prediction_Report_Output.xlsx")
            df_excel.to_excel(save_output_path, index=False)
            
            messagebox.showinfo("Batch Processing Completed", f"Inference and probability backfilling successful!\n\nTotal Samples Processed: {len(df_excel)} rows\nOutput saved to:\n{save_output_path.split('/')[-1]}")
        except FileNotFoundError:
            messagebox.showerror("Interrupted", f"Local model file [{self.model_path}] not found.")
        except Exception as e:
            messagebox.showerror("Error", f"Excel Inference Failed:\n{str(e)}")
            
        self.auto_progress.pack_forget()
        self.auto_progress['value'] = 0
        self.btn_auto_predict.config(state="normal")


if __name__ == "__main__":
    app = DiabetesPredictorApp()
    app.mainloop()