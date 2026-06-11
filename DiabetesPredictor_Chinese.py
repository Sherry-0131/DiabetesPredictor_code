import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import time
import threading
import pandas as pd
import joblib
import numpy as np

# ==========================================
# 📊 CDC 标准数据转换核心逻辑 (与你的随机森林21维特征字符完美对齐)
# ==========================================
def convert_user_input_to_cdc_features(raw_inputs):
    """
    将用户日常直观输入的数值与选项，严格转化为你的随机森林所期待的 21 维模型输入
    """
    cdc_features = {}
    
    # 1. 基础二分类映射 (是=1, 否=0) - 严格对齐 CSV 表头大小写
    binary_keys = [
        'HighBP', 'HighChol', 'HexCheck', 'Smoker', 'Stroke', 
        'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 
        'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'DiffWalk'
    ]
    
    for key in binary_keys:
        val = raw_inputs.get(key) if raw_inputs.get(key) is not None else raw_inputs.get('CholCheck' if key=='HexCheck' else key)
        cdc_features[key] = 1.0 if val == "是" else 0.0

    # 2. BMI
    try:
        cdc_features['BMI'] = float(raw_inputs['BMI'])
    except:
        cdc_features['BMI'] = 25.0

    # 3. 连续天数限制 (健康天数范围 0-30 天)
    for key in ['MentHlth', 'PhysHlth']:
        try:
            days = int(raw_inputs[key])
            cdc_features[key] = float(max(0, min(30, days)))
        except:
            cdc_features[key] = 0.0

    # 4. 性别映射 (男=1, 女=0)
    cdc_features['Sex'] = 1.0 if raw_inputs.get('Sex') == "男" else 0.0

    # 5. 主观健康状况 GenHlth
    gen_hlth_map = {"极好 (Excellent)": 1, "很好 (Very Good)": 2, "好 (Good)": 3, "一般 (Fair)": 4, "较差 (Poor)": 5}
    cdc_features['GenHlth'] = float(gen_hlth_map.get(raw_inputs.get('GenHlth'), 3))

    # 6. 教育程度 Education
    edu_map = {"大学毕业及以上": 6, "大学/专科肄业": 5, "高中毕业": 4, "初中肄业/普通高中": 3, "小学 (1-8年级)": 2, "从未上学/仅幼儿园": 1}
    cdc_features['Education'] = float(edu_map.get(raw_inputs.get('Education'), 4))

    # 7. 收入阶层 Income
    inc_map = {"年薪 > $75k (级别8)": 8, "环境偏高 $50k-$75k (级别7)": 7, "中产阶层 $35k-$50k (级别6)": 6, "温饱阶层 $25k-$35k (级别5)": 5, 
               "$20k-$25k (级别4)": 4, "$15k-$20k (级别3)": 3, "$10k-$15k (级别2)": 2, "年薪 < $10k (级别1)": 1}
    cdc_features['Income'] = float(inc_map.get(raw_inputs.get('Income'), 5))

    # 8. 年龄转换 (CDC 标准)
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

    # 9. 严格按照你随机森林训练集 X 的 21 个特征列名称及顺序完全重组
    ordered_columns = [
        'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 
        'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 
        'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 
        'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
    ]
    
    # 将内部特殊处理的 HexCheck 映射回随机森林认的 CholCheck
    cdc_features['CholCheck'] = cdc_features.get('HexCheck', 1.0)
    
    final_vector = [cdc_features[col] for col in ordered_columns]
    return np.array([final_vector]), ordered_columns


# ==========================================
# 🖥️ GUI 界面框架开发
# ==========================================
class DiabetesPredictorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("糖尿病风险智能预测系统 - 随机森林大作业部署软件")
        self.geometry("820x650")
        self.configure(bg="#f4f6f9")
        
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("TButton", font=("微软雅黑", 10, "bold"), padding=5)
        self.style.configure("TLabel", font=("微软雅黑", 9), background="#f4f6f9")
        self.style.configure("Header.TLabel", font=("微软雅黑", 15, "bold"), background="#f4f6f9", foreground="#2c3e50")
        
        self.model_path = "random_forest_diabetes_model.joblib"
        self.show_login_page()

    def clear_frame(self):
        for widget in self.winfo_children():
            widget.destroy()

    def show_login_page(self):
        self.clear_frame()
        frame = tk.Frame(self, bg="#ffffff", bd=2, relief="groove")
        frame.place(relx=0.5, rely=0.5, anchor="center", width=380, height=280)
        
        title = ttk.Label(frame, text="系统身份验证", style="Header.TLabel", background="#ffffff")
        title.pack(pady=20)
        
        row1 = tk.Frame(frame, bg="#ffffff")
        row1.pack(pady=10)
        ttk.Label(row1, text="账  号: ", background="#ffffff").pack(side="left")
        self.username_entry = ttk.Entry(row1, font=("微软雅黑", 10))
        self.username_entry.pack(side="left", padx=5)
        self.username_entry.insert(0, "admin")
        
        row2 = tk.Frame(frame, bg="#ffffff")
        row2.pack(pady=10)
        ttk.Label(row2, text="密  码: ", background="#ffffff").pack(side="left")
        self.password_entry = ttk.Entry(row2, show="*", font=("微软雅黑", 10))
        self.password_entry.pack(side="left", padx=5)
        self.password_entry.insert(0, "123456")
        
        ttk.Button(frame, text="登 录 系 统", command=self.handle_login).pack(pady=20)

    def handle_login(self):
        if self.username_entry.get() == "admin" and self.password_entry.get() == "123456":
            self.show_selection_page()
        else:
            messagebox.showerror("错误", "账号或密码错误！")

    def show_selection_page(self):
        self.clear_frame()
        ttk.Label(self, text="请选择数据导入与预测模式", style="Header.TLabel").pack(pady=80)
        
        btn_frame = tk.Frame(self, bg="#f4f6f9")
        btn_frame.pack(pady=20)
        
        tk.Button(btn_frame, text="✍️ 21维临床指标手动输入", font=("微软雅黑", 12, "bold"), bg="#3498db", fg="white", width=24, height=3, relief="raised", command=self.show_manual_input_page).pack(side="left", padx=25)
        tk.Button(btn_frame, text="📊 Excel 数据自动批量读取", font=("微软雅黑", 12, "bold"), bg="#2ecc71", fg="white", width=24, height=3, relief="raised", command=self.show_auto_input_page).pack(side="left", padx=25)
        
        ttk.Button(self, text="注销登录", command=self.show_login_page).pack(side="bottom", pady=50)

    def show_manual_input_page(self):
        self.clear_frame()
        ttk.Label(self, text="临床健康指标全特征采集面板 (21 Dimensions)", style="Header.TLabel").pack(pady=10)
        
        canvas = tk.Canvas(self, bg="#ffffff", borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#ffffff")
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", width=740)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True, padx=(30, 0), pady=10)
        scrollbar.pack(side="right", fill="y", padx=(0, 30), pady=10)
        
        self.input_vars = {}
        
        feature_configs = [
            ("1. 真实实际年龄 (Age / 岁):", "Age", "entry", "26"),
            ("2. 体质指数 (BMI):", "BMI", "entry", "24.5"),
            ("3. 近30天内坏心情/心理压力天数 (MentHlth / 0-30):", "MentHlth", "entry", "0"),
            ("4. 近30天内身体伤病/不适天数 (PhysHlth / 0-30):", "PhysHlth", "entry", "0"),
            ("5. 曾被告知患有高血压 (HighBP):", "HighBP", "combo", ["否", "是"]),
            ("6. 曾被告知患有高胆固醇 (HighChol):", "HighChol", "combo", ["否", "是"]),
            ("7. 过去5年内做过胆固醇检查 (CholCheck):", "CholCheck", "combo", ["是", "否"]),
            ("8. 一生中抽烟总数是否超过100支 (Smoker):", "Smoker", "combo", ["否", "是"]),
            ("9. 曾患有脑卒中/中风历史 (Stroke):", "Stroke", "combo", ["否", "是"]),
            ("10. 曾患有冠心病或心肌梗死 (HeartDiseaseorAttack):", "HeartDiseaseorAttack", "combo", ["否", "是"]),
            ("11. 近30天有常规身体锻炼 (PhysActivity):", "PhysActivity", "combo", ["是", "否"]),
            ("12. 每天坚持至少吃一次水果 (Fruits):", "Fruits", "combo", ["是", "否"]),
            ("13. 每天坚持至少吃一次蔬菜 (Veggies):", "Veggies", "combo", ["是", "否"]),
            ("14. 是否属于重度饮酒者 (HvyAlcoholConsump):", "HvyAlcoholConsump", "combo", ["否", "是"]),
            ("15. 拥有任何形式的医疗保险 (AnyHealthcare):", "AnyHealthcare", "combo", ["是", "否"]),
            ("16. 过去1年内是否曾因看病太贵而放弃就医 (NoDocbcCost):", "NoDocbcCost", "combo", ["否", "是"]),
            ("17. 卧床或独行上楼梯时存在行动障碍 (DiffWalk):", "DiffWalk", "combo", ["否", "是"]),
            ("18. 生物学性别 (Sex):", "Sex", "combo", ["男", "女"]),
            ("19. 主观评价目前的健康状况 (GenHlth):", "GenHlth", "combo", ["极好 (Excellent)", "很好 (Very Good)", "好 (Good)", "一般 (Fair)", "较差 (Poor)"]),
            ("20. 受教育文化程度 (Education):", "Education", "combo", ["大学毕业及以上", "大学/专科肄业", "高中毕业", "初中肄业/普通高中", "小学 (1-8年级)", "从未上学/仅幼儿园"]),
            ("21. 全家庭年收入阶层分布 (Income):", "Income", "combo", ["年薪 > $75k (级别8)", "环境偏高 $50k-$75k (级别7)", "中产阶层 $35k-$50k (级别6)", "温饱阶层 $25k-$35k (级别5)", "$20k-$25k (级别4)", "$15k-$20k (级别3)", "$10k-$15k (级别2)", "年薪 < $10k (级别1)"])
        ]
        
        for label_name, key, field_type, default in feature_configs:
            row = tk.Frame(scrollable_frame, bg="#ffffff")
            row.pack(fill="x", padx=15, pady=6)
            
            lbl = ttk.Label(row, text=label_name, font=("微软雅黑", 10), background="#ffffff", width=45, anchor="w")
            lbl.pack(side="left")
            
            if field_type == "entry":
                val_entry = ttk.Entry(row, font=("微软雅黑", 10))
                val_entry.pack(side="left", fill="x", expand=True)
                val_entry.insert(0, default)
                self.input_vars[key] = val_entry
            elif field_type == "combo":
                val_combo = ttk.Combobox(row, values=default, state="readonly", font=("微软雅黑", 9), width=25)
                val_combo.set(default[0])
                val_combo.pack(side="left")
                self.input_vars[key] = val_combo

        self.control_frame = tk.Frame(self, bg="#f4f6f9")
        self.control_frame.pack(fill="x", padx=30, pady=10)
        
        self.progress = ttk.Progressbar(self.control_frame, orient="horizontal", mode="determinate")
        self.btn_predict = ttk.Button(self.control_frame, text="🚀 启动 21-Dim 智能预测", command=self.start_manual_prediction)
        self.btn_predict.pack(side="right", padx=5)
        
        ttk.Button(self, text="⬅️ 返回模式选择", command=self.show_selection_page).pack(side="bottom", pady=5)

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
            
            print("\n[DEBUG] 随机森林期望维度:", model.n_features_in_)
            print("[DEBUG] 软件实际输入维度:", processed_vector.shape[1])
            
            # 💡 【核心新增】：使用 predict_proba 获取各个分类的概率矩阵
            probabilities = model.predict_proba(processed_vector)[0]  # 返回形如 [prob_0, prob_1, prob_2]
            prediction = int(np.argmax(probabilities))  # 概率最大的即为预测类别
            confidence = probabilities[prediction] * 100  # 转化为百分比
            
            result_map = {
                0: "🎉 健康状况优良，未检测到显著糖尿病风险。", 
                1: "⚠️ 警告提示：处于糖尿病前期（Prediabetes），高危非线性倾向，请立刻改善生活和饮食习惯！", 
                2: "🚨 红色诊断：已处于高危糖尿病（Diabetes）表征状态，建议即刻安排医院临床检查。"
            }
            final_conclusion = result_map.get(prediction, "数据未知")
            
            # 丰富可视化日志：打印全部类别的百分比概率
            prob_details = f" -> 正常状态概率: {probabilities[0]*100:.2f}%\n" \
                           f" -> 糖尿病前期概率: {probabilities[1]*100:.2f}%\n" \
                           f" -> 糖尿病状态概率: {probabilities[2]*100:.2f}%"
            
            log_details = "\n".join([f" -> {col_names[m]}: {processed_vector[0][m]}" for m in range(len(col_names))])
            
            messagebox.showinfo(
                "智能化分析报告", 
                f"【21维医学矩阵解析成功】\n\n"
                f"[模型多概率置信度解析]:\n{prob_details}\n\n"
                f"【模型最终预测结论】:\n{final_conclusion} (置信度: {confidence:.2f}%)\n\n"
                f"[底层输入向量序列 (CDC标准化后)]:\n{log_details}"
            )
            
        except FileNotFoundError:
            messagebox.showerror("阻断", f"本地未发现 [{self.model_path}] 文件，请先运行随机森林训练代码。")
        except Exception as e:
            messagebox.showerror("错误", f"预测异常: {str(e)}")
            
        self.progress.pack_forget()
        self.progress['value'] = 0
        self.btn_predict.config(state="normal")

    def show_auto_input_page(self):
        self.clear_frame()
        ttk.Label(self, text="体检中心 Excel 报告批处理面板", style="Header.TLabel").pack(pady=50)
        
        card_frame = tk.Frame(self, bg="#ffffff", bd=2, relief="groove")
        card_frame.pack(pady=20, padx=60, fill="both", expand=True)
        
        ttk.Label(card_frame, text="【批量导入说明】\n上传的 Excel 模板文件中必须包含完整的 21 个特征列（如 HighBP, HighChol, Age, BMI... 等，列名需与 CSV 表头完全一致）。\n系统将自动批量调用本地随机森林分类器进行推理，并将结果与概率回填。", 
                  font=("微软雅黑", 10), justify="left", background="#ffffff").pack(pady=25, padx=20)
        
        file_row = tk.Frame(card_frame, bg="#ffffff")
        file_row.pack(fill="x", padx=20, pady=10)
        self.file_path_entry = ttk.Entry(file_row, font=("微软雅黑", 10), state="readonly")
        self.file_path_entry.pack(side="left", fill="x", expand=True, padx=5)
        
        ttk.Button(file_row, text="📁 浏览文件", command=self.browse_excel_file).pack(side="left", padx=5)
        
        self.auto_control_frame = tk.Frame(card_frame, bg="#ffffff")
        self.auto_control_frame.pack(fill="x", padx=20, pady=20)
        
        self.auto_progress = ttk.Progressbar(self.auto_control_frame, orient="horizontal", mode="determinate")
        self.btn_auto_predict = ttk.Button(self.auto_control_frame, text="📊 启动批量推理", state="disabled", command=self.start_auto_prediction)
        self.btn_auto_predict.pack(side="right", padx=5)
        
        ttk.Button(self, text="⬅️ 返回模式选择", command=self.show_selection_page).pack(side="bottom", pady=20)

    def browse_excel_file(self):
        file_selected = filedialog.askopenfilename(filetypes=[("Excel 电子表格", "*.xlsx *.xls")])
        if file_selected:
            self.file_path_entry.config(state="normal")
            self.file_path_entry.delete(0, tk.END)
            self.file_path_entry.insert(0, file_selected)
            self.file_path_entry.config(state="readonly")
            self.btn_auto_predict.config(state="normal")

    def start_auto_prediction(self):
        self.btn_auto_predict.config(state="disabled")
        self.auto_progress.pack(side="fill", fill="x", expand=True, padx=5)
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
                raise ValueError("Excel 表格内未完全包含 21 个标准特征列，请检查列名拼写与大小写！")
            
            X_batch = df_excel[expected_cols].values
            
            # 💡 【Excel批量新增】：不仅预测结果，同时把各个类别的概率值写回 Excel 报表中
            preds = model.predict(X_batch)
            probs = model.predict_proba(X_batch)
            
            df_excel['Model_Prediction_Result'] = preds
            df_excel['Prob_Normal (%)'] = np.round(probs[:, 0] * 100, 2)
            df_excel['Prob_Prediabetes (%)'] = np.round(probs[:, 1] * 100, 2)
            df_excel['Prob_Diabetes (%)'] = np.round(probs[:, 2] * 100, 2)
            df_excel['Max_Confidence (%)'] = np.round(np.max(probs, axis=1) * 100, 2)
            
            save_output_path = excel_path.replace(".xlsx", "_预测报告输出.xlsx")
            df_excel.to_excel(save_output_path, index=False)
            
            messagebox.showinfo("批处理完成", f"批量诊断成功并完成多维度概率回填！\n\n读取样本总数: {len(df_excel)} 行\n结果已自动导出至:\n{save_output_path.split('/')[-1]}")
        except FileNotFoundError:
            messagebox.showerror("阻断", f"本地未发现 [{self.model_path}] 模型文件。")
        except Exception as e:
            messagebox.showerror("错误", f"Excel 推理失败:\n{str(e)}")
            
        self.auto_progress.pack_forget()
        self.auto_progress['value'] = 0
        self.btn_auto_predict.config(state="normal")


if __name__ == "__main__":
    app = DiabetesPredictorApp()
    app.mainloop()