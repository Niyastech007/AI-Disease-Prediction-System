import tkinter as tk
from tkinter import messagebox, filedialog
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Global model and accuracy
model = None
accuracy = 0.0

# -------------------------
# Train Model Function
# -------------------------
def train_model_from_csv():
    global model, accuracy
    try:
        file_path = filedialog.askopenfilename(title="Select CSV File", filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return

        df = pd.read_csv(file_path)

        required_columns = {"Age", "TopBP", "BottomBP", "Sugar", "ChestPain", "BMI", "Disease"}
        if not required_columns.issubset(df.columns):
            messagebox.showerror("Error", f"CSV must contain columns: {required_columns}")
            return

        # Convert ChestPain Yes/No → 1/0
        df["ChestPain"] = df["ChestPain"].map({"Yes": 1, "No": 0})

        X = df[["Age", "TopBP", "BottomBP", "Sugar", "BMI", "ChestPain"]]
        y = df["Disease"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        accuracy = accuracy_score(y_test, model.predict(X_test))

        messagebox.showinfo("Training Complete", f"Model trained successfully!\nAccuracy: {accuracy*100:.2f}%")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load/train model: {str(e)}")

# -------------------------
# Input Validation
# -------------------------
def validate_inputs(age, top_bp, bottom_bp, sugar, chest, bmi):
    if not (0 < age < 120):
        return "Age should be between 1 and 120."
    if not (90 <= top_bp <= 180):
        return "Top BP should be between 90 and 180."
    if not (60 <= bottom_bp <= 120):
        return "Bottom BP should be between 60 and 120."
    if not (50 <= sugar <= 500):
        return "Sugar Level should be between 50 and 500."
    if chest not in [0, 1]:
        return "Chest Pain must be 0 (No) or 1 (Yes)."
    if not (10 <= bmi <= 60):
        return "BMI should be between 10 and 60."
    return None

# -------------------------
# Severity Calculation
# -------------------------
def calculate_severity(top_bp, bottom_bp, sugar, bmi, chest):
    score = 0
    if top_bp > 140 or bottom_bp > 90:
        score += 1
    if sugar > 200:
        score += 1
    if bmi > 30:
        score += 1
    if chest == 1:
        score += 1

    if score >= 3:
        return "High"
    elif score == 2:
        return "Medium"
    else:
        return "Low"

# -------------------------
# Health Advice & Diet Plan
# -------------------------
def get_advice_and_diet(disease, severity):
    disease = disease.lower()
    advice, diet = "", ""

    if disease == "heart disease":
        if severity == "High":
            advice = "Consult cardiologist immediately; follow low-fat diet."
            diet = "Low-fat, high fiber, avoid fried foods, more fruits & veggies."
        elif severity == "Medium":
            advice = "Regular checkups & moderate exercise recommended."
            diet = "Balanced diet with limited red meat & processed foods."
        else:
            advice = "Maintain healthy lifestyle."
            diet = "Balanced diet and regular exercise."

    elif disease == "diabetes":
        if severity == "High":
            advice = "Monitor sugar closely and consult a doctor."
            diet = "Low sugar, high fiber, avoid sweets & white rice."
        elif severity == "Medium":
            advice = "Control diet and check sugar regularly."
            diet = "Balanced diet, portion control, include vegetables."
        else:
            advice = "Maintain healthy lifestyle."
            diet = "Balanced diet and regular exercise."

    elif disease == "hypertension":
        if severity == "High":
            advice = "Immediate doctor consultation recommended."
            diet = "Low salt diet, avoid junk foods, more fruits & veggies."
        elif severity == "Medium":
            advice = "Regular BP monitoring and mild exercise needed."
            diet = "Reduce salt intake, eat potassium-rich foods."
        else:
            advice = "Maintain normal BP with healthy diet."
            diet = "Balanced diet, low salt."

    elif disease == "obesity":
        if severity == "High":
            advice = "Serious lifestyle changes required; consult nutritionist."
            diet = "Strict calorie control, avoid fried & sugary foods."
        elif severity == "Medium":
            advice = "Regular exercise and portion control needed."
            diet = "High protein, low carb diet."
        else:
            advice = "Maintain current weight with balanced food."
            diet = "Balanced diet, regular activity."

    elif disease == "asthma":
        advice = "Avoid triggers; use inhalers as prescribed."
        diet = "Eat fruits & veggies; avoid allergens."

    else:
        advice = "You seem healthy. Maintain lifestyle."
        diet = "Balanced diet with proper nutrients."

    return advice, diet

# -------------------------
# Real-time Indicator Update
# -------------------------
def update_indicator(entry, indicator, min_val, max_val, high_risk_val=None):
    try:
        value = float(entry.get())
        if high_risk_val and value > high_risk_val:
            color = "red"
        elif value < min_val or value > max_val:
            color = "yellow"
        else:
            color = "green"
    except:
        color = "grey"
    indicator.config(bg=color)
    update_summary_panel()

# -------------------------
# Update Summary Panel
# -------------------------
def update_summary_panel():
    try:
        age = int(age_entry.get())
        top_bp = int(top_bp_entry.get())
        bottom_bp = int(bottom_bp_entry.get())
        sugar = int(sugar_entry.get())
        chest = int(chest_entry.get())
        bmi = float(bmi_entry.get())
    except:
        summary_label.config(text="Enter valid inputs to see summary...")
        return

    error = validate_inputs(age, top_bp, bottom_bp, sugar, chest, bmi)
    if error:
        summary_label.config(text="Waiting for valid inputs...")
        return

    if model:
        input_data = pd.DataFrame([[age, top_bp, bottom_bp, sugar, bmi, chest]],
                                  columns=["Age", "TopBP", "BottomBP", "Sugar", "BMI", "ChestPain"])
        prediction = model.predict(input_data)[0]
        sev = calculate_severity(top_bp, bottom_bp, sugar, bmi, chest)
        advice, diet = get_advice_and_diet(prediction, sev)
        summary_text = (f"Predicted Disease: {prediction}\n"
                        f"Severity Level: {sev}\n"
                        f"BP: {top_bp}/{bottom_bp} mmHg\n"
                        f"Model Accuracy: {accuracy*100:.2f}%\n"
                        f"Health Advice: {advice}\n"
                        f"Diet Plan: {diet}")
        summary_label.config(text=summary_text)
    else:
        summary_label.config(text="Load and train model to see predictions...")

# -------------------------
# Clear Function
# -------------------------
def clear_fields():
    for entry, placeholder, indicator in zip(entries, placeholders, indicators):
        entry.delete(0, tk.END)
        entry.insert(0, placeholder)
        entry.config(fg="grey")
        indicator.config(bg="grey")
    summary_label.config(text="Summary panel will show predictions here...")

# -------------------------
# Placeholder functions
# -------------------------
def on_entry_click(event, entry, placeholder):
    if entry.get() == placeholder:
        entry.delete(0, "end")
        entry.config(fg="black")

def on_focusout(event, entry, placeholder):
    if entry.get() == "":
        entry.insert(0, placeholder)
        entry.config(fg="grey")

# -------------------------
# Tooltip Class
# -------------------------
class ToolTip(object):
    def __init__(self, widget, text=''):
        self.widget = widget
        self.text = text
        self.tipwindow = None
        widget.bind("<Enter>", self.show_tip)
        widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, event=None):
        if self.tipwindow or not self.text:
            return
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, justify='left',
                         background="#ffffe0", relief='solid', borderwidth=1,
                         font=("Helvetica", 10))
        label.pack(ipadx=5, ipady=5)

    def hide_tip(self, event=None):
        if self.tipwindow:
            self.tipwindow.destroy()
        self.tipwindow = None

# -------------------------
# Tkinter GUI
# -------------------------
root = tk.Tk()
root.title("AI Disease Prediction System")
root.geometry("650x800")
root.configure(bg="#f0f8ff")

title_label = tk.Label(root, text="AI Disease Prediction System", font=("Helvetica", 18, "bold"),
                       bg="#f0f8ff", fg="#333")
title_label.pack(pady=20)

load_btn = tk.Button(root, text="Load Dataset (CSV)", command=train_model_from_csv,
                     bg="#1e90ff", fg="white", font=("Helvetica", 12, "bold"), padx=10, pady=5)
load_btn.pack(pady=10)

input_frame = tk.Frame(root, bg="#f0f8ff")
input_frame.pack(pady=10)

labels_info = [
    ("Age (1-90)", "Enter age in years", 1, 120, None, "Age in years; adult range 18-65 recommended."),
    ("BP Top (e.g., 120)", "Enter top number of BP", 90, 180, 140, "Top BP = higher number, normal: 90-120."),
    ("BP Bottom (e.g., 70)", "Enter bottom number of BP", 60, 120, 90, "Bottom BP = lower number, normal: 60-80."),
    ("Sugar Level (50-500)", "Enter sugar level in mg/dL", 50, 500, 200, "Fasting normal: 70-100 mg/dL."),
    ("Chest Pain (0=No,1=Yes)", "0 for No, 1 for Yes", 0, 1, 1, "0 = No chest pain, 1 = Experiencing chest pain."),
    ("BMI (10-60)", "Enter BMI value", 10, 60, 30, "Normal BMI: 18.5-24.9; above 30 considered obese.")
]

entries, placeholders, indicators = [], [], []

for i, (label_text, placeholder, min_val, max_val, high_risk_val, tip_text) in enumerate(labels_info):
    lbl = tk.Label(input_frame, text=label_text, font=("Helvetica", 12), bg="#f0f8ff", anchor="w")
    lbl.grid(row=i, column=0, sticky="w", pady=5, padx=5)

    ent = tk.Entry(input_frame, font=("Helvetica", 12), fg="grey")
    ent.grid(row=i, column=1, pady=5, padx=5)
    ent.insert(0, placeholder)
    ent.bind('<FocusIn>', lambda e, entry=ent, ph=placeholder: on_entry_click(e, entry, ph))
    ent.bind('<FocusOut>', lambda e, entry=ent, ph=placeholder: on_focusout(e, entry, ph))
    entries.append(ent)
    placeholders.append(placeholder)

    # Indicator box
    indicator = tk.Label(input_frame, text=" ", width=2, bg="grey")
    indicator.grid(row=i, column=2, padx=5)
    indicators.append(indicator)

    # Tooltip
    ToolTip(ent, tip_text)

    # Real-time validation
    def on_key(event, entry=ent, ind=indicator, min_v=min_val, max_v=max_val, high_v=high_risk_val):
        update_indicator(entry, ind, min_v, max_v, high_v)
    ent.bind('<KeyRelease>', on_key)

age_entry, top_bp_entry, bottom_bp_entry, sugar_entry, chest_entry, bmi_entry = entries

# Buttons Frame
btn_frame = tk.Frame(root, bg="#f0f8ff")
btn_frame.pack(pady=10)

predict_btn = tk.Button(btn_frame, text="Predict Disease", command=lambda: update_summary_panel(),
                        bg="#28a745", fg="white", font=("Helvetica", 12, "bold"), padx=10, pady=5)
predict_btn.grid(row=0, column=0, padx=10)

clear_btn = tk.Button(btn_frame, text="Clear Fields", command=clear_fields,
                      bg="#dc3545", fg="white", font=("Helvetica", 12, "bold"), padx=10, pady=5)
clear_btn.grid(row=0, column=1, padx=10)

# Summary Panel
summary_label = tk.Label(root, text="Summary panel will show predictions here...",
                         font=("Helvetica", 12), bg="#e6f2ff", fg="#333", justify="left",
                         wraplength=600, relief="sunken", bd=2, padx=10, pady=10)
summary_label.pack(pady=20, fill="both")

root.mainloop()
