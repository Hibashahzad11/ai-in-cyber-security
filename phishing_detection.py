# RULE-BASED PHISHING DETECTION SYSTEM COMPARISON
import pandas as pd
# Step 1: Load Dataset
df = pd.read_csv("phishing.csv")
print("✅ Dataset Loaded Successfully!\n")
print(df.head())

# Step 2: Define First Rule-Based Algorithm

# (Rule-based logic using multiple conditions)
class PhishingRuleBasedAI:
    def __init__(self):
        self.rules = [
            ("URL_Length", -1, "Suspiciously long URL → Likely Phishing"),
            ("having_IP_Address", 1, "Contains IP in URL → Likely Phishing"),
            ("Shortining_Service", 1, "Uses Short URL → Likely Phishing"),
            ("Prefix_Suffix", 1, "Uses '-' in domain → Likely Phishing"),
        ]
    
    def diagnose(self, row):
        phishing_signs = 0
        for feature, value, message in self.rules:
            if feature in row and row[feature] == value:
                phishing_signs += 1
        if phishing_signs >= 2:
            return "Phishing Website"
        else:
            return "Legitimate Website"
        
# Step 3: Define Second Rule-Based Algorithm (Simplified Scoring)
class PhishingScoreAI:
    def diagnose(self, row):
        score = 0
        if "URL_Length" in row and row["URL_Length"] == -1:
            score += 2
        if "having_IP_Address" in row and row["having_IP_Address"] == 1:
            score += 2
        if "Shortining_Service" in row and row["Shortining_Service"] == 1:
            score += 1
        if "Prefix_Suffix" in row and row["Prefix_Suffix"] == 1:
            score += 1
        
        if score >= 4:
            return "Phishing Website"
        else:
            return "Legitimate Website"
        
# Step 4: Create Instances of Both Algorithms
ai1 = PhishingRuleBasedAI()
ai2 = PhishingScoreAI()

# Step 5: Apply Algorithms to Dataset
df["AI1_Diagnosis"] = df.apply(lambda x: ai1.diagnose(x), axis=1)
df["AI2_Diagnosis"] = df.apply(lambda x: ai2.diagnose(x), axis=1)

# Convert dataset Result column to readable labels
df["True_Label"] = df["Result"].apply(lambda x: "Phishing Website" if x == -1 else "Legitimate Website")

# Step 6: Evaluate Accuracy
df["AI1_Correct"] = df["AI1_Diagnosis"] == df["True_Label"]
df["AI2_Correct"] = df["AI2_Diagnosis"] == df["True_Label"]

ai1_accuracy = df["AI1_Correct"].mean()
ai2_accuracy = df["AI2_Correct"].mean()

# Step 7: Display Results
print("\n🔍 Sample Comparison:")
print(df[["URL_Length", "having_IP_Address", "Shortining_Service", "Prefix_Suffix", "Result", "AI1_Diagnosis", "AI2_Diagnosis"]].head())

print(f"\n✅ PhishingRuleBasedAI Accuracy: {ai1_accuracy * 100:.2f}%")
print(f"✅ PhishingScoreAI Accuracy: {ai2_accuracy * 100:.2f}%")
import matplotlib.pyplot as plt

models = ["URL Rule-Based", "Content Rule-Based"]
accuracy = [44.60, 44.54]

plt.figure(figsize=(10,6))
bars = plt.bar(models, accuracy)

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.3, 
             f"{height:.2f}%", ha='center', fontsize=12)

plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()