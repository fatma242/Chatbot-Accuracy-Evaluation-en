import pandas as pd
import requests
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib.backends.backend_pdf import PdfPages

# === CONFIG ===
CSV_PATH = "downsampled_dataset.csv"
API_KEY = "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"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "deepseek/deepseek-r1:free"
VALID_LABELS = ["Anxiety", "Stress", "Depression", "Suicidal", "Normal"]

# === PROMPT ===
prompt = """You are Stoic AI, an AI life coach. Classify the user's mental state according to ICD-11 and DSM-5 criteria into:
Anxiety: Recommend breathing exercises, meditation, or yoga. Link: https://www.youtube.com/@YogaWithRawda
Stress: Suggest relaxation techniques or time management tips.
Depression: Offer inspirational stories or mood-boosting support.
Suicidal: Direct to crisis hotlines and websites: https://www.shezlong.com/ar, https://befrienders.org/ar/, https://www.betterhelp.com/get-started/
Normal: Cheer them up and reinforce positive emotions.

Start your response with:
Classification: [Mental State Label(s)]
Then write the supportive message tailored to the classified state(s).

Respond with emotional support tailored to the user's mental state, without explaining the classification process. Offer encouragement and recommend professional help if necessary.
Always respond in the user's language or accent. Default to English if unsure.
Stoic AI does not curse, use obscene, racist, or trendy slang words. If the user makes an offensive, racist, or vulgar request, Stoic AI will politely refuse, saying: "I'm here to support you positively, but I can't respond to that request." Always reply in the user's language or accent.
Classify the user's mental state into at most two categories from: Anxiety, Stress, Depression, Suicidal, Normal.
"""

def call_chatbot(message_text):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": message_text}
        ],
        "top_p": 1,
        "temperature": 0.7,
        "repetition_penalty": 1
    }

    try:
        response = requests.post(API_URL, headers=headers, json=data, timeout=30)
        content = response.json()["choices"][0]["message"]["content"]
        match = re.search(r"Classification:\s*([^\n]+)", content, re.IGNORECASE)
        if match:
            raw_labels = match.group(1)
            predicted = [label.strip().capitalize() for label in raw_labels.split(",")]
            return [label for label in predicted if label in VALID_LABELS][:2], content.strip()
        else:
            return [], content.strip()
    except Exception as e:
        print("Error:", e)
        return [], str(e)

def evaluate():
    df = pd.read_csv(CSV_PATH)
    true_labels = []
    predicted_labels = []
    responses_log = []

    acc1_correct = 0
    acc2_correct = 0
    total = 0

    for i, row in df.iterrows():
        msg = row["text"]
        true = row["label"].strip().capitalize()
        print(f"Sending: {msg}")
        predicted, full_response = call_chatbot(msg)
        print(f"True: {true} | Predicted: {predicted}")

        responses_log.append({
            "text": msg,
            "true_label": true,
            "predicted_labels": ", ".join(predicted),
            "full_response": full_response
        })

        if true in VALID_LABELS and any(p in VALID_LABELS for p in predicted):
            total += 1
            if predicted and predicted[0] == true:
                acc1_correct += 1
            if true in predicted:
                acc2_correct += 1

            true_labels.append(true)
            predicted_labels.append(predicted[0] if predicted else "Unknown")
        else:
            print(f"Skipping: true={true}, predicted={predicted}")

    acc1 = acc1_correct / total if total > 0 else 0
    acc2 = acc2_correct / total if total > 0 else 0

    report = classification_report(true_labels, predicted_labels, labels=VALID_LABELS)
    cm = confusion_matrix(true_labels, predicted_labels, labels=VALID_LABELS)

    # === PDF REPORT ===
    with PdfPages("Stoic_Evaluation_Report.pdf") as pdf:
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=VALID_LABELS, yticklabels=VALID_LABELS)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        pdf.savefig()
        plt.close()

        # Text report
        plt.figure(figsize=(8.5, 11))
        plt.axis("off")
        text = (
            f"Stoic Evaluation Report\n\n"
            f"Accuracy@1: {acc1:.2%} ({acc1_correct}/{total})\n"
            f"Accuracy@2: {acc2:.2%} ({acc2_correct}/{total})\n\n"
            f"Classification Report:\n\n{report}"
        )
        plt.text(0, 1, text, fontsize=10, va='top', family='monospace')
        pdf.savefig()
        plt.close()

    # === SAVE RESPONSES ===
    pd.DataFrame(responses_log).to_csv("all_responses.csv", index=False)
    print("📁 Saved all responses to: all_responses.csv")
    print("📄 PDF report saved to: Stoic_Evaluation_Report.pdf")
    print(f"✅ Accuracy@1 = {acc1:.2%} ({acc1_correct}/{total})")
    print(f"✅ Accuracy@2 = {acc2:.2%} ({acc2_correct}/{total})")

    # === SAVE MISCLASSIFIED CASES ===
    misclassified = [row for row in responses_log if row["true_label"] not in row["predicted_labels"]]
    pd.DataFrame(misclassified).to_csv("misclassified_cases.csv", index=False)
    print("📁 Saved misclassified cases to: misclassified_cases.csv")


if __name__ == "__main__":
    evaluate()
