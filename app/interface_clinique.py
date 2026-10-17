"""Command-line clinical interface: a doctor types the patient's values.

Start:  python -m app.interface_clinique
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.api import load_predictor  # noqa: E402

QUESTIONS = [
    ("age", "Age (years)"),
    ("temperature", "Temperature (°C)"),
    ("cough", "Cough? (1 = yes, 0 = no)"),
    ("fatigue", "Fatigue? (1 = yes, 0 = no)"),
    ("loss_smell", "Loss of smell? (1 = yes, 0 = no)"),
    ("contact", "Contact with an infected person? (1 = yes, 0 = no)"),
    ("oxygen_saturation", "Oxygen saturation (%)"),
    ("heart_rate", "Heart rate (bpm)"),
    ("wbc_count", "White blood cells (G/L)"),
    ("crp", "CRP (mg/L)"),
]


def ask_float(label):
    while True:
        try:
            return float(input(f"  {label}: ").replace(",", "."))
        except ValueError:
            print("  -> please enter a number")


def main():
    predictor = load_predictor()
    print("=== Clinical interface - virus_diag ===  (Ctrl+C to quit)")
    while True:
        print("\nNew patient:")
        patient = {key: ask_float(label) for key, label in QUESTIONS}
        prob = predictor.probability(patient)
        print(f"\n  >>> Diagnosis: {predictor.diagnose(patient)}  (probability = {prob:.1%})")
        if input("\nAnother patient? (y/n): ").strip().lower() != "y":
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye.")
