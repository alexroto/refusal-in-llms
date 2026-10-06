pip install -r requirements.txt
export HF_TOKEN=""
python data/generate_harmful_data.py
python data/generate_harmless_data.py