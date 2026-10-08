pip install -r requirements.txt
export HF_TOKEN=""
python data/generate_harmful_data.py
python data/generate_harmless_data.py
python baseline_refusal.py --model="llama2" --dataset="harmful"
python baseline_refusal.py --model="qwen" --dataset="harmful"
python baseline_refusal.py --model="vicuna" --dataset="harmful"
python baseline_refusal.py --model="llama2" --dataset="harmless"
python baseline_refusal.py --model="qwen" --dataset="harmless"
python baseline_refusal.py --model="vicuna" --dataset="harmless"