pip install -r requirements.txt
export HF_TOKEN=""
python data/generate_harmful_data.py
python data/generate_harmless_data.py
hf cache rm model/MaartenGr/BERTopic_Wikipedia -y
python baseline_refusal.py --model="llama2" --dataset="harmful"
python baseline_refusal.py --model="llama2" --dataset="harmless"
hf cache rm model/meta-llama/Llama-2-7b-chat-hf -y
python baseline_refusal.py --model="qwen" --dataset="harmful"
python baseline_refusal.py --model="qwen" --dataset="harmless"
hf cache rm model/Qwen/Qwen2.5-7B-Instruct -y
python baseline_refusal.py --model="vicuna" --dataset="harmful"
python baseline_refusal.py --model="vicuna" --dataset="harmless"
hf cache rm model/lmsys/vicuna-7b-v1.5 -y