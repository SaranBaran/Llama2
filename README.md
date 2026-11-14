# Min-Llama (CS2470 HW4)

A small Llama 2-style language model (42M parameters) built from scratch for a Brown CS2470 assignment. It's pretrained on children's stories and can generate text, classify sentiment, and be fine-tuned.

## Setup
```
conda activate csci2470
bash setup.sh        # Apple Silicon: ./setup_silicon.sh
```
Downloads the pretrained weights `stories42M.pt`.

## Use it

**Generate text** (saves to `generated-sentence-temp-0.txt` and `generated-sentence-temp-1.txt`):
```
python run_llama.py --option generate
```

**Zero-shot sentiment** (swap `sst` for `cfimdb` to use the other dataset):
```
python run_llama.py --option prompt --batch_size 10 --train data/sst-train.txt --dev data/sst-dev.txt --test data/sst-test.txt --label-names data/sst-label-mapping.json --dev_out sst-dev-prompting-output.txt --test_out sst-test-prompting-output.txt
```

**Fine-tune** (for `cfimdb`, swap the paths and use `--batch_size 10`; add `--use_gpu` if you have a GPU):
```
python run_llama.py --option finetune --epochs 5 --lr 2e-5 --batch_size 80 --train data/sst-train.txt --dev data/sst-dev.txt --test data/sst-test.txt --label-names data/sst-label-mapping.json --dev_out sst-dev-finetuning-output.txt --test_out sst-test-finetuning-output.txt
```

## Write a short story
This is a base model, not a chatbot. It continues text and doesn't follow instructions, so give it the start of a story, such as "Once upon a time, there was a little girl named Lily who".

1. `cp run_llama.py story.py`
2. In `story.py`, set `prefix` to your opening and raise `max_new_tokens` (e.g. 200).
3. Run:
```
python story.py --option generate --generated_sentence_low_temp_out story-temp-0.txt --generated_sentence_high_temp_out story-temp-1.txt --seed 7
```
Temperature 0 gives the same story each time. Temperature 1 gives a random one, so change `--seed` to vary it.

## Tests
```
python rope_test.py
python sanity_check.py
python optimizer_test.py
```
