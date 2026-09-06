python3 multiturn_ner_benchmark_batched.py \
  --mode llm_zero \
  --hard

python3 multiturn_ner_benchmark_batched.py \
  --mode llm_eval \
  --hard \
  --eval-batch-size 32

python3 encoder.py \
  --mode train \
  --max-train 1000 \
  --train-batch-size 12 \
  --epochs 1

python3 encoder_dst_fixed.py \
  --mode eval \
  --hard \
  --eval-batch-size 100
