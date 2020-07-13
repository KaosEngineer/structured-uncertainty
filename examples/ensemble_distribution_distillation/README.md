# Generating ensemble predictions

First, generate predictions on train data (num_shards/shard_id are used to distribute the inference between several GPUs)
```
f"python3 {source_dir}/fairseq-py/generate.py "
                f"{input_dir}/data "
                f"--path {model_paths} "
                f"--max-tokens 1024 "
                f"--fp16 "
                f"--nbest 1 "
                f"--gen-subset train "
                f"--num-shards {args.num_shards} "
                f"--shard-id {args.shard_id} "
                f"> ensemble_predictions.out "
```

Then prepare the data for preprocessing with
```
f"python3 {source_dir}/fairseq-py/examples/ensemble_distribution_distillation/extract_predictions.py "
                    f"--input ensemble_predictions.out "
                    f"--output {output_dir}/ensemble_predictions_split_{args.shard_id} "
                    f"--srclang en "
                    f"--tgtlang de "
```

Then preprocess the training dataset and gather it along with reference outputs for train/validation/test data:
```
cat ${INPUT_PATH}/ensemble_predictions_split_*.de > ensemble_predictions_split.de;
cat ${INPUT_PATH}/ensemble_predictions_split_*.en > ensemble_predictions_split.en;
rm ${INPUT_PATH}/ensemble_predictions_split_*.de;
rm ${INPUT_PATH}/ensemble_predictions_split_*.en;

f"python3 {source_dir}/fairseq-py/preprocess.py "
                       f"--trainpref ensemble_predictions_split "
                       f"--destdir {output_dir}/data_generated "
                       f"--source-lang en "
                       f"--target-lang de "
                       f"--srcdict {input_dir}/data/dict.en.txt "
                       f"--tgtdict {input_dir}/data/dict.de.txt "
                       f"--workers 32 "

cp -r ${INPUT_PATH}/* ${TMP_OUTPUT_PATH}

PARA_DATA=${TMP_OUTPUT_PATH}/data
GEN_DATA=${TMP_OUTPUT_PATH}/data_generated
COMB_DATA=${TMP_OUTPUT_PATH}/data_combined
mkdir -p $COMB_DATA
cd $COMB_DATA

PARA_DATA="../data"
GEN_DATA="../data_generated"

for LANG in en de; do \
    ln -rs ${PARA_DATA}/dict.$LANG.txt ${COMB_DATA}/dict.$LANG.txt; \
    for EXT in bin idx; do \
        ln -rs ${PARA_DATA}/train.en-de.$LANG.$EXT ${COMB_DATA}/train.en-de.$LANG.$EXT; \
        ln -rs ${GEN_DATA}/train.en-de.$LANG.$EXT ${COMB_DATA}/train1.en-de.$LANG.$EXT; \
        ln -rs ${PARA_DATA}/valid.en-de.$LANG.$EXT ${COMB_DATA}/valid.en-de.$LANG.$EXT; \
        ln -rs ${PARA_DATA}/test.en-de.$LANG.$EXT ${COMB_DATA}/test.en-de.$LANG.$EXT; \
    done; \
done
```

In order to train on ensemble predictions, specify either `f"{input_dir}/data_combined --upsample-primary 0 "` or `f"{input_dir}/data_generated"` as input data for train.py.

# Training

Example training script:

```
f"python3 {source_dir}/fairseq-py/train.py "
            f"{input_dir}/data_combined "
            f"--upsample-primary 0 "
            f"--arch dirichlet_transformer_wmt_en_de_big "
            f"--tensorboard-logdir {logs_path} "
            f"--share-decoder-input-output-embed "
            f"--num-workers 4 "
            f"--optimizer adam "
            f"--adam-betas '(0.9, 0.98)' "
            f"--clip-norm 10.0 "
            f"--lr 5e-4 "
            f"--lr-scheduler inverse_sqrt "
            f"--warmup-updates 4000 "
            f"--dropout 0.3 "
            f"--weight-decay 0.0001 "
            f"--criterion dirichlet_mediator_distillation "
            f"--target-concentration epkl "
            f"--model-offset 1 "
            f"--target-offset 1 "
            f"--task distillation "
            f"--ensemble-paths {model_paths} "
            f"--max-tokens 256 "
            f"--update-freq 32 "
            f"--save-dir {snapshot_dir} "
            f"--max-update 50000 "
            f"--anneal-start 0 "
            f"--anneal-end 1 "
            f"--init-temp 1 "
            f"--keep-last-epochs 10 "
            f"--valid-subset valid,test "
            f"--seed {hyp['seed']} "
            f"--ddp-backend=no_c10d "
            f"--user-dir {source_dir}/fairseq-py/examples/ensemble_distribution_distillation "
```
You can:
* Replace `dirichlet_transformer_wmt_en_de_big` with `transformer_wmt_en_de_big` to switch to a regular parametrization
* Change `dirichlet_mediator_distillation` with `sequence_distribution_distillation` or change `--target-concentration`
* Remove `--model-offset` or `--target-offset`
* Add `--topk-loss K`
* Change `init-temp`, `final-temp`, `anneal-start`, `anneal-end`, `init-xent-weight`, `final-xent-weight` (make sure to change `xent-type` when using Dirichlet distillation)