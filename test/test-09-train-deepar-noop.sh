#!/usr/bin/env bash

SRC=src
INPUT=refdata

echo -e '\nDeepAR...'
python $SRC/train.py --s3_dataset $INPUT \
    --algo gluonts.model.deepar.DeepAREstimator \
    --trainer.__class__ gluonts.trainer.Trainer \
    --trainer.epochs 10 \
    --distr_output.__class__ gluonts.distribution.gaussian.GaussianOutput \
    --use_feat_static_cat True \
    --cardinality '[5]' \
    --prediction_length 3 #\
#    2>&1 | egrep --color=always -i 'prediction_length|freq|epochs|\.[a-zA-Z]+Estimator|$'
