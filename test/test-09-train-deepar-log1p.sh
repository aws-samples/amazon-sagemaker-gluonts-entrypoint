#!/usr/bin/env bash

SRC=src
INPUT=refdata

echo -e '\nDeepAR...'
python $SRC/entrypoint.py --s3_dataset $INPUT \
    --y_transform log1p \
    --algo gluonts.model.deepar.DeepAREstimator \
    --trainer gluonts.trainer.Trainer \
    --trainer.epochs 3 \
    --distr_output gluonts.distribution.gaussian.GaussianOutput \
    --use_feat_static_cat True \
    --cardinality '[5]' \
    --prediction_length 3 #\
#    2>&1 | egrep --color=always -i 'prediction_length|freq|epochs|\.[a-zA-Z]+Estimator|$'
