#!/bin/bash

SRC=src
INPUT=refdata

python $SRC/train.py --stop_before train \
    --s3_dataset $INPUT \
    --distr_output.__class__ gluonts.distribution.gaussian.GaussianOutput \
    --use_feat_static_cat True \
    --cardinality '[5]' \
    --prediction_length 5 \
    --freq M \
    2>&1 | egrep --color=always -i 'prediction_length|freq|epochs|\.[a-zA-Z]+Estimator|$'

echo -e '\nMinus --freq ...'
python $SRC/train.py --stop_before train \
    --s3_dataset $INPUT \
    --distr_output.__class__ gluonts.distribution.gaussian.GaussianOutput \
    --use_feat_static_cat True \
    --cardinality '[5]' \
    --prediction_length 5 \
    2>&1 | egrep --color=always -i 'prediction_length|freq|epochs|\.[a-zA-Z]+Estimator|$'

echo -e '\nMinus --prediction_length ...'
python $SRC/train.py --stop_before train \
    --s3_dataset $INPUT \
    --distr_output.__class__ gluonts.distribution.gaussian.GaussianOutput \
    --use_feat_static_cat True \
    --cardinality '[5]' \
    2>&1 | egrep --color=always -i 'prediction_length|freq|epochs|\.[a-zA-Z]+Estimator|$'
