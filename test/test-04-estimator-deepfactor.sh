#!/bin/bash

SRC=src
INPUT=refdata

python $SRC/train.py --stop_before train \
    --s3_dataset refdata \
    --algo gluonts.model.deep_factor.DeepFactorEstimator \
    --prediction_length 5  \
    --freq M \
    2>&1 | egrep --color=always -i 'prediction_length|freq|epochs|\.[a-zA-Z]+Estimator|$'

echo -e '\nMinus --freq ...'
python $SRC/train.py --stop_before train \
    --s3_dataset refdata \
    --algo gluonts.model.deep_factor.DeepFactorEstimator \
    --prediction_length 5 \
    2>&1 | egrep --color=always -i 'prediction_length|freq|epochs|\.[a-zA-Z]+Estimator|$'

echo -e '\nMinus --prediction_length ...'
python $SRC/train.py --stop_before train \
    --s3_dataset refdata \
    --algo gluonts.model.deep_factor.DeepFactorEstimator \
    2>&1 | egrep --color=always -i 'prediction_length|freq|epochs|\.[a-zA-Z]+Estimator|$'
