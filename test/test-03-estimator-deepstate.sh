#!/bin/bash

SRC=src
INPUT=refdata

python $SRC/train.py --stop_before train \
    --s3_dataset refdata \
    --algo gluonts.model.deepstate.DeepStateEstimator \
    --use_feat_static_cat True \
    --cardinality '[5]' \
    --noise_std_bounds.__class__ gluonts.distribution.lds.ParameterBounds \
    --noise_std_bounds.lower 1e-5 \
    --noise_std_bounds.upper 1e-1 \
    --prediction_length 5  \
    --freq M \
    2>&1 | egrep --color=always -i 'prediction_length|freq|epochs|\.[a-zA-Z]+Estimator|$'

echo -e '\nMinus --freq ...'
python $SRC/train.py --stop_before train \
    --s3_dataset refdata \
    --algo gluonts.model.deepstate.DeepStateEstimator \
    --use_feat_static_cat True \
    --cardinality '[5]' \
    --noise_std_bounds.__class__ gluonts.distribution.lds.ParameterBounds \
    --noise_std_bounds.lower 1e-5 \
    --noise_std_bounds.upper 1e-1 \
    --prediction_length 5 \
    2>&1 | egrep --color=always -i 'prediction_length|freq|epochs|\.[a-zA-Z]+Estimator|$'

echo -e '\nMinus --prediction_length ...'
python $SRC/train.py --stop_before train \
    --s3_dataset refdata \
    --algo gluonts.model.deepstate.DeepStateEstimator \
    --use_feat_static_cat True \
    --cardinality '[5]' \
    --noise_std_bounds.__class__ gluonts.distribution.lds.ParameterBounds \
    --noise_std_bounds.lower 1e-5 \
    --noise_std_bounds.upper 1e-1 \
    2>&1 | egrep --color=always -i 'prediction_length|freq|epochs|\.[a-zA-Z]+Estimator|$'
