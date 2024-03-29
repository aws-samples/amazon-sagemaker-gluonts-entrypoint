#!/bin/bash

SRC=src/entrypoint
INPUT=refdata

python $SRC/train.py --stop_before train \
    --s3_dataset $INPUT \
    --algo gluonts.model.gp_forecaster.GaussianProcessEstimator \
    --cardinality 2 \
    --prediction_length 5  \
    --freq M \
    2>&1 | egrep --color=always -i 'prediction_length|freq|\.[a-zA-Z]+kerneloutput|epochs|\.[a-zA-Z]+Estimator|$'

echo -e '\nMinus --freq ...'
python $SRC/train.py --stop_before train \
    --s3_dataset $INPUT \
    --algo gluonts.model.gp_forecaster.GaussianProcessEstimator \
    --cardinality 2 \
    --prediction_length 5 \
    2>&1 | egrep --color=always -i 'prediction_length|freq|\.[a-zA-Z]+kerneloutput|epochs|\.[a-zA-Z]+Estimator|$'

echo -e '\nMinus --prediction_length ...'
python $SRC/train.py --stop_before train \
    --s3_dataset $INPUT \
    --algo gluonts.model.gp_forecaster.GaussianProcessEstimator \
    --cardinality 2 \
    2>&1 | egrep --color=always -i 'prediction_length|freq|\.[a-zA-Z]+kerneloutput|epochs|\.[a-zA-Z]+Estimator|$'

echo -e '\nWith --kernel_output ...'
python $SRC/train.py --stop_before train \
    --s3_dataset $INPUT \
    --algo gluonts.model.gp_forecaster.GaussianProcessEstimator \
    --cardinality 2 \
    --kernel_output.__class__ gluonts.mx.kernels.PeriodicKernelOutput \
    --prediction_length 5 \
    2>&1 | egrep --color=always -i 'prediction_length|freq|\.[a-zA-Z]+kerneloutput|epochs|\.[a-zA-Z]+Estimator|$'
