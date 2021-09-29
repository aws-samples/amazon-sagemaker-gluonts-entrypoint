#!/bin/bash

SRC=src/entrypoint
INPUT=refdata

echo -e '\nDeepAR...'
python $SRC/train.py --stop_before eval \
    --s3_dataset $INPUT \
    --algo gluonts.model.deepar.DeepAREstimator \
    --trainer.__class__ gluonts.mx.trainer.Trainer \
    --trainer.epochs 3 \
    --distr_output.__class__ gluonts.mx.distribution.GaussianOutput \
    --use_feat_static_cat True \
    --cardinality '[5]' \
    --prediction_length 2 \
    | egrep --color=always -i 'prediction_length|freq|epochs|\.[a-zA-Z]+Estimator|$'

echo -e '\nDeepState...'
python $SRC/train.py --stop_before eval \
    --s3_dataset $INPUT \
    --algo gluonts.model.deepstate.DeepStateEstimator \
    --trainer.__class__ gluonts.mx.trainer.Trainer \
    --trainer.epochs 3 \
    --use_feat_static_cat True \
    --cardinality '[5]' \
    --noise_std_bounds.__class__ gluonts.mx.distribution.lds.ParameterBounds \
    --noise_std_bounds.lower 1e-5 \
    --noise_std_bounds.upper 1e-1 \
    | egrep --color=always -i 'prediction_length|freq|epochs|\.[a-zA-Z]+Estimator|$'

echo -e '\nDeepFactor...'
python $SRC/train.py --stop_before eval \
    --s3_dataset $INPUT \
    --algo gluonts.model.deep_factor.DeepFactorEstimator \
    --trainer.__class__ gluonts.mx.trainer.Trainer \
    --trainer.epochs 3 \
    | egrep --color=always -i 'prediction_length|freq|epochs|\.[a-zA-Z]+Estimator|$'

echo -e '\nTransformer...'
python $SRC/train.py --stop_before eval \
    --s3_dataset $INPUT \
    --algo gluonts.model.transformer.TransformerEstimator \
    --trainer.__class__ gluonts.mx.trainer.Trainer \
    --trainer.epochs 3 \
    --distr_output.__class__ gluonts.mx.distribution.GaussianOutput \
    | egrep --color=always -i 'prediction_length|freq|epochs|\.[a-zA-Z]+Estimator|$'

echo -e '\nGaussian process with RBF kernel...'
python $SRC/train.py --stop_before eval \
    --s3_dataset $INPUT \
    --algo gluonts.model.gp_forecaster.GaussianProcessEstimator \
    --trainer.__class__ gluonts.mx.trainer.Trainer \
    --trainer.epochs 3 \
    --cardinality 2 \
    --prediction_length 5 \
    | egrep --color=always -i 'prediction_length|freq|\.[a-zA-Z]+kerneloutput|epochs|\.[a-zA-Z]+Estimator|$'

echo -e '\nGaussian process with periodic kernel -- expect this to fail...'
python $SRC/train.py --stop_before eval \
    --s3_dataset $INPUT \
    --algo gluonts.model.gp_forecaster.GaussianProcessEstimator \
    --trainer.__class__ gluonts.mx.trainer.Trainer \
    --trainer.epochs 3 \
    --cardinality 2 \
    --kernel_output.__class__ gluonts.mx.kernels.PeriodicKernelOutput \
    --prediction_length 5 \
    | egrep --color=always -i 'prediction_length|freq|\.[a-zA-Z]+kerneloutput|epochs|\.[a-zA-Z]+Estimator|$'

echo -e '\nNPTS...'
python $SRC/train.py --stop_before eval \
    --s3_dataset $INPUT \
    --algo gluonts.model.npts.NPTSEstimator \
    | egrep --color=always -i 'prediction_length|freq|\.[a-zA-Z]+kerneloutput|epochs|\.[a-zA-Z]+Estimator|$'
