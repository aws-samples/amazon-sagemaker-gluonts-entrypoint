echo -e '\nDeepAR...'
python entrypoint.py --stop_before eval --s3-dataset s3_dir --algo gluonts.model.deepar.DeepAREstimator --trainer gluonts.trainer.Trainer --trainer.epochs 3 --distr_output gluonts.distribution.gaussian.GaussianOutput --use_feat_static_cat True --cardinality '[5]' --prediction_length 2 \
    | egrep --color=always -i 'prediction_length|freq|epochs|\.[a-zA-Z]+Estimator|$'

echo -e '\nDeepState...'
python entrypoint.py --stop_before eval --s3-dataset s3_dir --algo gluonts.model.deepstate.DeepStateEstimator --trainer gluonts.trainer.Trainer --trainer.epochs 3 --use_feat_static_cat True --cardinality '[5]' --noise_std_bounds gluonts.distribution.lds.ParameterBounds --noise_std_bounds.lower 1e-5 --noise_std_bounds.upper 1e-1 \
    | egrep --color=always -i 'prediction_length|freq|epochs|\.[a-zA-Z]+Estimator|$'

echo -e '\nDeepFactor...'
python entrypoint.py --stop_before eval --s3-dataset s3_dir --algo gluonts.model.deep_factor.DeepFactorEstimator --trainer gluonts.trainer.Trainer --trainer.epochs 3 \
    | egrep --color=always -i 'prediction_length|freq|epochs|\.[a-zA-Z]+Estimator|$'

echo -e '\nTransformer...'
python entrypoint.py --stop_before eval --s3-dataset s3_dir --algo gluonts.model.transformer.TransformerEstimator --trainer gluonts.trainer.Trainer --trainer.epochs 3 --distr_output gluonts.distribution.gaussian.GaussianOutput \
    | egrep --color=always -i 'prediction_length|freq|epochs|\.[a-zA-Z]+Estimator|$'

echo -e '\nGaussian process with RBF kernel...'
python entrypoint.py --stop_before eval --s3-dataset s3_dir --algo gluonts.model.gp_forecaster.GaussianProcessEstimator --trainer gluonts.trainer.Trainer --trainer.epochs 3 --cardinality 2 --prediction_length 5 \
    | egrep --color=always -i 'prediction_length|freq|\.[a-zA-Z]+kerneloutput|epochs|\.[a-zA-Z]+Estimator|$'

echo -e '\nGaussian process with periodic kernel -- expect this to fail...'
python entrypoint.py --stop_before eval --s3-dataset s3_dir --algo gluonts.model.gp_forecaster.GaussianProcessEstimator --trainer gluonts.trainer.Trainer --trainer.epochs 3 --cardinality 2 --kernel_output gluonts.kernels.PeriodicKernelOutput --prediction_length 5 \
    | egrep --color=always -i 'prediction_length|freq|\.[a-zA-Z]+kerneloutput|epochs|\.[a-zA-Z]+Estimator|$'

echo -e '\nNPTS...'
python entrypoint.py --stop_before eval --s3-dataset s3_dir --algo gluonts.model.npts.NPTSEstimator \
    | egrep --color=always -i 'prediction_length|freq|\.[a-zA-Z]+kerneloutput|epochs|\.[a-zA-Z]+Estimator|$'
