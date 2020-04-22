python entrypoint.py --stop_before train --s3-dataset s3_dir --algo gluonts.model.deep_factor.DeepFactorEstimator --prediction_length 5  --freq M \
    2>&1 | egrep --color=always -i 'prediction_length|freq|epochs|\.[a-zA-Z]+Estimator|$'

echo -e '\nMinus --freq ...'
python entrypoint.py --stop_before train --s3-dataset s3_dir --algo gluonts.model.deep_factor.DeepFactorEstimator --prediction_length 5 \
    2>&1 | egrep --color=always -i 'prediction_length|freq|epochs|\.[a-zA-Z]+Estimator|$'

echo -e '\nMinus --prediction_length ...'
python entrypoint.py --stop_before train --s3-dataset s3_dir --algo gluonts.model.deep_factor.DeepFactorEstimator \
    2>&1 | egrep --color=always -i 'prediction_length|freq|epochs|\.[a-zA-Z]+Estimator|$'
