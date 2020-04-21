python entrypoint.py --stop_before train --s3-dataset s3_dir --algo gluonts.model.transformer.TransformerEstimator --distr_output gluonts.distribution.gaussian.GaussianOutput --prediction_length 5  --freq M \
    2>&1 | egrep --color=always -i 'prediction_length|freq|epochs|\.[a-zA-Z]+Estimator'

echo -e '\nMinus --freq ...'
python entrypoint.py --stop_before train --s3-dataset s3_dir --algo gluonts.model.transformer.TransformerEstimator --distr_output gluonts.distribution.gaussian.GaussianOutput --prediction_length 5 \
    2>&1 | egrep --color=always -i 'prediction_length|freq|epochs|\.[a-zA-Z]+Estimator'

echo -e '\nMinus --prediction_length ...'
python entrypoint.py --stop_before train --s3-dataset s3_dir --algo gluonts.model.transformer.TransformerEstimator --distr_output gluonts.distribution.gaussian.GaussianOutput \
    2>&1 | egrep --color=always -i 'prediction_length|freq|epochs|\.[a-zA-Z]+Estimator'
