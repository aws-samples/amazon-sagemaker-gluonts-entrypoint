python entrypoint.py --stop_before train --s3-dataset s3_dir --distr_output gluonts.distribution.gaussian.GaussianOutput --use_feat_static_cat True --cardinality '[5]' --prediction_length 5 --freq M \
    2>&1 | egrep --color=always -i 'prediction_length|freq|epochs|\.[a-zA-Z]+Estimator|$'

echo -e '\nMinus --freq ...'
python entrypoint.py --stop_before train --s3-dataset s3_dir --distr_output gluonts.distribution.gaussian.GaussianOutput --use_feat_static_cat True --cardinality '[5]' --prediction_length 5 \
    2>&1 | egrep --color=always -i 'prediction_length|freq|epochs|\.[a-zA-Z]+Estimator|$'

echo -e '\nMinus --prediction_length ...'
python entrypoint.py --stop_before train --s3-dataset s3_dir --distr_output gluonts.distribution.gaussian.GaussianOutput --use_feat_static_cat True --cardinality '[5]' \
    2>&1 | egrep --color=always -i 'prediction_length|freq|epochs|\.[a-zA-Z]+Estimator|$'
