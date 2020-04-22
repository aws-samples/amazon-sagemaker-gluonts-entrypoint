python entrypoint.py --stop_before train --s3-dataset s3_dir --algo gluonts.model.npts.NPTSEstimator --prediction_length 5  --freq M \
    2>&1 | egrep --color=always -i 'prediction_length|freq|\.[a-zA-Z]+kerneloutput|epochs|\.[a-zA-Z]+Estimator|$'

echo -e '\nMinus --freq ...'
python entrypoint.py --stop_before train --s3-dataset s3_dir --algo gluonts.model.npts.NPTSEstimator --prediction_length 5 \
    2>&1 | egrep --color=always -i 'prediction_length|freq|\.[a-zA-Z]+kerneloutput|epochs|\.[a-zA-Z]+Estimator|$'

echo -e '\nMinus --prediction_length ...'
python entrypoint.py --stop_before train --s3-dataset s3_dir --algo gluonts.model.npts.NPTSEstimator \
    2>&1 | egrep --color=always -i 'prediction_length|freq|\.[a-zA-Z]+kerneloutput|epochs|\.[a-zA-Z]+Estimator|$'

echo -e '\nWith --kernel_output ...'
python entrypoint.py --stop_before train --s3-dataset s3_dir --algo gluonts.model.npts.NPTSEstimator --kernel_type uniform \
    2>&1 | egrep --color=always -i 'prediction_length|freq|\.[a-zA-Z]+kerneloutput|epochs|\.[a-zA-Z]+Estimator|$'
