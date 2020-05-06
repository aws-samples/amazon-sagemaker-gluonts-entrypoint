import re

regex = [
    r"Epoch\[\d+\] Evaluation metric 'epoch_loss'=(\S+)",
    r"Epoch\[\d+\] Evaluation metric 'validation_epoch_loss'=(\S+)",
    r"Epoch\[\d+\] Learning rate is (\S+)",
    r"gluonts\[metric-abs_error\]: (\S+)",
    r"gluonts\[metric-RMSE\]: (\S+)",
    r"gluonts\[metric-wMAPE\]: (\S+)",
    r"asdf",
]


text = """[2020-04-23 09:26:06] [INFO] root Epoch[1] Evaluation metric 'epoch_loss'=6.790306
[2020-05-06 14:52:09] [INFO] root Epoch[2] Evaluation metric 'validation_epoch_loss'=0.874746
[2020-04-23 09:26:06] [INFO] root Epoch[2] Learning rate is 0.001
[2020-04-23 09:26:10] [INFO] __main__ gluonts[metric-RMSE]: 47143.54635620117
[2020-04-23 09:26:10] [INFO] __main__ gluonts[metric-abs_error]: 685.4429550170898
[2020-04-23 09:26:10] [INFO] __main__ gluonts[metric-wMAPE]: 0.130905881524086
"""

for r in regex:
    match = re.search(r, text)
    print("\n", r, sep="")
    if match:
        print(match.group(0))
        print(match.group(1))
    else:
        print("No match.")
