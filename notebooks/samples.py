from gluonts.dataset.common import load_datasets

# Format of metadata: gluonts/dataset/common.py
tds = load_datasets("metadata", "train", "test")
# train = FileDataset("train", freq="M")
# test = FileDataset("test", freq="M")
train = list(tds.train)
test = list(tds.test)

print(train)
print(test)
