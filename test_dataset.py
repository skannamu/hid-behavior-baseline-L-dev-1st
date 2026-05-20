from src.dataset import HIDDataset


dataset = HIDDataset(
    "data/processed/window.csv"
)

sample = dataset[0]

print(sample.shape)
print(sample)