import os

n = os.listdir("data/landcover/splitted/images")

print(len(n))

train = os.listdir("data/landcover/train/images")
val = os.listdir("data/landcover/val/images")
test = os.listdir("data/landcover/test/images")

print(len(train))
print(len(val))
print(len(test))
