require 'torch'
require 'paths'

local matio = require 'matio'

dataset = {"data1.mat", "data2.mat", "data3.mat", "data4.mat"}

data_paths = "./data/"

function load_dataset(data_name)
	local path2data = data_paths .. data_name
	local label_data = matio.load(path2data)
	return label_data
end

-- music_dataset[i] for class i(i = 1, 2, 3, 4).
music_dataset = {} 

-- load c1
music_dataset[1] = {}
temp = load_dataset(dataset[1])
feature_size = temp.c1:size(2) - 1 -- 1 for label column
music_dataset[1].label  = temp.c1: narrow(2, 1, 1):clone()
music_dataset[1].data = temp.c1:narrow(2, 2, feature_size):clone()

-- load c2
music_dataset[2] = {}
temp = load_dataset(dataset[2])
music_dataset[2].label  = temp.c2: narrow(2, 1, 1):clone()
music_dataset[2].data = temp.c2:narrow(2, 2, feature_size):clone()

-- load c3
music_dataset[3] = {}
temp = load_dataset(dataset[3])
music_dataset[3].label  = temp.c3: narrow(2, 1, 1):clone()
music_dataset[3].data = temp.c3:narrow(2, 2, feature_size):clone()

-- load c4
music_dataset[4] = {}
temp = load_dataset(dataset[4])
music_dataset[4].label  = temp.c4: narrow(2, 1, 1):clone()
music_dataset[4].data = temp.c4:narrow(2, 2, feature_size):clone()

-- train dataset: random choose 375 data points respectively for each class (class balance)
-- test dataset: the remaining data are test data
train_data = {}  
test_data = {}
for i = 1, #music_dataset do 
	train_data[i] = {}
	test_data[i] = {}
	permute_idx = torch.randperm(500) -- 500 for each class
	train_idx = permute_idx:sub(1, 375):long()
	test_idx = permute_idx:sub(376, 500):long()

	train_data[i].data = music_dataset[i].data:index(1, train_idx)
	train_data[i].label = music_dataset[i].label:index(1, train_idx)

	test_data[i].data = music_dataset[i].data:index(1, test_idx)
	test_data[i].label = music_dataset[i].label:index(1, test_idx)
end

-- merge the 4 classes dataset to train_data and test_data
-- 1500 training examples and 500 test examples
trsize = 1500
tesize = 500
tr_data = torch.Tensor(1500, 24)
tr_label = torch.Tensor(1500, 1)

tst_data = torch.Tensor(500, 24)
tst_label = torch.Tensor(500, 1)

-- making training data
start = 1 
for i = 1, #train_data do 
	tr_data[{{start, start - 1 + train_data[i].data:size(1)}}] = train_data[i].data
	tr_label[{{start, start - 1 + train_data[i].label:size(1)}}] = train_data[i].label
	start = start + train_data[i].data:size(1)
end

-- making test data
start = 1
for i = 1, #test_data do 
	tst_data[{{start, start - 1 + test_data[i].data:size(1)}}] = test_data[i].data
	tst_label[{{start, start - 1 + test_data[i].label:size(1)}}] = test_data[i].label
	start = start + test_data[i].data:size(1)
end



