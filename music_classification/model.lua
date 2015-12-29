require 'nn'
require 'optim'
require 'xlua'
require 'data.lua'


print "==> making an mlp"

-- 4 class problem
noutputs = 4

-- input dimensions
nfeats = 24

-- number of hidden units
nhiddens =  nfeats * 5

-- construct models
-- 2 layer
print "==> constructing mlp model for music classification"

--[[-- model 1
model = nn.Sequential()
model:add(nn.Linear(nfeats, nhiddens))

model:add(nn.Tanh())
model:add(nn.Linear(nhiddens, nhiddens))
model:add(nn.Tanh())
model:add(nn.Linear(nhiddens, noutputs))
--]]


model = nn.Sequential()
model:add(nn.Linear(nfeats, nhiddens))

model:add(nn.ReLU())
model:add(nn.Linear(nhiddens, nhiddens))
model:add(nn.ReLU())
model:add(nn.Linear(nhiddens, noutputs))

print "==> model configuration done"
print '==> here is the model:'
print(model)
