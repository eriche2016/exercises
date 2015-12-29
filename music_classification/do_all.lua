require 'torch'

torch.manualSeed(1234)

----------------------------------------------------------------------
print '==> executing all'

dofile 'data.lua'
dofile 'model.lua'
dofile 'loss.lua'
dofile 'train.lua'
dofile 'test.lua'

----------------------------------------------------------------------
print '==> training!'

while true do
   train()
   test()
end
