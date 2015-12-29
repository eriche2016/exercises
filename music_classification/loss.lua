print "==> define loss"
model:add(nn.LogSoftMax()) 
criterion = nn.ClassNLLCriterion()