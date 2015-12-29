require 'torch'
require 'nn'
require 'optim'

-- create training data
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()

   cmd:text()
   cmd:text('Options:')

   cmd:option('-optimization', 'SGD', 'optimization method: SGD | CG ')
   cmd:text()
   opt = cmd:parse(arg or {})
end

-- training the network
-- configure the optimizer

if opt.optimization == 'CG' then
   optimState = {
      maxIter = 2
   }
   -- optimMethod refers to a method
   optimMethod = optim.cg

elseif opt.optimization == 'SGD' then
   optimState = {
     	learningRate = 1e-2, 
		learningRateDecay = 1e-3, 
		weightDecay = 0, 
		momentum = 0
   }
   optimMethod = optim.sgd

end


inputs = torch.Tensor{
	{1.1, -1.1, 2.0},
	{1.5, 2.0, 1.0},
	{1.2, 3.0, -1.6}, 
	{-0.3, -0.5, 0.9}	
}
targets = torch.Tensor{
	{0.51, 1.1, 3.0, -1.0},
	{3.1, -1.2, 0.8, 0.1}, 
	{-2.2, 1.7, -1.8, -1.0}, 
	{1.4, -0.4, -0.4, 0.6}
}

-- create model 
model = nn.Sequential()
ninputs = 3
noutputs = 4
model:add(nn.Linear(ninputs, noutputs))

-- create criterion
criterion = nn.MSECriterion()

-- training the model 
params, gradParams  = model:getParameters()

-- define the closure to get f(x) and df/dx
feval = function(x)
	if x ~= params then
		params:copy(x)
	end

	-- select a new training example
	nidx = (nidx or 0) + 1
	if nidx > inputs:size(1) then nidx = 1 end

	local input = inputs[nidx]
	local target = targets[nidx]

	-- reset the gradients
	gradParams:zero()

	-- evaluate the loss function and derivatives w.r.t parameters
	local output = model:forward(input)
	local f =  criterion:forward(output, target)
	model:backward(input, criterion:backward(output, target))

	return f, gradParams
end


aver_loss = math.huge
iterations = 0
while aver_loss > 1e-3 do 
	sum_loss = 0
	for i = 1, inputs:size(1) do 
        _, loss_l = optimMethod(feval, params, optimState)
		sum_loss = sum_loss + loss_l[1]
	end
	aver_loss = sum_loss / inputs:size(1)
	iterations = iterations + 1
	print('iterations =  '..iterations..', average loss = ' .. aver_loss)
end