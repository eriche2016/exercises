require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

----------------------------------------------------------------------
----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Options:')
   cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
   cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS')
   cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
   cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic)')
   cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
   cmd:option('-momentum', 0, 'momentum (SGD only)')
   cmd:option('-t0', 1, 'start averaging at t0 (ASGD only), in nb of epochs')
   cmd:option('-maxIter', 2, 'maximum nb of iterations for CG and LBFGS')
   cmd:text()
   opt = cmd:parse(arg or {})
end
opt.save = './model/'

print '==> defining some tools'

-- classes
classes = {'1','2','3','4'}

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- Log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the model
-- into a 1-dim vector
parameters,gradParameters = model:getParameters()


----------------------------------------------------------------------
print '==> configuring optimizer'
-- training network with different optimization method 
if opt.optimization == 'CG' then
   optimState = {
      maxIter = opt.maxIter
   }
   -- optimMethod refers to a method
   optimMethod = optim.cg

elseif opt.optimization == 'LBFGS' then
   optimState = {
      learningRate = opt.learningRate,
      maxIter = opt.maxIter,
      nCorrection = 10
   }
   optimMethod = optim.lbfgs

elseif opt.optimization == 'SGD' then
   optimState = {
      learningRate = opt.learningRate,
      weightDecay = opt.weightDecay,
      momentum = opt.momentum,
      learningRateDecay = 1e-7
   }
   optimMethod = optim.sgd

elseif opt.optimization == 'ASGD' then
   optimState = {
      eta0 = opt.learningRate,
      t0 = trsize * opt.t0
   }
   optimMethod = optim.asgd

else
   error('unknown optimization method')
end

----------------------------------------------------------------------
print '==> defining training procedure'

test_targets = {}
function train()

   -- epoch tracker
   epoch = epoch or 1  -- epoch is 1 by default

   -- local vars
   local time = sys.clock()

   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   model:training()

   -- shuffle at each epoch
   shuffle = torch.randperm(trsize)

   -- do one epoch
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   
   for t = 1,trsize, opt.batchSize do
      -- disp progress
      xlua.progress(t, trsize)

      -- create mini batch
      local inputs = {}
      local targets = {}
      for i = t,math.min(t+opt.batchSize-1,trsize) do
         -- load new sample
         local input = tr_data[shuffle[i]]
         local target = tr_label[shuffle[i]]

         table.insert(inputs, input)
         table.insert(targets, target)
      end
      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
                       -- get new parameters
                       if x ~= parameters then
                          parameters:copy(x)
                       end

                       -- reset gradients
                       gradParameters:zero()

                       -- f is the average of all criterions
                       local f = 0

                       -- evaluate function for complete mini batch
                       for i = 1,#inputs do
                          -- estimate f
                          local output = model:forward(inputs[i])
                          local err = criterion:forward(output, targets[i][1])
                          f = f + err

                          -- estimate df/dW
                          local df_do = criterion:backward(output, targets[i][1])
                          model:backward(inputs[i], df_do)

                          -- update confusion
                          table.insert(test_targets, targets[i][1]) -- targets[i] is userdata, if use targets[i] directly, 
                                                                    -- type(targets)== 'userdata', confusion table will always use index
                                                                    -- ie 1, so class for 2, 3, 4 will ways be NaN%
                          confusion:add(output, targets[i][1])
                       end

                       -- normalize gradients and f(X)
                       gradParameters:div(#inputs)
                       -- f is the loss 
                       f = f/#inputs

                       -- return f(loss)and df/dX(gradient w.r.t parameters)
                       return f,gradParameters
                    end

      -- optimize on current mini-batch
      if optimMethod == optim.asgd then
         _,_,average = optimMethod(feval, parameters, optimState)
      else
         optimMethod(feval, parameters, optimState)
      end
   end

   -- time taken
   time = sys.clock() - time
   time = time / trsize
   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- update logger/plot
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   
   -- save/log current net
   local filename = paths.concat(opt.save, 'model.net')
   -- or we can use paths.mkdir()
   -- mkdir -p: no error if the current dir already existed
   --  sys.dirname('dir/model.net') will return 'dir' 
   -- make directory 
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('==> saving model to '..filename)
   torch.save(filename, model)

   -- next epoch
   confusion:zero()
   epoch = epoch + 1
end
