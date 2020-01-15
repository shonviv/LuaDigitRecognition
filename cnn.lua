local cnn = {}

local digits = require(game:GetService("ReplicatedStorage").digits)
local sc = require(game:GetService("ReplicatedStorage").sctransform)

-- math.randomseed(os.time())

function zeros(d0, d1)
	d0 = math.floor(d0)
	d1 = math.floor(d1)
	
	local narray = {}
	
	for i = 1, d0 do
		local cd = {}
		for j = 1, d1 do
			table.insert(cd, 0)
		end
		
		table.insert(narray, cd)
	end
	
	return narray
end

local Y_train_ = zeros(10, 1437)
for i = 1, 1437 do
	Y_train_[digits.y_train[i] + 1][i] = 1
end

-- Box-Muller
function randn_bm()
	local u = 0
	local v = 0
	
    while u == 0 do 
		u = math.random()
	end

    while v == 0 do
		v = math.random()
	end

    local num = math.sqrt(-2.0 * math.log( u )) * math.cos(2.0 * math.pi * v)
    num = num / 10.0 + 0.5

    if num > 1 or num < 0 then
		return randn_bm()
	end 

    if num > 0.5 then
		num = num * -1
	end

	num = num * 3
	
    return num 
end

function randn(d0, d1, mult)
	d0 = math.floor(d0)
	d1 = math.floor(d1)
	
	local narray = {}
	
	for i = 1, d0 do
		local cd = {}
		for j = 1, d1 do
			table.insert(cd, randn_bm() * mult)
		end
		
		table.insert(narray, cd)
	end
	
	return narray
end

--[[
	function dot(a, b)
		local ret = 0
		
		for i = 1, #a do
			ret = ret + a[i] * b[i]
		end
		
		return ret
	end
--]]


--[[
	function MatMul( m1, m2 )
	-- multiply rows with columns
	local mtx = {}
	for i = 1,#m1 do
		mtx[i] = {}
		for j = 1,#m2[1] do
			local num = m1[i][1] * m2[1][j]
			for n = 2,#m1[1] do
				num = num + m1[i][n] * m2[n][j]
			end
			mtx[i][j] = num
		end
	end
	return mtx
end
--]]

function MatMul( m1, m2 )
    if #m1[1] ~= #m2 then       -- inner matrix-dimensions must agree
        return nil      
    end 
 
    local res = {}
 
    for i = 1, #m1 do
        res[i] = {}
        for j = 1, #m2[1] do
            res[i][j] = 0
            for k = 1, #m2 do
                res[i][j] = res[i][j] + m1[i][k] * m2[k][j]
            end
        end
    end
 	
    return res
end





function initialize_parameters_deep(layer_dims)
	math.randomseed(os.time())
	local parameters = {}
	local L = #layer_dims
	
	for l = 2, L do
		parameters["W" .. (l - 1)] = randn(layer_dims[l], layer_dims[l - 1], 0.01)
		parameters["b" .. (l - 1)] = zeros(layer_dims[l], 1)
	end
	
	return parameters
end

function addValueToArray(array, value)
	local new = {}
	
	for i, v in pairs(array) do
		table.insert(new, {})
		
		for j, d in pairs(v) do
			table.insert(new[i], d + value)
		end
	end
	
	return new
end

function addArrays(array1, array2)
	local new = {}
	
	for i, v in pairs(array1) do
		table.insert(new, {})
		
		for j, d in pairs(v) do
			table.insert(new[i], d + array2[i][j])
		end
		
	end
	
	return new
end

function addB(a, b)
	local new = {}
	
	if (a == nil) then
		print("X")
	end
	
	for i, v in pairs(a) do
		table.insert(new, {})
		
		for j, d in pairs(v) do
			table.insert(new[i], d + b[i][1])
		end
	end
	
	return new
end

function linear_forward(A, W, b)
	local Z = addB(MatMul(W, A), b)
	
	local cache = {}
	table.insert(cache, A)
	table.insert(cache, W)
	table.insert(cache, b)
	
	return {Z, cache}
end

function sigmoid_(Z)
	local new = {}
	
	if type(Z[1]) == "number" then
		for j, d in pairs(Z) do
			table.insert(new, 1 / (1+math.exp(-d)))
		end
	
	else
		for i, v in pairs(Z) do
			table.insert(new, {})
			
			if (type(v) == "number") then
				print("STOP")
			end
			for j, d in pairs(v) do
				table.insert(new[i], 1 / (1+math.exp(-d)))
			end
		end
		
	end
	
	return new
end

function relu_(Z)
	local new = {}
	
	for i, v in pairs(Z) do
		table.insert(new, {})
		
		for j, d in pairs(v) do
			if tonumber(d) > 0 then
				table.insert(new[i], d)
			else
				table.insert(new[i], 0)
			end
		end
	end
	
	return new
end

function drelu_(Z)
	local new = {}
	
	for i, v in pairs(Z) do
		table.insert(new, {})
		
		for j, d in pairs(v) do
			if d > 0 then
				table.insert(new[i], 1)
			else
				table.insert(new[i], 0)
			end
		end
	end
	
	return new
end

function dsigmoid_(Z)
	return multiplyArrays(sigmoid_(Z), baseValueSubtractionArray(sigmoid_(Z), 1))
end

function sigmoid(Z)
	
	return {sigmoid_(Z), Z}
end

function relu(Z)
	
	return {relu_(Z), Z}
end

function linear_activation_forward(A_prev,W,b,activation)
	local l_args = linear_forward(A_prev, W, b)
	
	local Z = l_args[1]
	local linear_cache = l_args[2]
	
	local A
	local activation_cache
	
	if activation == "sigmoid" then
		local s_args = sigmoid(Z)
		A = s_args[1]
		activation_cache = s_args[2]
		
	elseif activation == "relu" then
		local s_args = relu(Z)
		A = s_args[1]
		activation_cache = s_args[2]
	end
	
	local cache = {}
	table.insert(cache, linear_cache)
	table.insert(cache, activation_cache)
	
	return {A, cache}
end

function L_model_forward(X, parameters)
	local caches = {}
	local A = X
	local L = 3 --math.floor(#parameters / 2)
	
	for l = 1, L - 1 do
		local A_prev = A
		local lap_args = linear_activation_forward(A_prev, parameters["W" .. l], parameters["b" .. l], "relu")
		A = lap_args[1]
		local cache = lap_args[2]
		
		table.insert(caches, cache)
	end
	
	local la_args = linear_activation_forward(A, parameters["W" .. L], parameters["b" .. L], "sigmoid")
	table.insert(caches, la_args[2])
	
	local AL = la_args[1]
	return {AL, caches}
end

-- 1D arrays of ints
function sumArray(array)
	local sum = 0
	
	for i, v in pairs(array) do
		for j, d in pairs(v) do
			sum = sum + d
		end
	end
	
	return sum
end

function baseValueSubtractionArray(array, value)
	local new = {}
	
	for i, v in pairs(array) do
		table.insert(new, {})
		
		for j, d in pairs(v) do
			table.insert(new[i], value - d)
		end
	end
	
	return new
end

function multiplyArrays(array1, array2)
	local new = {}
	
	for i, v in pairs(array1) do
		table.insert(new, {})
		
		for j, d in pairs(v) do
			table.insert(new[i], d * array2[i][j])
		end
	end
	
	return new
end

function logArray(array)
	local new = {}
	
	for i, v in pairs(array) do
		table.insert(new, {})
		
		for j, d in pairs(v) do
			table.insert(new[i], math.log(d))
		end
	end
	
	return new
end

function compute_cost(AL, Y)
	local m = 1437 -- #Y[2]
	
	
	local left = multiplyArrays(Y, logArray(AL))
	local right = multiplyArrays(baseValueSubtractionArray(Y, 1), logArray(baseValueSubtractionArray(AL, 1)))
	
	local add = addArrays(left, right)
	local final = sumArray(add)
	
	local cost = (-(1/m)) * final
	
	return cost
end

function Transpose( m )
    local res = {}
 
    for i = 1, #m[1] do
        res[i] = {}
        for j = 1, #m do
            res[i][j] = m[j][i]
        end
    end
 
    return res
end

function sumArray1D(array)
	local sum = 0
	
	for i, v in pairs(array) do
		sum = sum + v
	end
	
	return sum
end

function sumDims(array)
	local new = {}
	
	for i, v in pairs(array) do
		table.insert(new, {})
		table.insert(new[i], sumArray1D(v))
	end
	
	return new
end

function linear_backward(dZ, cache)
	local A_prev = cache[1]
	local W = cache[2]
	local b = cache[3]
	local m = 1437 -- #A_prev[2] NOT TESTED
	
	local dW = multiplyArray(MatMul(dZ, Transpose(A_prev)), (1/m))
	local db = multiplyArray(sumDims(dZ), (1/m))
	local dA_prev = MatMul(Transpose(W), dZ)
	
	return {dA_prev, dW, db}
end

function relu_backward(dA, activation_cache)
	return multiplyArrays(dA, drelu_(activation_cache))
end

function sigmoid_backward(dA, activation_cache)
	return multiplyArrays(dA, dsigmoid_(activation_cache))
end

function linear_activation_backward(dA, cache, activation)
	local linear_cache = cache[1]
	local activation_cache = cache[2]
	
	local dA_prev
	local dW
	local db
	
	if activation == "relu" then
		local dZ = relu_backward(dA, activation_cache)
		local l_args = linear_backward(dZ, linear_cache)
		
		dA_prev = l_args[1]
		dW = l_args[2]
		db = l_args[3]
	elseif activation == "sigmoid" then
		local dZ = sigmoid_backward(dA, activation_cache)
		local l_args = linear_backward(dZ, linear_cache)
		
		dA_prev = l_args[1]
		dW = l_args[2]
		db = l_args[3]
	end
	
	return {dA_prev, dW, db}
end

function divideArrays(array1, array2)
	local new = {}
	
	for i, v in pairs(array1) do
		table.insert(new, {})
		
		for j, d in pairs(v) do
			table.insert(new[i], d / array2[i][j])
		end
	end
	
	return new
end

function subtractArrays(array1, array2)
	local new = {}
	
	for i, v in pairs(array1) do
		table.insert(new, {})
		
		for j, d in pairs(v) do
			table.insert(new[i], d - array2[i][j])
		end
	end
	
	return new
end

function multiplyArray(array, value)
	local new = {}
	
	for i, v in pairs(array) do
		table.insert(new, {})
		
		for j, d in pairs(v) do
			table.insert(new[i], d * value)
		end
	end
	
	return new
end

function L_model_backward(AL, Y, caches)
	local grads = {}
	local L = #caches
	local m = 1437 -- #AL[2]
	
	local dAL = multiplyArray(subtractArrays((divideArrays(Y, AL)), divideArrays(baseValueSubtractionArray(Y, 1), baseValueSubtractionArray(AL, 1)) ),  -1)
	local current_cache = caches[L]
	
	local l_args = linear_activation_backward(dAL,current_cache,"sigmoid")
	grads["dA" .. L - 1] = l_args[1]
	grads["dW" .. L] = l_args[2]
	grads["db" .. L] = l_args[3]
	
	for l = L - 1, 1, -1 do
		current_cache = caches[l]
		local lb_args = linear_activation_backward(grads["dA" .. (l)], current_cache, "relu")
		local dA_prev_temp = lb_args[1]
		local dW_temp = lb_args[2]
		local db_temp = lb_args[3]
		
		grads["dA" .. (l - 1)] = dA_prev_temp
        grads["dW" .. (l)] = dW_temp
        grads["db" .. (l)] = db_temp
	end
	
	return grads
end

function update_parameters(parameters, grads, learning_rate)
	local L = 3 --math.floor(#parameters / 2)
	
	for l = 1, L do
		parameters["W" .. (l)] =  multiplyArrays(addValueToArray(parameters["W" .. (l)], -(learning_rate)), grads["dW" .. (l)])
		parameters["b" .. (l)] =  multiplyArrays(addValueToArray(parameters["b" .. (l)], -(learning_rate)), grads["db" .. (l)])
	end
	
	return parameters
end

local layers_dims = {digits.n_x, 60, 10, 10}

function L_layer_model(X, Y, layers_dims, learning_rate, num_iterations, print_cost)
	math.randomseed(os.time())
	local costs = {}
	
	local parameters = initialize_parameters_deep(layers_dims)
	
	
	for i = 1, num_iterations do
		wait()
		local lm_args = L_model_forward(X, parameters)
		local AL = lm_args[1]
		local caches = lm_args[2]
		
		local cost = compute_cost(AL, Y)
		local grads = L_model_backward(AL, Y, caches)
		
		parameters = update_parameters(parameters, grads, learning_rate)
		
		if print_cost and i % 1 == 0 then
			print("Cost after iteration " .. i .. ": " .. cost)
			table.insert(costs, cost)
		end
	end
	
	return parameters
end

--parameters = L_layer_model(digits.X_train, Y_train_, layers_dims, 0.005, 50000, true)
parameters = require(game.ReplicatedStorage.parameters).parameters

function argMax(mat)
	local new = {}
	
	for i = 1, #mat do
		local axis = {}
		
		for j = 1, #mat[i] do
			table.insert(axis, mat[i][j])
		end
		
		table.insert(new, {math.max(unpack(axis))})
	end
	
	return new
end


function predict_L_layer(X, parameters)
	local l_args = L_model_forward(X,parameters)
	local AL = l_args[1]
	local caches = l_args[2]
	
	return argMax(AL)
end

function cnn.predict(mat)
	local img = sc.transform(mat)
	local predicted_digit = predict_L_layer(img, parameters)
	return predicted_digit
end

return cnn