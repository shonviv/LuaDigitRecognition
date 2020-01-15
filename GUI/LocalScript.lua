local UserInputService = game:GetService("UserInputService")
local Held = false

local DrawFrame = script.Parent.DrawFrame
local Values = script.Parent.Values

local dims = {16, 16}
local targetDims = {8, 8}

local cnn = require(game:GetService("ReplicatedStorage").cnn)

game.Workspace.Value.Value = game.Workspace.Value.Value + 1.7

function cells(d0, d1)
	d0 = math.floor(d0)
	d1 = math.floor(d1)
	
	local narray = {}
	
	local index = 1
	for i = 1, d0 do
		local cd = {}
		for j = 1, d1 do
			table.insert(cd, index)
			
			index = index + 1
		end
		
		table.insert(narray, cd)
	end
	
	return narray
end

local cellDims = cells(dims[1], dims[2])

local function setGray(i, j)
	if i < 1 or i > dims[1] or j < 1 or j > dims[2] then return end
	local currentCell = DrawFrame["Cell" .. cellDims[i][j]]
	currentCell.BackgroundColor3 = Color3.new(currentCell.BackgroundColor3.r + 0.4, currentCell.BackgroundColor3.g + 0.4, currentCell.BackgroundColor3.b + 0.4)
end

index = 1
for i = 1, dims[1] do
	for j = 1, dims[2] do
		local pixel = Instance.new("TextLabel")
		pixel.Text = ""
		pixel.Size = UDim2.new(1 / dims[1], 0, 1 / dims[2], 0)
		pixel.Position = UDim2.new((j - 1) / dims[1], 0, (i - 1) / dims[2], 0)
		pixel.Name = "Cell" .. index
		pixel.BorderSizePixel = 0
		pixel.BackgroundColor3 = Color3.new(0, 0, 0)
		pixel.Parent = DrawFrame
		
		index = index + 1
		
		pixel.MouseMoved:connect(function()
		    if not Held or pixel.BackgroundColor3 == Color3.new(1, 1, 1) then return end
			
			pixel.BackgroundColor3 = Color3.new(1, 1, 1)
			
			setGray(i - 1, j)
			setGray(i, j - 1)
			setGray(i - 1, j - 1)
			setGray(i + 1, j)
			setGray(i, j + 1)
			setGray(i + 1, j + 1)
			setGray(i + 1, j - 1)
			setGray(i - 1, j - 1)
		end)
	end
end


UserInputService.InputBegan:Connect(function(inputObject)
	if inputObject.UserInputType == Enum.UserInputType.MouseButton1 then
		Held = true
	end
	
	if inputObject.KeyCode == Enum.KeyCode.C then
		for i, v in pairs(DrawFrame:GetChildren()) do
			v.BackgroundColor3 = Color3.new(0, 0, 0)
		end
	end
end)

UserInputService.InputEnded:Connect(function(inputObject)
	if inputObject.UserInputType == Enum.UserInputType.MouseButton1 then
		Held = false
	end
end)

spawn(function()
	while wait(.1) do
		print(tostring(Held))
	end
end)

function round(num, numDecimalPlaces)
	local mult = 10^(numDecimalPlaces or 0)
	return math.floor(num * mult + 0.5) / mult
end

spawn(function()
	while wait(0.1) do
		local mat = {}
		local lmat = {}
		
		for i, v in pairs(DrawFrame:GetChildren()) do
			if v.BackgroundColor3 ~= Color3.new(0, 0, 0) then
				table.insert(lmat, 16)
			else
				table.insert(lmat, 0)
			end
		end
		
		local i = 1
		
		while i < 240 do
			if (i - 1) % 16 == 0 and i - 1 ~= 0 then
				i = i + 16	
			end
			
			print(i)
			table.insert(mat, (lmat[i] + lmat[i + 1] + lmat[i + 16] + lmat[i + 16 + 1])/4  )
			
			i = i + 2
		end
		
		--[[for i, v in pairs(DrawFrame:GetChildren()) do
			if v.BackgroundColor3 ~= Color3.new(0, 0, 0) then
				table.insert(mat, 16)
			else
				table.insert(mat, 0)
			end
		end]]
		
		local image = cnn.predict(mat)
		local text = ""
		
		for i, v in pairs(image) do
			text = text .. (i - 1) .. ": " .. string.format("%.2f", (v[1] * 100)) .. "%\n"
		end
		
		Values.Numbers.Text = text
	end
end)