# LuaDigitRecognition
Application of a deep neural network in handwritten digit recognition/

The neural net model consists of 3 hidden layers and an output layers to train the model. A Python program using the sci-kit learn library was written to export the initial transformed data for feature scaling. 

Over 1400 handwritten digits comprised by an 8x8 matrix of cells representing brightness were trained in the Lua program using both Sigmoid and ReLu activation functions.

The model was trained over 250,000 iterations and recieved a testing accuracy of 98.729%.

Below is an example of a few predictions by the neural network:

![github-small](https://raw.githubusercontent.com/shonvivier/LuaDigitRecognition/master/readme/1.PNG)
![github-small](https://raw.githubusercontent.com/shonvivier/LuaDigitRecognition/master/readme/2.PNG)
![github-small](https://raw.githubusercontent.com/shonvivier/LuaDigitRecognition/master/readme/4.PNG)
![github-small](https://raw.githubusercontent.com/shonvivier/LuaDigitRecognition/master/readme/7.PNG)
![github-small](https://raw.githubusercontent.com/shonvivier/LuaDigitRecognition/master/readme/9.PNG)

All weight and bias adjustments, forward, and backward propogation was done in Lua.

Try writing out your own digits and seeing the predictions here: https://www.roblox.com/games/4533433898/Handwritten-Digit-Classification
