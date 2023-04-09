trainRoot = './data/captcha/train/'
testRoot = './data/captcha/test/'
demoRoot = './data/captcha/demo/'

# target web styles
charNumber = 4
charLength = 62  
ImageWidth_custom = 64
ImageHeight_custom = 24
font_sizes_custom=(20, 22, 23)

# fixed params
ImageWidth = 32  # final size as ResNet input
ImageHeight = 32  # final size as ResNet input

# revisable params
learningRate = 1e-3
totalEpoch = 200
batchSize = 256
printCircle = 100
testCircle = 100
testNum = 30
saveCircle = 200