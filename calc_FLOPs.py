from model.MDASR import mdasr1
from model.MDASR import mdasrcnew
from FLOPs.profile import profile

width = 360
height = 240
model = mdasrcnew.MDASR()
flops, params = profile(model, input_size=(1, 3, height, width))
print('IMDN_light: {} x {}, flops: {:.10f} GFLOPs, params: {}'.format(height,width,flops/(1e9),params))
