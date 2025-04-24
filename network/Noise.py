from . import *
from .noise_layers import *


class Noise(nn.Module):

	def __init__(self, layers):
		super(Noise, self).__init__()
		self.layers = []
		self.layer_types = []
		
		# 将字符串表示的噪声层转换为实际的nn.Module对象
		for i in range(len(layers)):
			layer = eval(layers[i])
			self.layers.append(layer)
			
			# 检测并记录噪声类型
			if isinstance(layer, DeepfakeProxy) or isinstance(layer, RegionalDestruction) or isinstance(layer, FrequencyModulation):
				self.layer_types.append("deepfake")
			else:
				self.layer_types.append("benign")
				
		# 创建顺序模块
		self.noise = nn.Sequential(*self.layers)
		
		# 记录当前使用的噪声层索引
		self.current_index = 0

	def forward(self, image_and_cover):
		# 如果是Combined噪声层，需要特殊处理，因为它会随机选择一个子噪声层
		if len(self.layers) == 1 and isinstance(self.layers[0], Combined):
			# 先获取当前选择的索引（在Combined的forward方法中设置）
			self.current_index = self.layers[0].current_index
			# 返回Combined处理后的图像
			noised_image = self.noise(image_and_cover)
			return noised_image
		else:
			# 常规情况，直接应用噪声
			noised_image = self.noise(image_and_cover)
			return noised_image
	
	def get_current_noise_type(self):
		"""获取当前使用的噪声类型，返回'benign'或'deepfake'"""
		if len(self.layers) == 1 and isinstance(self.layers[0], Combined):
			# 检查Combined中当前选择的噪声类型
			combined_layer = self.layers[0]
			sub_layer = combined_layer.list[self.current_index]
			
			if (isinstance(sub_layer, DeepfakeProxy) or 
				isinstance(sub_layer, RegionalDestruction) or 
				isinstance(sub_layer, FrequencyModulation)):
				return "deepfake"
			else:
				return "benign"
		elif len(self.layer_types) > 0:
			# 非Combined情况，返回已记录的类型
			return self.layer_types[0]
		else:
			# 默认情况
			return "benign"
