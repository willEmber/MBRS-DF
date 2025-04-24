from . import Identity
import torch.nn as nn
from . import get_random_int


class Combined(nn.Module):

	def __init__(self, list=None):
		super(Combined, self).__init__()
		if list is None:
			list = [Identity()]
		self.list = list
		# 添加一个属性来跟踪当前选择的噪声层索引
		self.current_index = 0

	def forward(self, image_and_cover):
		# 随机选择一个噪声层并保存索引
		self.current_index = get_random_int([0, len(self.list) - 1])
		return self.list[self.current_index](image_and_cover)
