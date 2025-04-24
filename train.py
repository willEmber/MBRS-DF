from torch.utils.data import DataLoader, Subset
from utils import *
from network.Network import *
from tqdm import tqdm

from utils.load_train_setting import *

# 添加配置变量，控制是否限制训练和验证数据量
limit_training_images = True  # 设置为False可恢复使用全部训练数据
training_images_limit = 1000  # 限制训练使用的图片数量

limit_validation_images = True  # 设置为False可恢复使用全部验证数据
validation_images_limit = 500  # 限制验证使用的图片数量

'''
train
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

network = Network(H, W, message_length, noise_layers, device, batch_size, lr, with_diffusion, only_decoder)

train_dataset = MBRSDataset(os.path.join(dataset_path, "train"), H, W)

# 限制训练数据数量
if limit_training_images:
    # 使用固定的随机种子以确保可重复性
    indices = list(range(len(train_dataset)))
    np.random.seed(42)  # 固定随机种子
    np.random.shuffle(indices)
    indices = indices[:training_images_limit]  # 只取前1000张图片
    train_dataset = Subset(train_dataset, indices)
    print(f"\nLimited training to {training_images_limit} images\n")

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

val_dataset = MBRSDataset(os.path.join(dataset_path, "validation"), H, W)

# 限制验证数据数量
if limit_validation_images:
    # 使用固定的随机种子以确保可重复性
    indices = list(range(len(val_dataset)))
    np.random.seed(43)  # 使用不同的随机种子
    np.random.shuffle(indices)
    indices = indices[:validation_images_limit]  # 只取指定数量的图片
    val_dataset = Subset(val_dataset, indices)
    print(f"\nLimited validation to {validation_images_limit} images\n")

val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

if train_continue:
	EC_path = "results/" + train_continue_path + "/models/EC_" + str(train_continue_epoch) + ".pth"
	D_path = "results/" + train_continue_path + "/models/D_" + str(train_continue_epoch) + ".pth"
	network.load_model(EC_path, D_path)

print("\nStart training : \n\n")

for epoch in range(epoch_number):

	epoch += train_continue_epoch if train_continue else 0

	running_result = {
		"error_rate": 0.0,
		"psnr": 0.0,
		"ssim": 0.0,
		"g_loss": 0.0,
		"g_loss_on_discriminator": 0.0,
		"g_loss_on_encoder": 0.0,
		"g_loss_on_decoder": 0.0,
		"d_cover_loss": 0.0,
		"d_encoded_loss": 0.0,
		"noise_type": ""  # 添加noise_type键来支持新的结果字典
	}

	start_time = time.time()

	'''
	train
	'''
	num = 0
	train_pbar = tqdm(train_dataloader, desc=f'Epoch {epoch} [Train]', leave=True)
	for _, images, in enumerate(train_pbar):
		image = images.to(device)
		# 生成随机消息
		message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], message_length))).to(device)
		# 训练网络
		result = network.train(image, message) if not only_decoder else network.train_only_decoder(image, message)
		# 仅训练判别器
		for key in result:
			# 修改此处，确保从 tensor 中提取标量值
			if key == "noise_type":  # 对字符串类型的键特殊处理
				running_result[key] = result[key]  # 只保存最后一个batch的噪声类型
			elif torch.is_tensor(result[key]):
				if result[key].numel() == 1:  # 检查是否为单元素张量
					running_result[key] += float(result[key].item())
				else:
					# 如果是多元素张量，可以使用均值或其他聚合方法
					running_result[key] += float(result[key].mean().item())
			else:
				running_result[key] += float(result[key])
		
		# 更新进度条显示当前的损失和指标
		display_info = {
			'g_loss': running_result['g_loss'] / (num + 1),
			'error_rate': running_result['error_rate'] / (num + 1),
			'noise_type': running_result['noise_type']  # 显示当前噪声类型
		}
		train_pbar.set_postfix(display_info)

		num += 1

	'''
	train results
	'''
	content = "Epoch " + str(epoch) + " : " + str(int(time.time() - start_time)) + "\n"
	for key in running_result:
		if key == "noise_type":
			content += key + "=" + running_result[key] + ","
		else:
			content += key + "=" + str(running_result[key] / num) + ","
	content += "\n"

	with open(result_folder + "/train_log.txt", "a") as file:
		file.write(content)
	print(content)

	'''
	validation
	'''

	val_result = {
		"error_rate": 0.0,
		"psnr": 0.0,
		"ssim": 0.0,
		"g_loss": 0.0,
		"g_loss_on_discriminator": 0.0,
		"g_loss_on_encoder": 0.0,
		"g_loss_on_decoder": 0.0,
		"d_cover_loss": 0.0,
		"d_encoded_loss": 0.0,
		"noise_type": ""  # 添加noise_type键来支持新的结果字典
	}

	start_time = time.time()

	saved_iterations = np.random.choice(np.arange(len(val_dataloader)), size=save_images_number, replace=False)
	saved_all = None

	num = 0
	val_pbar = tqdm(val_dataloader, desc=f'Epoch {epoch} [Validation]', leave=True)
	for i, images in enumerate(val_pbar):
		image = images.to(device)
		message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], message_length))).to(device)

		result, (images, encoded_images, noised_images, messages, decoded_messages) = network.validation(image, message)

		for key in result:
			# 修改此处，确保从 tensor 中提取标量值
			if key == "noise_type":  # 对字符串类型的键特殊处理
				val_result[key] = result[key]  # 只保存最后一个batch的噪声类型
			elif torch.is_tensor(result[key]):
				if result[key].numel() == 1:  # 检查是否为单元素张量
					val_result[key] += float(result[key].item())
				else:
					# 如果是多元素张量，可以使用均值或其他聚合方法
					val_result[key] += float(result[key].mean().item())
			else:
				val_result[key] += float(result[key])
				
		# 更新进度条显示当前的损失和指标
		display_info = {
			'g_loss': val_result['g_loss'] / (num + 1),
			'error_rate': val_result['error_rate'] / (num + 1),
			'noise_type': val_result['noise_type']  # 显示当前噪声类型
		}
		val_pbar.set_postfix(display_info)

		num += 1

		if i in saved_iterations:
			if saved_all is None:
				saved_all = get_random_images(image, encoded_images, noised_images)
			else:
				saved_all = concatenate_images(saved_all, image, encoded_images, noised_images)

	save_images(saved_all, epoch, result_folder + "images/", resize_to=(W, H))

	'''
	validation results
	'''
	content = "Epoch " + str(epoch) + " : " + str(int(time.time() - start_time)) + "\n"
	for key in val_result:
		if key == "noise_type":
			content += key + "=" + val_result[key] + ","
		else:
			content += key + "=" + str(val_result[key] / num) + ","
	content += "\n"

	with open(result_folder + "/val_log.txt", "a") as file:
		file.write(content)
	print(content)
	
	# 添加训练结果分析
	if epoch == 0 or (epoch + 1) % 5 == 0:  # 每5个epoch分析一次
		print("\n===== 训练结果分析 =====")
		print(f"训练集错误率: {running_result['error_rate'] / num:.4f}")
		print(f"验证集错误率: {val_result['error_rate'] / num:.4f}")
		
		# 比较训练集和验证集的性能差异
		error_diff = running_result['error_rate'] / num - val_result['error_rate'] / num
		print(f"错误率差异（训练-验证）: {error_diff:.4f}")
		
		if error_diff > 0.05:
			print("警告: 训练集错误率明显高于验证集，可能出现欠拟合现象")
		elif error_diff < -0.05:
			print("警告: 验证集错误率明显高于训练集，可能出现过拟合现象")
			
		# PSNR和SSIM分析
		print(f"训练集PSNR: {running_result['psnr'] / num:.2f}")
		print(f"验证集PSNR: {val_result['psnr'] / num:.2f}")
		print(f"训练集SSIM: {running_result['ssim'] / num:.4f}")
		print(f"验证集SSIM: {val_result['ssim'] / num:.4f}")
		
		# 损失分析
		print(f"训练集总损失: {running_result['g_loss'] / num:.4f}")
		print(f"验证集总损失: {val_result['g_loss'] / num:.4f}")
		print("===== 分析结束 =====\n")

	'''
	save model
	'''
	path_model = result_folder + "models/"
	path_encoder_decoder = path_model + "EC_" + str(epoch) + ".pth"
	path_discriminator = path_model + "D_" + str(epoch) + ".pth"
	network.save_model(path_encoder_decoder, path_discriminator)
