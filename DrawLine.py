import matplotlib.pyplot as plt

log_file_path = '/home/ubuntu/MOTRv2-trans/exps/motrv2DSText/run1_20query/output.log'
log_file_path1 = '/home/ubuntu/MOTRv2-transRR/exps_before/motrv2DSText_Resnet_backbone/run2_changeQuery_inter10_epoch20_drop5/output.log'
num_item = 0
loss = []
loss_angle = []
loss_bbox = []
loss_ce = []
loss_giou = []
num_item1 = 0
loss1 = []
loss_angle1 = []
loss_bbox1 = []
loss_ce1 = []
loss_giou1 = []

item = -1
item1 = -1
# 遍历数据，得到各指标值
with open(log_file_path, 'r') as file:
    for line in file:
        if 'Epoch: [' in line:
            # print(line)
            item = item + 1  
            if item%10==0:
                chars = line.split('  ')
                for char in chars:
                    if 'loss:' in char:
                        num_item = num_item + 1
                        indicator = char.split(' ')
                        loss.append(float(indicator[1]))
                    elif 'frame_0_loss_angle' in char or 'frame_1_loss_angle' in char or 'frame_2_loss_angle' in char or 'frame_3_loss_angle' in char or 'frame_4_loss_angle' in char:
                        indicator = char.split(' ')
                        loss_angle.append(float(indicator[1]))
                    elif 'frame_0_loss_bbox' in char or 'frame_1_loss_bbox' in char or 'frame_2_loss_bbox' in char or 'frame_3_loss_bbox' in char or 'frame_4_loss_bbox' in char:
                        indicator = char.split(' ')
                        loss_bbox.append(float(indicator[1]))
                    elif 'frame_0_loss_ce' in char or 'frame_1_loss_ce' in char or 'frame_2_loss_ce' in char or 'frame_3_loss_ce' in char or 'frame_4_loss_ce' in char:
                        indicator = char.split(' ')
                        loss_ce.append(float(indicator[1]))
                    elif 'frame_0_loss_giou' in char or 'frame_1_loss_giou' in char or 'frame_2_loss_giou' in char or 'frame_3_loss_giou' in char or 'frame_4_loss_giou' in char:
                        indicator = char.split(' ')
                        loss_giou.append(float(indicator[1]))


with open(log_file_path1, 'r') as file:
    for line in file:
        if 'Epoch: [' in line:
            # print(line)
            item1 = item1 + 1  
            if item1%10==0:
                chars = line.split('  ')
                for char in chars:
                    if 'loss:' in char:
                        num_item1 = num_item1 + 1
                        indicator = char.split(' ')
                        loss1.append(float(indicator[1]))
                    elif 'frame_0_loss_angle' in char or 'frame_1_loss_angle' in char or 'frame_2_loss_angle' in char or 'frame_3_loss_angle' in char or 'frame_4_loss_angle' in char:
                        indicator = char.split(' ')
                        loss_angle1.append(float(indicator[1]))
                    elif 'frame_0_loss_bbox' in char or 'frame_1_loss_bbox' in char or 'frame_2_loss_bbox' in char or 'frame_3_loss_bbox' in char or 'frame_4_loss_bbox' in char:
                        indicator = char.split(' ')
                        loss_bbox1.append(float(indicator[1]))
                    elif 'frame_0_loss_ce' in char or 'frame_1_loss_ce' in char or 'frame_2_loss_ce' in char or 'frame_3_loss_ce' in char or 'frame_4_loss_ce' in char:
                        indicator = char.split(' ')
                        loss_ce1.append(float(indicator[1]))
                    elif 'frame_0_loss_giou' in char or 'frame_1_loss_giou' in char or 'frame_2_loss_giou' in char or 'frame_3_loss_giou' in char or 'frame_4_loss_giou' in char:
                        indicator = char.split(' ')
                        loss_giou1.append(float(indicator[1]))
                    
                        
                    
            # print(chars)


# 模拟一些训练数据，您需要用实际的数据替换这部分
# items = list(range(0, num_item*5))
items = list(range(0, num_item))
items1 = list(range(0, num_item1))
# 绘制损失图
plt.figure(figsize=(20, 10))
plt.plot(items, loss, label='Loss_changeproposal')
plt.plot(items1, loss1, label='Loss_noChange')
# plt.plot(items, loss_angle, label='Loss_angle')
# plt.plot(items, loss_bbox, label='Loss_bbox')
# plt.plot(items, loss_ce, label='Loss_ce')
# plt.plot(items, loss_giou, label='Loss_giou')
plt.title('Training Loss decline chart')
plt.xlabel('Items (x100)')
plt.ylabel('Training loss')
# plt.yticks(range(int(min(loss)), int(max(loss)+1), 10))
# plt.yticks(range(int(min(loss_angle)), int(max(loss_angle)+1), 10))
plt.legend()
plt.grid(True)
plt.savefig('/home/ubuntu/MOTRv2-trans/Loss_line.jpg')
plt.show()

