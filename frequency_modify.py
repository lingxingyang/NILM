import os
import pandas as pd
import csv

file_path = "C:/YLX/NILM/Lenet/data/submetered_new/"
save_path = 'C:/Users/h534294/Desktop/data_3K/'

'''
df = pd.read_csv(file_path,usecols=[0,1])

data = [df.iloc[i].tolist() for i in range(len(df)) if i % 5 ==2]

current = [data[0] for data in data]
voltage = [data[1] for data in data]
#print("current ",current)
#print("voltage ",voltage)

with open(save_path,'w',newline="") as csv_file:
    csvwrite = csv.writer(csv_file)
    for e in range(len(current)):
        csvwrite.writerow([current[e],voltage[e]])

'''  
#批量压缩数据，从3W-3K
def bulk_modify(file_path, save_path):
    files = os.listdir(file_path)
    for file in files:
        path = os.path.join(file_path, file)
        
        # 读取CSV并跳过每10行中的行
        df = pd.read_csv(path, usecols=[0, 1], skiprows=lambda x: x > 0 and x % 10 != 0)
        
        print("df ", df) 
        
        # 用列名而不是列索引来提取数据
        current = df.iloc[:, 0].tolist()  # 第一列
        voltage = df.iloc[:, 1].tolist()  # 第二列
        
        
        # 确保保存路径存在
        save_file_path = os.path.join(save_path, file)
        
        # 保存数据到CSV文件
        with open(save_file_path, 'a', newline="") as csv_file:
            csvwrite = csv.writer(csv_file)
            # 如果文件是空的，可以写入标题行
            
            # 写入数据行
            for c, v in zip(current, voltage):
                csvwrite.writerow([c, v])

    return

# 调用函数时请确保文件路径正确
bulk_modify(file_path, save_path)
  
