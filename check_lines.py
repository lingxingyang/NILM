import os
import random

file_path = "C:/Users/h534294/Desktop/NILM_experiment/Our/txt/laptop_7k.txt"
save_path = "C:/Users/h534294/Desktop/NILM_experiment/Our/laptop/laptop_7k.txt"

FREQUENCY_SAMPLE =6960
GRID_FREQUENCY = 60
sap_per_cycle = int(FREQUENCY_SAMPLE/GRID_FREQUENCY)

 
def lines_to_delete(file_path, save_path):
    data = []
    data_filter = [] 
    indices_to_sap = []
    indices_to_data = []
    indices_to_del = []

    with open(file_path, 'r') as txt_file:
        lines = txt_file.readlines()

    for line in lines:
        columns = line.strip().split(',')
        data.append([float(columns[0]), float(columns[1])])
    
    #判断周期的初始
    for i in range(len(data) - 1):
        current_value = data[i][1]
        next_value =data[i + 1][1]
        #if (current_value * next_value) < 0:
            # indices_to_delete 是首周期的前一行
            #indices_to_delete.append(i + 1)
        if current_value >0 and  next_value <0:
            indices_to_sap.append(i + 1)

    #row = int(indices_to_delete[-1])
    
    data = [data[w] for w in range(indices_to_sap[0], len(data))]
    
    rows_to_delete = len(data) - indices_to_sap[-1]    #删除前indices_to_sap[0]个数据后，行号索引变了。rows_to_delete是需要删除的尾部数据数量
    lines_end = len(data) - indices_to_sap[0]   #末尾行号为lines_end
    num_lines = lines_end - rows_to_delete      # 列表更新后，总共num_lines个数据
    data1 = [data[q] for q in range(0,num_lines-1)]
   
    print(data1[-1])
    with open(save_path, 'w') as file:
        for b in data1:
            file.write(f"{b[0]},{b[1]}\n")
 


    with open(save_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        columns = line.strip().split(',')
        data_filter.append([float(columns[0]), float(columns[1])])
    
    for n in range(len(data_filter) - 1):
        current_value1 = data_filter[n][1]
        next_value1 =data_filter[n + 1][1]
        #indices_to_data是每个周期的末尾的行号
        if current_value1 >0 and  next_value1 <0:
            indices_to_data.append(n+1)
            #print(data_filter[indices_to_data[-1]])

    
    
    for k in range(len(indices_to_data)-1):
        #ilf左区间，irt右区间，实际是ilf+1行，irt+1行
    #一个周期117-234
        ilf = indices_to_data[k]+1
        irt = indices_to_data[k+1]

        #每个周期的行号差，判断数据是否整齐
        current_cycle = irt - indices_to_data[k]
        dif = current_cycle - sap_per_cycle
        #get_index = data[ilf:irt]
        #数据多了需要随机删除一个数据，少就删除整个周期
        
        if dif >=1:

            
           #从差值到每个周期的采样点数中随机选取dif个数进行删除
            #print("{}周期数据异常，多了{}个数据".format((k+1),dif))
            seed = 1
            while seed <= dif:
                idel = random.randint(1, sap_per_cycle)
                line_to_del = ilf + idel
                indices_to_del.append(line_to_del)
                seed +=1

        #数据少了就删除这个周期的所有数据
        elif dif <0:
            
            #print("{}周期数据异常,少了{}个数据".format((k+1),dif))
            for x in range(current_cycle):
                indices_to_del.append(x+ilf)
    
    
    for i in range(indices_to_data[0]):
        indices_to_del.append(i)
    for w in range(indices_to_data[-1],len(data_filter)):
        indices_to_del.append(w)    

    for h in sorted(indices_to_del,reverse=True):
        del data_filter[h]
    
       
    with open(save_path, 'w') as f1:
        for m in data_filter:
            f1.write(f"{m[0]},{m[1]}\n")
    return 


lines_to_delete(file_path, save_path)
#lines_to_delete(save_path, save_path)



