
import numpy as np
import os
import matplotlib.pyplot as plt
from cycler import cycler
import pandas as pd

filepath ="C:/Users/h534294/Desktop/NILM_experiment/Our/log/hairdyer_3k.log"
#csvpath = "C:/Users/h534294/Desktop/NILM_experiment/Our/hairdyer_3k.log"
newfile= "C:/Users/h534294/Desktop/NILM_experiment/open-source-2014/data_compress/Air Conditioner/n_16.txt"
savepath = "C:/Users/h534294/Desktop/NILM_experiment/open-source-2014/data_compress/Air Conditioner/"
HIGHEST_ODD_HARMONIC_ORDER = 21
SAMPLE_FREQUENCY = 3000
appliance_name = "n_16" 

SAMPLE_CYCLES = 12

GRID_FREQUENCY=60
# 采样间隔
sap_interval=1/SAMPLE_FREQUENCY
# 每个电网周期的样本数量
sap_num_cycle=SAMPLE_FREQUENCY/GRID_FREQUENCY

monochrome = (cycler('color', ['k']))

#将log转换为txt且删除多余行和列
def convert_log_to_txt(filepath,newfile):
    with open(filepath,'r') as log_file:
        log_content = log_file.read()
    
    with open(newfile,'w',encoding='utf-8') as txt_file:
        txt_file.write(log_content)

    with open(newfile,'r',encoding='utf-8') as txt_file:   
        lines = txt_file.readlines()
        lines = lines[3:]
        del lines[-4:]
        for i in range(len(lines)):
            lines[i] = lines[i][4:]
    with open(newfile, "w",encoding='utf-8') as txt_file:
        txt_file.writelines(lines)    
    return(filepath,newfile)

def read_csv(csvpath):
    df = pd.read_csv(csvpath,usecols=[0,1])

    current = df.iloc[:,0].tolist()
    voltage = df.iloc[:,1].tolist()
    #print("current ",current)
    #print("voltage ",voltage)

    return(current,voltage)

#从txt文件中取出两列数据
def get_VI_samples(newfile):
    current = []
    voltage = []
    with open(newfile, "r",encoding='utf-8') as file:
        lines = file.readlines()
        current = []
        voltage = []
        for line in lines:
            line = line.replace('\n', '') 
            #print(line)
            #x = line[:4]
            line = line.strip()
            # 使用逗号分隔行内容
            parts = line.split(',')
            if len(parts) > 1:
                # 获取逗号左侧的内容
                current.append(parts[0])
                voltage.append(parts[1])
            '''
            if (len(current) != (len(current))):
                print("Sample data missing")
            else:
                sap_count = len(current) 
'''
    return(current,voltage)

'''
#从excel中取数据
def get_Excel():
    df = pd.read_excel('C:/Users/h534294/Desktop/1.xlsx')  
    current = df['Column1'].tolist()
    voltage = df['Column2'].tolist()    
    return (current, voltage)
'''


def convert_to_float(value_list):
    float_list = []
    for i,item in enumerate(value_list):
        try:
            float_value = float(item)
            float_list.append(float_value)
        except ValueError:
            print("Error converting to float:", item,i)
    return float_list

     


#这段代码是在找到信号中稳定的信号段，计算了均值，均值阈值判断>0.03
# 判断信号段中的所有值是否在均值的 90% 到 110% 之间。
#如果所有值都在这个范围内，认为信号段稳定。
def reconstruct(signal_in_fft,highest_harmonic_order,length=None):
    dict_harmonics={}
    harmonic_pairs=[]
    #获取信号在频域不为零的所有指标
    harmonic_indices=np.where(abs(signal_in_fft)!=0)  
    #将索引对的列表构造为元组(负频率和正频率)
    for i in range(len(harmonic_indices[0])//2):    
        harmonic_pairs.append((harmonic_indices[0][i],harmonic_indices[0][-i-1]))    
    mag_list=[]
    harmonic_number=1   # Starts with first harmonic
    for i,j in harmonic_pairs:  
        if harmonic_number<=highest_harmonic_order:
            harmonic_in_time_domain=np.zeros(len(signal_in_fft)).astype('complex128')  # Zero complex array in range of signal
            harmonic_in_time_domain[i]=signal_in_fft[i]          # First index of pair
            harmonic_in_time_domain[j]=signal_in_fft[j]          # Second index of pair 
            harmonic_in_time_domain=np.fft.ifft(harmonic_in_time_domain)        # Inverse Fourier Transform of pair
            mag_list.append(max(harmonic_in_time_domain.real))
            if length==None:
                dict_harmonics[harmonic_number]=harmonic_in_time_domain.real
            else:
                dict_harmonics[harmonic_number]=harmonic_in_time_domain.real[0:length]
            harmonic_number+=1          
    THD=sum(np.square(mag_list[1:]))**0.5/mag_list[0]           # Total Harmonic Distortion (THD) formula
    return dict_harmonics,THD             #Returns harmonic dictionary and THD value

#返回绝对差数组中最小值的索引，这个索引对应于 signal 中最接近 value 的元素。
def find_near_index(signal, value):
   
    #np.abs用来计算绝对值
    #argmin用于返回数组中最小值的索引
    idx = (np.abs(signal - value)).argmin()
    return idx

#计算电压、电流的有效值（均方根）
def generate_rms(signal,mode=None):   
    #采样点数量为n
    n = len(signal)   
    #duration采样时间
    duration=n/SAMPLE_FREQUENCY 
    #np.linspace 函数用于生成一个在指定范围内均匀分布的数组，生成了一个时间戳。从0到采样时间，长度是采样点数 
    time   = np.linspace(0,duration,n)
    #定义一个numpy数组
    signal_rms=np.array([])
    '''if mode=='half_cycle':
        resolution=sap_num_cycle/2
    elif mode=='full_cycle':
        resolution=sap_num_cycle
    else:
        resolution=SAMPLE_CYCLES'''
    #resolution 每个周期的采样点数
    resolution=sap_num_cycle
    #生成等间隔的索引数组，从0到采样时间，
    interv=np.arange(0,len(time),resolution)
    for i in interv:
        signal_pow=0                      
        if (i+resolution)<=(len(time)):  
            signal_pow=[signal[j]**2 for j in range(int(i),int(i+(resolution)))]
            signal_pow=sum(signal_pow)
            i_rms=[np.sqrt(signal_pow/(resolution))]*int(resolution)
            signal_rms=np.concatenate((signal_rms, i_rms), axis=None)  
        else:
            signal_pow=[signal[j]**2 for j in range(int(i),int(len(time)-i))]
            signal_pow=sum(signal_pow)
            i_rms=[np.sqrt(signal_pow/(len(time)-i))]*int(len(time)-i)
            signal_rms=np.concatenate((signal_rms, i_rms), axis=None)  
    return signal_rms




#此处将原来的函数直接修改为full_cycle模式，n_cycle = sap_num_cycle ,每个周期采样点数量

def get_indices(signal_rms,mode=None,sample_cycles=None,aggregated=0):
    sample_dict={}
    sample_frag=[]
    n = len(signal_rms) 
    #samples_per_cycle一个周期的样本数
    samples_per_cycle=SAMPLE_FREQUENCY/GRID_FREQUENCY
    if mode=='half_cycle':
        resolution=samples_per_cycle/2
    else:
        resolution=samples_per_cycle
   
    #med_rms=np.mean(signal_rms)
    if sample_cycles==None:
        
        sample_cycles=12
        for k in range(int(n/(resolution)-sample_cycles*samples_per_cycle/resolution)+1):
            inf=int(k*resolution)
            sup=int(inf+sample_cycles*samples_per_cycle)        
            med=np.mean(signal_rms[inf:sup])     
            sample_dict[(inf,sup)]=np.var(signal_rms[inf:sup]/med)
        indices=min(sample_dict, key=sample_dict.get)
        indices=list(indices)
        return indices
    elif aggregated==0:                
        for k in range(int(n/(resolution)-sample_cycles*samples_per_cycle/resolution)+1):
            inf=int(k*resolution)
            sup=int(inf+sample_cycles*samples_per_cycle)  
            #把每（Sample_cycle个采样周期*每周期的采样点数）放在一个索引间隔中，np.mean是计算数组元素的算术平均值。此处计算了数组中信号有效值的算术平均数。   
            #为什么要计算算术平均数，此处可能是判断信号段是否稳定，是否需要处理噪声等影响
            med=np.mean(signal_rms[inf:sup])                   
            flag=0            
            if med>0.03:
                #if all是用来检查列表中所有元素是否满足条件。
                #此处为什么是列中所有信号的均方根值处于0.9*med-1.1*med，依旧是用来判断信号是否稳定
                if all(signal_rms[inf:sup]>0.9*med) and all(signal_rms[inf:sup]<1.1*med):
                    #np.var是计算方差，主要用于筛选出稳定信号段
                    sample_dict[(inf,sup)]=np.var(signal_rms[inf:sup]/med)
        #找到某个信号段在不同段中的最小方差
        if sample_dict!={}:            
            indices=min(sample_dict, key=sample_dict.get)
            indices=list(indices)           
            return indices
        else:
            return None
    else:
        k=0
        while k<=int(n/(samples_per_cycle/2)-sample_cycles):
            inf=int(k*samples_per_cycle/2)
            sup=int(inf+samples_per_cycle*sample_cycles)        
            med=np.mean(signal_rms[inf:sup])                   
            flag=0          
            if med>0.01:
                for j in range(2*sample_cycles):   
                    med_local=(signal_rms[inf+(j-1)*(int(samples_per_cycle/2))]+signal_rms[inf+j*(int(samples_per_cycle/2))])/2                         
                    if med_local>1.01*med or med_local<0.99*med:
                        flag=1
                        break
                if flag==0:
                    if sample_frag!=[]:
                        if sample_frag[-1][1]==inf:
                            sample_frag[-1][1]=sup
                        else:
                            sample_frag.append([inf,sup])
                    else:
                        sample_frag.append([inf,sup])
                    k+=2*sample_cycles-1
            k+=1
        print(sample_frag)
        return sample_frag
 
#计算相位角
def lag_value_in_degrees(current,voltage): 
    # Samples per cycle
    current = np.array(current)
    voltage = np.array(voltage)
    #print("lag_value_in_degrees current:",current)
    #print("lag_value_in_degrees voltage:",voltage)
    
    #每个周期的采样点数
    samples_per_cycle=int(SAMPLE_FREQUENCY/GRID_FREQUENCY)
    
    #I最接近0的位置，目的应该是从非0值开始取电流值
    i_cross_zero=find_near_index(current,0)
    #i_cross_zero+int(samples_per_cycle/2)是波峰位置的索引，+1后是下半个周期的开始
    v_cross_zero=find_near_index(voltage[i_cross_zero:i_cross_zero+int(samples_per_cycle/2)+1],0)+i_cross_zero
    #print("v_cross_zero",v_cross_zero)
    if i_cross_zero-v_cross_zero<-samples_per_cycle/4:
        lag=-int(i_cross_zero+samples_per_cycle/2-v_cross_zero)*360/samples_per_cycle
    else:
        lag=-int(i_cross_zero-v_cross_zero)*360/samples_per_cycle
        
    return lag



#计算电流信号和电压信号之间的相位差。通过找到电流和电压信号的零交叉点位置，并根据这些位置计算相位差。

#current_fft_amp,current_fft_phase=filter_harmonics(current,21)
#voltage_fft_amp,voltage_fft_phase=filter_harmonics(voltage,1)
def filter_harmonics(signal,highest_harmonic_order=None):
    
    signal=np.array(signal,dtype=np.float32) 
    # 计算采样间隔,grid_frequency为50HZ
    sap_interval=1/SAMPLE_FREQUENCY

    # 计算每个电网周期的样本数量
    sap_num_cycle=SAMPLE_FREQUENCY/GRID_FREQUENCY
    # 样本数
    n=len(signal)
    # 计算的是整数个周期的剩余样本数量
    sap_remainder = n % sap_num_cycle
    # 计算周期数
    #num_cycle = n // sap_num_cycle
    # 在做FFT之前需要保证样本数为整数周期，以下为截取整数周期的样本
    if sap_remainder !=0:
        signal=signal[:-sap_remainder]
        n = len(signal) 
    
    # 快速傅里叶变换
    fft_signal=np.fft.fft(signal,n)
    # 频域上的信号幅度
    fft_signal_amp=np.abs(fft_signal)
    
    # 频域上的信号幅度
    fft_signal_phase=np.angle(fft_signal)
    #print(fft_signal_phase)
    # 计算频率轴
    freq_axes = np.fft.fftfreq(n, d=sap_interval)

    # 获取谐波频率的索引，初始化一个谐波频率的索引列表
    harmonic_indices=[]
  
    if highest_harmonic_order==None: 
        #生成负频率范围内的谐波频率索引。               
        first_half=np.arange(-GRID_FREQUENCY,min(freq_axes),-GRID_FREQUENCY)   
        #生成正频率范围内的谐波频率索引。                                  
        second_half=np.arange(GRID_FREQUENCY,max(freq_axes),GRID_FREQUENCY)   
        #获取每个谐波频率附近的频率索引：
        harmonic_indices=np.append(first_half,second_half,axis=0) 
    else:                                                           
        first_half=np.arange(-GRID_FREQUENCY,-GRID_FREQUENCY*(highest_harmonic_order+1),-GRID_FREQUENCY)    # to +harmonic_order*fn                                     
        second_half=np.arange(GRID_FREQUENCY,GRID_FREQUENCY*(highest_harmonic_order+1),GRID_FREQUENCY)    
        harmonic_indices=np.append(first_half,second_half,axis=0)
    # Extract frequency indices around each harmonic order frequency
    ind_freq=[np.where((freq_axes >= (harmonic_indices[i]-GRID_FREQUENCY/10)) & (freq_axes <= (harmonic_indices[i]+GRID_FREQUENCY/10))) for i in range(len(harmonic_indices))]
    indices=[]
    for i in ind_freq:
        index_max=np.where(fft_signal_amp == max(fft_signal_amp[i])) # 获取fft_signal为最大值的索引
        ind=np.intersect1d(index_max,i)    #丢弃其余的索引
        indices.append(ind)              # 创建一个索引列表
    
    # 创建仅包含谐波分量的 FFT 信号
    fft_signal_clean_amp=np.zeros(len(fft_signal_amp))
    #print("fft_signal_clean_amp:",fft_signal_clean_amp)
    fft_signal_clean_phase=np.zeros(len(fft_signal_phase))
    #print("fft_signal_clean_phase:",fft_signal_clean_phase)


    for i in indices:
        fft_signal_clean_amp[i]=fft_signal_amp[i]           # fft_mjsignal clean = fft_signal at indices, 0 otherwise
        fft_signal_clean_phase[i]=fft_signal_phase[i]   

    return fft_signal_clean_amp,fft_signal_clean_phase # Returns magnitude and phase signal without noise in frequency domain

def construct_harmonic_dict(signal_dict,highest_harmonic_order):
    
    #自建一个谐波字典
    harmonic_dict = {
        'mean_lag':[],
        'mean_THD_current':[],
        'max_current':[],
        'first_harmonic_mag':[],
        'device':{},
        'harmonics_proportions':{}
    }
    harmonic_list = range(1,highest_harmonic_order+1)
    harmonic_dict['device'][appliance_name]={'THD_current':None, 'THD_voltage':None,'harmonic_order':{}}

    for harmonic_order in harmonic_list:
            harmonic_dict['device'][appliance_name]['harmonic_order'][harmonic_order]={'current': [],'voltage':[]}                    
            harmonic_dict['harmonics_proportions'][harmonic_order]=[]
            
    indices=signal_dict['indices']  
    current=signal_dict['current']   
    voltage=signal_dict['voltage']
    voltage=voltage[indices[0]:indices[1]]     
    current=current[indices[0]:indices[1]]
    #此处电压电流是有值的
    
    current_fft_amp,current_fft_phase=filter_harmonics(current,highest_harmonic_order)
    current_fft=current_fft_amp*np.exp(current_fft_phase*1j)
    current_decomposed,THD_current=reconstruct(current_fft,21)
    #print("current_decomposed:",current_decomposed)
    harmonic_dict['device'][appliance_name]['THD_current']=THD_current

    voltage_fft_amp,voltage_fft_phase=filter_harmonics(voltage,1)
    voltage_fft=voltage_fft_amp*np.exp(voltage_fft_phase*1j)
    voltage_decomposed,THD_voltage=reconstruct(voltage_fft,1)
    
    harmonic_dict['device'][appliance_name]['harmonic_order'][1]['voltage']=(voltage_decomposed[1])
    harmonic_dict['device'][appliance_name]['THD_voltage']=THD_voltage
    harmonic_dict['first_harmonic_mag'].append(max(current_decomposed[1]))
    lag=lag_value_in_degrees(current,voltage)
    
    harmonic_dict['mean_lag'].append(lag)
    harmonic_dict['mean_THD_current'].append(THD_current)
    harmonic_dict['max_current'].append(max(current))

    for harmonic_order in current_decomposed:         
        harmonic_dict['device'][appliance_name]['harmonic_order'][harmonic_order]['current']=(current_decomposed[harmonic_order])
        harmonic_dict['harmonics_proportions'][harmonic_order].append(max(current_decomposed[harmonic_order])/max(current_decomposed[1]))           
    
    harmonic_dict['mean_lag']=int(np.mean(harmonic_dict['mean_lag'])) 
    harmonic_dict['mean_THD_current']=np.mean(harmonic_dict['mean_THD_current'])
    harmonic_dict['max_current']=max(harmonic_dict['max_current'])

    for harmonic_order in harmonic_dict['harmonics_proportions']:
        harmonic_dict['harmonics_proportions'][harmonic_order]=np.mean(harmonic_dict['harmonics_proportions'][harmonic_order])
    
    return harmonic_dict
#这个函数应该计算电流和电压信号之间的相位差（以度为单位）
def shift_phase(current,voltage,lag=0): 
    #samples_per_cycle每个周期样本数
    samples_per_cycle=int(SAMPLE_FREQUENCY/GRID_FREQUENCY)
    current = np.array(current)
    voltage = np.array(voltage)
    phase = int(lag_value_in_degrees(current,voltage)*samples_per_cycle/360)
    
    phase+=lag
      
    if int(phase)>0:
        current=current[int(phase):]
        voltage=voltage[:-int(phase)]        
    elif int(phase)<0:
        current=current[:int(phase)]
        voltage=voltage[int(abs(phase)):]  
    return current,voltage,phase
#odd为0时，处理的是所有谐波，odd为1仅处理奇次谐波
def harmonics_selection(harmonic_dict,appliance_name,lag=None,odd=0):
    voltage=harmonic_dict['device'][appliance_name]['harmonic_order'][1]['voltage']
    
    current=np.zeros(len(harmonic_dict['device'][appliance_name]['harmonic_order'][1]['current']))

    if odd==0:
        for i in range(1,HIGHEST_ODD_HARMONIC_ORDER+1,1):
            harmonic=harmonic_dict['device'][appliance_name]['harmonic_order'][i]['current']
            current+=harmonic
    else:
        for i in range(1,HIGHEST_ODD_HARMONIC_ORDER+1,2):
            harmonic=harmonic_dict['device'][appliance_name]['harmonic_order'][i]['current']
            current+=harmonic
    if lag==1:
        current,voltage,phase=shift_phase(current,voltage,harmonic_dict['mean_lag'])
    if lag==0:
        current,voltage,phase=shift_phase(current,voltage,lag)

    return current,voltage
def save_trajectory(filepath,appliance_name,voltage,current,mode=0):

    fig = plt.figure(frameon = False)
    fig.set_size_inches(1, 1)   
    ax = plt.axes()
    ax.set_prop_cycle(monochrome)
    ax.set_axis_off()
    fig.add_axes(ax)

    img=plt.plot(voltage,current) 
    if not os.path.exists(f"{filepath}/{appliance_name}"):
            os.makedirs(f"{filepath}/{appliance_name}")
    plt.savefig(f"{filepath}/{appliance_name}/_" + str(mode) + ".png",dpi=128) 

    plt.close(fig)
                   
def generate_VI_images(harmonic_dict,filepath=savepath):     
  
    max_current=[] 
    if harmonic_dict['mean_THD_current']<0.05:
        max_current.append(harmonic_dict['max_current'])         
    for appliance_name in harmonic_dict['device']:                                    
        mode=0
        odd=0
        #for i in range(2):
        for i in range(1):                                 
            current,voltage=harmonics_selection(harmonic_dict,appliance_name,lag=i,odd=odd)
            
            save_trajectory(filepath,appliance_name,voltage,current,mode) 
            
            mode+=1 
            odd+=1
            
            #current,voltage=harmonics_selection(harmonic_dict,appliance_name,lag=i,odd=odd)
            #save_trajectory(filepath,appliance_name,voltage,current,mode)
            #mode+=1
            #odd-=1
            
    print(f"V-I trajectories saved in '{filepath}'\n")




#convert_log_to_txt(filepath,newfile)
current,voltage = get_VI_samples(newfile)
#current,voltage = read_csv(csvpath)
current = convert_to_float(current)
voltage = convert_to_float(voltage)

n=len(current)
#print(current)
# 计算的是整数个周期的剩余样本数量
sap_remainder = n % sap_num_cycle
# 计算周期数
num_cycle = n // sap_num_cycle

current_rms=generate_rms(current)
indices=get_indices(current_rms,mode='full_cycle',sample_cycles=SAMPLE_CYCLES) 
#创建一个关于电压信号和电流信号的字典
signal_dict = {

    'current': current,
    'voltage': voltage,
    #'index': list(range(len(current))),
    'indices': indices,
}
#print(signal_dict)
harmonic_dict = construct_harmonic_dict(signal_dict,HIGHEST_ODD_HARMONIC_ORDER) 
generate_VI_images(harmonic_dict,filepath=savepath)          
    