from matplotlib import pyplot as plt

nodeVec=[1000,5000,10000,12000,20000]
fastGAT_time=[1.3238,5.9442,41.0973,60.5840,166.1725]
GAT_time=[3.3200,67.2227,254.8308,351.8460]
nodeVecSpecial=[1000,5000,10000,12000]


plt.plot(nodeVec,fastGAT_time,marker='o',mec='r',mfc='w',label='FastGAT(our method)')
plt.plot(nodeVecSpecial,GAT_time,marker='*',mec='b',ms=10,label='GAT(baseline)')
plt.legend()
plt.xticks(nodeVec)

plt.margins(0)

plt.subplots_adjust(bottom=0.10)
plt.xlabel('Node of Log_10(numNode)') #X轴标签
plt.ylabel("time /epoch") #Y轴标签
pyplot.yticks(GAT_time)
#plt.title("A simple plot") #标题
plt.savefig('./sdata_plot_time.jpg',dpi = 900)
