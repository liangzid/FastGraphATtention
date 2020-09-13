from matplotlib import pyplot as plt

nodeVec=[1000,5000,10000,12000,20000]
fastGAT_m=[0.4,1.7,16.2,22.6,61.9]
GAT_m=[0.8,14.1,56.6,81.5]
nodeVecSpecial=[1000,5000,10000,12000]

xshow=[1000,5000,10000,15000,20000,25000]
yshow=[0,20,40,60,80,100]

plt.plot(nodeVec,fastGAT_m,marker='o',mec='r',mfc='w',label='FastGAT(our method)')
plt.plot(nodeVecSpecial,GAT_m,marker='*',mec='b',ms=10,label='GAT(baseline)')
plt.legend()
plt.xticks(xshow)

# plt.margins(0)

plt.subplots_adjust(bottom=0.10)
plt.xlabel('Number of Node') #X轴标签
plt.ylabel("memory % ") #Y轴标签
plt.yticks(yshow)

plt.savefig('./sdata_plot_m.png',dpi=1000)
