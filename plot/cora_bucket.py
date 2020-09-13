from matplotlib import pyplot as plt

nodeVec=[2,4,6,8]
time=[293.0392,251.9738,226.7881,211.4224]
# GAT_m=[0.8,14.1,56.6,81.5]
# nodeVecSpecial=[1000,5000,10000,12000]

xshow=nodeVec
yshow=[200,220,240,260,280,300]

plt.plot(nodeVec,time,marker='D',mec='b',mfc='w',label='FastGAT(our method)')
# plt.plot(nodeVecSpecial,GAT_m,marker='*',mec='b',ms=10,label='GAT(baseline)')
plt.legend()
plt.xticks(xshow)

# plt.margins(0)

plt.subplots_adjust(bottom=0.10)
plt.xlabel('Number of Buckets') #X轴标签
plt.ylabel("time /s ") #Y轴标签
plt.yticks(yshow)

plt.savefig('./cora_bucket.png',dpi=1000)
