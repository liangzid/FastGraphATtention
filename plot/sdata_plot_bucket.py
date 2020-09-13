from matplotlib import pyplot as plt

nodeVec=[2,4,6,8,10,12]
acc=[0.8240,0.8070,0.7920,0.7440,0.7790,0.7580]
# GAT_m=[0.8,14.1,56.6,81.5]
# nodeVecSpecial=[1000,5000,10000,12000]

xshow=nodeVec
yshow=[0,0.20,0.40,0.60,0.80,1.00]

plt.plot(nodeVec,acc,marker='D',mec='b',mfc='w',label='FastGAT(our method)')
# plt.plot(nodeVecSpecial,GAT_m,marker='*',mec='b',ms=10,label='GAT(baseline)')
plt.legend()
plt.xticks(xshow)

# plt.margins(0)

plt.subplots_adjust(bottom=0.10)
plt.xlabel('Number of Buckets') #X轴标签
plt.ylabel("accuarcy ") #Y轴标签
plt.yticks(yshow)

plt.savefig('./sdata_plot_bucket.png',dpi=1000)
