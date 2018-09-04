import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
myfont = fm.FontProperties(fname=u'C:\Windows\Fonts\simsun.ttc',size=10)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles
#%% 生成分组多维正态分布的数据
def create_data(show_scatter=False):
    # 生成协方差cov=2，维度n_features=2，类别n_classes=2,特征均值mean=(0,0)
    X1, y1 = make_gaussian_quantiles(mean=(0,0),cov=2.,n_samples=200, n_features=2,
                                     n_classes=2, random_state=1)
    # 生成协方差cov=2，维度n_features=2，类别n_classes=2,特征均值mean=(3, 3)
    X2, y2 = make_gaussian_quantiles(mean=(3, 3), cov=1.5,n_samples=300, 
                                     n_features=2,n_classes=2, random_state=1)
    # 合并X1，X2
    X = np.concatenate((X1, X2))
    y = np.concatenate((y1, - y2 + 1))
    # 图示样本的分布
    if show_scatter == True:
        fig,axs =plt.subplots(1,2)
        axs[0].scatter(X1[:,0],X1[:,1],c=y1)
        axs[0].set_title('类别0的分布',fontproperties=myfont)
        axs[1].scatter(X2[:,0],X2[:,1],c=y2)
        axs[1].set_title('类别1的分布',fontproperties=myfont)
        fig.show()
    return X,y
#%% 绘制AdaBoostClassifier分类效果图
def plt_AdaBoostClassifier(clf,*data):
    X,y = data
    fig,axs = plt.subplots(1,2)
    #
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
    np.arange(y_min, y_max, plot_step))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    axs[0].contourf(xx, yy, Z, cmap=plt.cm.Paired)
    axs[0].axis("tight")
    # 绘制训练集
    for i, n, c in zip(range(2), class_names, plot_colors):
        idx = np.where(y == i)
        axs[0].scatter(X[idx, 0], X[idx, 1],
                        c=c, cmap=plt.cm.Paired,
                        s=20, edgecolor='k',
                        label="类 %s" % n)
    axs[0].set_xlim(x_min, x_max)
    axs[0].set_ylim(y_min, y_max)
    axs[0].legend(loc='best',prop=myfont)
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    axs[0].set_title('决策边界',fontproperties=myfont)
    # Plot the two-class decision scores
    twoclass_output = clf.decision_function(X)
    plot_range = (twoclass_output.min(), twoclass_output.max())
    
    for i, n, c in zip(range(2), class_names, plot_colors):
        axs[1].hist(twoclass_output[y == i],
                 bins=10,
                 range=plot_range,
                 facecolor=c,
                 label='类 %s' % n,
                 alpha=.5,
                 edgecolor='k')
    x1, x2, y1, y2 = plt.axis()
    axs[1].axis((x1, x2, y1, y2*1.2))
    axs[1].legend(loc='upper right',prop=myfont)
    axs[1].set_ylabel('样本数量',fontproperties=myfont)
    axs[1].set_xlabel('得分',fontproperties=myfont)
    axs[1].set_title('决策得分',fontproperties=myfont)
    fig.set_tight_layout(True)
    fig.subplots_adjust(wspace=0.35)
    fig.show()   
    return    
    
#%% 主程序
if __name__ == '__main__':    
    X,y =create_data(show_scatter=True)
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),algorithm="SAMME",n_estimators=200)
    
    clf.fit(X, y)
    plot_colors = "br"
    plot_step = 0.02
    class_names = "AB"
    plt_AdaBoostClassifier(clf,X,y)
