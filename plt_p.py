from util.utils import kde_data2
from matplotlib import pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D as ax3d
import numpy as np
import pandas as pd
from matplotlib import dates
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages as pdf


preprocess_max = {
    'BJTaxi':1292.0,
    'citybike':526.0,
    'nyctaxi':9806.0
}


def makecolor(fracs, rgb=[1,0,0], alpha=0.5):
    return [[rgb[0]*x, rgb[1]*x, rgb[2]*x, alpha] for x in fracs]



def plt_3d(trgt):

    row, col = trgt.shape[1], trgt.shape[2]
    x, y = np.arange(0, row), np.arange(0, col)
    x, y = np.meshgrid(x, y)
    z = trgt[-5, :, :, 0]
    fig = plt.figure()
    ax = ax3d(fig)
    ax.plot_surface(x, y, z, rstride=3, cstride=3, cmap='rainbow')
    dz = z.ravel()
    offset = dz + np.abs(dz.min())
    fracs = offset / offset.max()
    # norm = colors.Normalize(fracs.min(), fracs.max())
    # color_values = cm.jet(norm(fracs.tolist()))
    cmp = plt.get_cmap('Blues')
    cnorm = colors.Normalize(vmin=0, vmax=1)
    scalar = cm.ScalarMappable(norm=cnorm, cmap=cmp)
    color_values = np.array([scalar.to_rgba(x) for x in fracs])
    # color_values = np.multiply(color_values, [1,1,1,0.8])

    ax.bar3d(x.ravel(), y.ravel(), np.zeros_like(y.ravel()), dx=1, dy=1, dz=z.ravel(), color=color_values)
    ax.view_init(75, 165)
    # ax.set_axis_off()
    # plt.show()
    pp = pdf('figs/0.pdf')
    plt.savefig(pp, format='pdf')
    pp.close()

    fig = plt.figure()
    ax = ax3d(fig)
    gx, gy = np.mgrid[0:31:160j, 0:31:160j]
    gxy = np.array([x.ravel(), y.ravel()])
    gz = griddata(gxy.transpose(), z.ravel(), (gx, gy), method='cubic')
    gz = (gz - gz.min()) / gz.sum()

    z1 = trgt[-10, :, :, 1]
    gz1 = griddata(gxy.transpose(), z1.ravel(), (gx, gy), method='cubic')
    ax.plot_surface(gx, gy, gz, rstride=1, cstride=1, cmap='Reds')
    # ax.plot_surface(gx-8, gy+24, gz1+1000, rstride=1, cstride=1, cmap='Reds')
    # ax.plot_surface(gx, gy, -gz-50, rstride=1, cstride=1, cmap='rainbow')
    ax.view_init(75,165)
    # ax.set_axis_off()
    # plt.show()
    pp = pdf('figs/3.pdf')
    plt.savefig(pp, format='pdf')
    pp.close()
    print('end')


def plot_with_date(tb_i, te_i, trgt, dataset='citybike', fname='fig1', ylabel='In-Flow of [16,16]'):
    gridi = 16 if dataset=='BJTaxi' else 8
    tslot = 30 if dataset=='BJTaxi' else 60
    resh = 48 if dataset=='BJTaxi' else 24
    if dataset=='BJTaxi':
        tindex = dateindex('2015-11-01', '2016-04-09', tslot)
    else:
        tindex = dateindex('2015-07-01', '2016-06-30', tslot)
    inflow = trgt[-len(tindex):, gridi, gridi, 0]
    inflow = inflow.reshape([-1,resh]).sum(axis=1)


    if dataset=='BJTaxi':
        tindex = dateindex('2015-11-01', '2016-04-09', resh*tslot)
    else:
        tindex = dateindex('2015-07-01', '2016-06-30', resh*tslot)
    tindex = list(tindex)
    years = dates.YearLocator()
    months = dates.MonthLocator()
    dfmt = dates.DateFormatter('%b')
    ax = plt.figure()
    ax.set_size_inches(5, 2)

    ax = ax.add_subplot(111)
    ax.xaxis.set_major_locator(months)
    # ax.xaxis.set_minor_locator(years)
    ax.xaxis.set_major_formatter(dfmt)
    ax.set_xlim(tindex[tb_i], tindex[te_i])

    lw, ls = 0.5, '-'
    # plt.plot(tindex, read_line,color='r',linewidth=lw,linestyle='--',label='average')
    # plt.plot(tindex, result_line,color='black',linewidth=lw,linestyle=ls,label='rainy on Oct 27')
    plt.plot(tindex, inflow, color='r', linewidth=lw, label='In-Flow')
    # plt.plot(tindex, outflow, color='r', linewidth=lw, linestyle='--', label='Out-FLow')

    plt.xticks(fontsize=6,color='black')
    plt.ylabel(ylabel,fontsize=6,color='black')
    plt.yticks(fontsize=6,color='black')
    plt.grid()
    pp = pdf('figs/'+fname+'.pdf')
    plt.savefig(pp,format='pdf')
    pp.close()


def plt_6steps(dataset):
    data = pd.read_csv('result-collect/'+dataset+'_self.csv')
    data = data.values
    colors = ['r', 'g', 'b', 'y', 'black', 'grey','c', 'm']
    marker = ['.','s','^','+','*','2','x','o']
    linestyle = [':','-.','--','-',':','-.','--','-']
    # models = ['ResNet','ST-UNet','ST-UNet+','ST-UNet-','ConvLSTM','AttConvLSTM','PCRN','ST-Attn']
    models = ['ST-Attn','ST-Attn_kde','ST-Attn_hm']
    plt.figure(figsize=(6, 4))
    for i in range(3):
        plt.plot([1,2,3,4,5,6], data[i,:],
                 color=colors[i],
                 linestyle=linestyle[i], lw=1,
                 marker=marker[i], ms=6,
                 label=models[i])

    fontsize = 10
    plt.xticks(fontsize=fontsize,color='black')
    plt.ylabel('RMSE',fontsize=fontsize,color='black')
    plt.xlabel('predicting step',fontsize=fontsize,color='black')
    plt.yticks(fontsize=fontsize,color='black')
    plt.legend(fontsize=fontsize-2)
    plt.grid()
    # plt.show()
    pp = pdf('figs/'+dataset+'_self.pdf')
    plt.savefig(pp,format='pdf')
    pp.close()


def improvements():
    xi = []
    datasets = ['BJTaxi', 'nyctaxi', 'citybike']
    for dataset in datasets:
        data = pd.read_csv('result-collect/' + dataset + '.csv')
        x = data.values
        x1 = x[-1,:]
        # x2 = x[:-1,:]
        # x2.sort(axis=0)
        # x2 = x2[0,:]
        x2 = x[-3,:]
        xi.append((x2-x1)/x2)

    colors = ['r', 'g', 'b']
    marker = ['.', 's', '^']
    linestyle = [':', '-.','--']
    xi = np.array(xi)
    plt.figure(figsize=(6, 4))
    for i in range(3):
        plt.plot([1,2,3,4,5,6], xi[i,:],
                 color=colors[i],
                 linestyle=linestyle[i], lw=1,
                 marker=marker[i], ms=6,
                 label=datasets[i])

    fontsize = 10
    plt.xticks(fontsize=fontsize,color='black')
    plt.ylabel('RMSE',fontsize=fontsize,color='black')
    plt.xlabel('predicting step',fontsize=fontsize,color='black')
    plt.yticks(fontsize=fontsize,color='black')
    plt.legend(fontsize=fontsize-2)
    plt.grid()
    # plt.show()
    pp = pdf('figs/improvement.pdf')
    plt.savefig(pp,format='pdf')
    pp.close()


def plt_heatmap(dataset):
    results_path = '/cluster/zhouyirong09/peer-work/ST-Attn/result-collect/' + dataset + '/'
    trgt = np.vstack(np.load(results_path + 'ST-Attn/target.npy')) * preprocess_max[dataset]
    if dataset in ['citybike']:
        plt.imshow(np.log(trgt[-112, 3, :, :, 0] + 1), cmap='Reds')
    elif dataset in ['nyctaxi']:
        plt.imshow(np.log(trgt[-112, 3, ::-1, :, 0] + 1), cmap='Reds')
    else:
        plt.imshow(trgt[-112, 3, :, :, 0], cmap='Reds')
    plt.tick_params(which='both', left=False, bottom=False, labelleft=False, labelbottom=False)

    pp = pdf('figs/'+dataset+'_heatmap.pdf')
    plt.savefig(pp,format='pdf')
    pp.close()


def dateindex(begin, end, interval):
    from datetime import timedelta
    out = []
    for dt in pd.date_range(begin,end):
        total = int(24 * 60 / interval)
        tmp = [dt+timedelta(minutes=i*interval) for i in range(total)]
        out.extend(tmp)
    return pd.Series(out)

if __name__ == '__main__':
    # # improvements()
    dataset = 'BJTaxi'
    # for dataset in ['BJTaxi', 'nyctaxi', 'citybike']:  #
    #     # plt_heatmap(dataset)
    #     plt_6steps(dataset)

    trgt = kde_data2(dataset)

    # plot_with_date(0, -1, trgt, dataset)
    plt_3d(trgt)