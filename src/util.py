import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plotPlant(data, exp, plant, target, dataInterp=None, nsteps=None, savePlot=False, figSize=(12,6), title=None, highlightInterpolation=True):
    
    labels = { 'healthy':'green', 'uncertain':'gray', 'stress':'red', 'recovery':'blue'}

    if (exp == None) and (plant == None):
        df = data
    
    df = data[((data.Exp==exp) & (data.Plant==plant))]
    n = df[((df.Exp==exp) & (df.Plant==plant))][target].shape[0]
    
    plt.figure(figsize=figSize)
    
    if nsteps != None:
        plt.subplot(1,2,1)
    
    doPlotPlant(df, target, labels)
    
    if title==None:
        plt.title(f'{exp}/{plant}  {target} raw data - # obs {n}');
    else: 
        plt.title(title)
    
    if nsteps != None:
        plt.subplot(1,2,2)
        doPlotPlant(df, target, labels)
        plt.title(f'{exp}/{plant} rolling mean ( {nsteps} )');
        plt.xlabel('Time')
        plt.ylabel(target)

    if not dataInterp is None:
        
        
        if highlightInterpolation:
            lstyle='*'
        else:
            lstyle='-'
            
        showLabel=True
        xx = np.where(df[target].isnull())
        df2 = dataInterp[((data.Exp==exp) & (data.Plant==plant))]
        for x in xx[0]:
            y = df2[target].iloc[x]
            l = df2['label'].iloc[x]
            if showLabel:
                plt.plot(df2.Day.iloc[x],y, lstyle, color=labels[l])
                showLabel = False
            else:
                plt.plot(df2.Day.iloc[x],y, lstyle, color=labels[l])

    plt.xlabel('Time')
    plt.ylabel(target)
#     plt.legend( loc='lower left', bbox_to_anchor=(1, 0.5))
    plt.legend(loc='lower center')
   
    if savePlot:
        plt.savefig(f"{exp}-{plant}-{target}.png")
    
    return

    

def doPlotPlant(df, target, labels):
    for l in labels:
        df_tmp = df[df.label==l]
        x = np.diff(df_tmp.index.values)
        splits=np.where(x>1)[0] + 1 # accounts for diff lag
        if len(splits)>0:
            prevIdx = 0
            for idx in splits:
                plt.plot(df_tmp['Day'][prevIdx:idx], df_tmp[target][prevIdx:idx], c=labels[l])
                prevIdx = idx

            plt.plot(df_tmp['Day'][idx:], df_tmp[target][idx:], c=labels[l], label=l)
        else:
            plt.plot(df_tmp['Day'], df_tmp[target], c=labels[l], label=l)
    return


def toTS(data, exp, plant, addExpInfo=False):
    df = data[((data.Exp==exp) & (data.Plant==plant))]
    tsdata = {}
    for target in df.columns[4:]:
        values = df[target].values
        ts = pd.Series(values, index = pd.date_range('1/1/2000', periods=len(values), freq='T'))
        tsdata[target]=ts

    df=pd.DataFrame(tsdata, index=pd.date_range('1/1/2000', periods=len(values), freq='T'))
    
    if addExpInfo:
        df['Exp']=exp
        df['Plant']=plant
    
    return df


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def getLags(data, targets, nlag, smoothedTarget=False):
    lagged = {}
        
    if type(targets)== str:
        targets = [ targets ]
    
    idx = []
    lagged = { }
    for t in targets:
        lagged[t] = []
    
    for tag, gdf in data.groupby(['Exp', 'Plant']):
        for t in targets:
            rolling_window = lagged[t]

            values = gdf[t].to_numpy()
            for i in range(nlag, len(values)):
                rolling_window.append( (values[(i-nlag):i]))
        
        for i in range(nlag, len(values)):
            idx.append( (*tag, data.index[i][2]))
    
    n = len(idx)
    
    df = pd.DataFrame(idx, columns=['Exp', 'Plant', 'idx'])

    for t in lagged.keys():
        for lag in range(0, nlag):

            d = np.zeros(n)
            c = f"{t}_{(nlag-lag)-1}"
            for k in range(n):
                d[k] = lagged[t][k][lag]
            df[c] = d
    
    df = df.dropna()
    df.set_index(['Exp', 'Plant', 'idx'], inplace=True)
    
    if smoothedTarget: 
        df['starget']=data.loc[df.index]['starget'].to_numpy().astype(int)
    else:
        df['target']=data.loc[df.index]['target']
    
    return df


def getTensors(data, target, nsteps):
    df = data.copy() # set_index(['Exp', 'Plant'])
    df['prediction'] = data.groupby(['Exp', 'Plant'])[target].shift(-nsteps)

    df = df.dropna();

    Y = df['prediction'].to_numpy(dtype=np.int8)

    del df['prediction']
    
    X = df.loc[:, df.columns != target].to_numpy()


    n = Y.shape[0]


    Y2 = np.empty(n)
    YM = np.zeros((n,4), np.int8)

    # change on stressed/non stressed

    for i in range(len(Y)):
        if Y[i]==2:
            Y2[i]=1
        else:
            Y2[i]= 0

        YM[i,Y[i]]=1
    
    return X, YM, Y2




