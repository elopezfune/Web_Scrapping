# For data visualization
import numpy as np
import pandas as pd
import joblib
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
matplotlib.rc('font', size=16) # To use big fonts...
plt.rcParams['figure.figsize'] = (20.0, 8.0) # To set figure sizes for big plots
from decimal import Decimal
import warnings
warnings.filterwarnings("ignore")


def forecasting(df,target,path_to_model,forecast):
    # Copies the dataframe
    df = df.copy()
    
    #Does a minmax scaling
    minmaxsc = 0
    if any(df[target])>1:
        minmaxsc = 1
        df[target] = (df[target]-min(df[target]))/max(df[target])

    
    # Loads the model
    model = joblib.load(path_to_model)
    
    # Creates dates for forecasting
    strt, offs = pd.to_datetime(df.index).max(), pd.DateOffset(days=1)
    dates = pd.date_range(strt + offs, freq=offs, periods=forecast).strftime('%Y-%m-%d').tolist()
    
    # Makes predictions
    prediction = model.forecast(forecast)
    predd = pd.DataFrame(prediction)
    predd['days'] = dates
    predd = predd.set_index('days')
    predd = pd.concat([model.fittedvalues,predd],axis=0)
    predd['CTR_Pred'] = predd
    predd = predd.drop(0,axis=1)
    df_plot = pd.concat([df,predd],axis=1)
    del predd
    
    #Undoes the minmax scaling
    if minmaxsc==1:
        df_plot[target] = min(df_plot[target])+df_plot[target]*max(df_plot[target])
    
    #Plots the data
    #plt.style.use('Solarize_Light2')
    plt.scatter(df_plot.index,df_plot[target],marker='x', color='black',label=target)
    plt.scatter(df_plot.index, df_plot['CTR_Pred'], color="g",label="Exponential Smoothing predictions up to "+str(forecast)+" days")
    plt.plot(df_plot.index[:-forecast], df_plot['CTR_Pred'][:-forecast], color="green",label="Exponential Smoothing")
    
    #Confidence intervals fit
    #[90: 1.645, 95: 1.96, 99: 2.575]
    z = 1.645
    sse = df_plot['CTR_Pred'].std()
    predint_xminus = df_plot['CTR_Pred'] - z * sse * np.sqrt(1.0/len(df_plot))
    predint_xplus  = df_plot['CTR_Pred'] + z * sse * np.sqrt(1.0/len(df_plot))
    plt.plot(df_plot.index,predint_xminus.values, color='green', alpha=0.2)
    plt.plot(df_plot.index,predint_xplus.values, color='green', alpha=0.2)
    plt.fill_between(df_plot.index,predint_xminus.values, predint_xplus.values,alpha=0.5, label='90% CI')
    
    z = 1.96
    predint_xminus = df_plot['CTR_Pred'] - z * sse * np.sqrt(1.0/len(df_plot))
    predint_xplus  = df_plot['CTR_Pred'] + z * sse * np.sqrt(1.0/len(df_plot))
    plt.plot(df_plot.index,predint_xminus.values, color='green', alpha=0.2)
    plt.plot(df_plot.index,predint_xplus.values, color='green', alpha=0.2)
    plt.fill_between(df_plot.index,predint_xminus.values, predint_xplus.values,alpha=0.3, label='95% CI')
    
    z = 2.575
    predint_xminus = df_plot['CTR_Pred'] - z * sse * np.sqrt(1.0/len(df_plot))
    predint_xplus  = df_plot['CTR_Pred'] + z * sse * np.sqrt(1.0/len(df_plot))
    plt.plot(df_plot.index,predint_xminus.values, color='green', alpha=0.2)
    plt.plot(df_plot.index,predint_xplus.values, color='green', alpha=0.2)
    plt.fill_between(df_plot.index,predint_xminus.values, predint_xplus.values,alpha=0.1, label='99% CI')

    #plt.xlim(min(df_plot.index),max(df_plot.index))
    plt.autoscale(axis=df_plot.index.name)
    plt.legend(loc='best',fontsize=12)
    plt.xlabel("Days", fontsize=20)
    plt.xticks(rotation=45)
    plt.ylabel(target, fontsize=20)
    plt.show()





def Data_Analytics(df,colname,targetname):
    ### This function checks the target value difference of a given cathegory in the case
    ### of binary classifications.
    
    ## Arguments:
    # df: is a data frame.
    # colname: is a string. The column name to be evaluated.
    # targetname: is a string. The column name of the target variable.
    
    order = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    
    # caculate aggregate stats
    df_cate = df.groupby([colname])[targetname].agg(['count', 'sum', 'mean'])
    df_cate.reset_index(inplace=True)
    #print(df_cate)
    
    # plot visuals
    f, ax = plt.subplots(figsize=(20, 8))
    plt1 = sns.lineplot(x=colname, y="mean", data=df_cate,color="b")
    if colname == 'Centre':
        plt.xticks(size=18,rotation=90)
    else:
        plt.xticks(size=18,rotation=0)
    plt.yticks(size=20,rotation=0)
    
    for tl in ax.get_yticklabels():
        tl.set_color('b')

    ax2 = ax.twinx()
    plt2 = sns.barplot(x=colname, y="count", data=df_cate,
                       ax=ax2,alpha=0.5, order=order)