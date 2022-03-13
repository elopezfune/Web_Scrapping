import numpy as np
import pandas as pd
import datetime 
from scipy.stats import ttest_ind
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from decimal import Decimal
import warnings
warnings.filterwarnings("ignore")
from scipy.optimize import basinhopping



def load_data(path,index):
    df = pd.read_csv(path)
    df.set_index(index,inplace=True)
    df.columns = df.columns.map(lambda x: x.title())
    # Returns the dataframe
    return df


# Converts epoch time to datetime and sort by date
# I leave the format YY/mm/DD/HH:MM:SS since a priory we don't know the time scale of events
def to_datetime(df,var):
    # Copies the dataframe
    df = df.copy()
    df[var] = pd.to_datetime(df[var], utc=True, format = "%Y%m%d%H%M%S").dt.strftime("%Y-%m-%d-%H:%M:%S")
    df.sort_values(by=[var],inplace=True)
    # Returns the dataframe
    return df


# Checks for duplicated data
def duplicated_data(df):
    # Copies the dataframe
    df = df.copy()
    # Rows containing duplicate data
    print("Removed ", df[df.duplicated()].shape[0], ' duplicated rows.')
    # Returns a dataframe with the duplicated rows removed
    return df.drop_duplicates()


# Checks for columns with missing values (NaNs)
def check_missing_values(df,cols=None,axis=0):
    # Copies the dataframe
    df = df.copy()
    if cols != None:
        df = df[cols]
    missing_num = df.isnull().sum(axis).to_frame().rename(columns={0:'missing_num'})
    missing_num['missing_percent'] = df.isnull().mean(axis)*100
    result = missing_num.sort_values(by='missing_percent',ascending = False)
    # Returns a dataframe with columns with missing data as index and the number and percent of NaNs
    return result[result["missing_percent"]>0.0]


# Encodes the categorical variable Action
def cat_encoder(df,variables):
    # Copies the dataframe
    df = df.copy()
    df_to_encode = df[variables]
    df_encoded = pd.get_dummies(df_to_encode,drop_first=False)
    #Formats the names of the variables
    #print(df_encoded.columns)
    #df_encoded.columns = [el.replace('_','[')+']' for el in df_encoded.columns]
    df = pd.concat([df,df_encoded],axis=1).drop(variables,axis=1)
    del df_to_encode, df_encoded
    # Returns the dataframe with the one-hot-encoded categorical variables
    return df

# Computes a time-series of the daily clickthrough rates
def clickthrough_rate(df,timeline,session,visit_page):
    # Copies the dataframe
    df = df.copy()
    # Gets unique days
    df[timeline] = df[timeline].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d-%H:%M:%S").date())
    date_day = df[timeline].unique()
    
    # Computes the daily CTR
    result = []
    for el in date_day:
        df_day = df[df[timeline]==el]
        # Counts the number of sessions that yielded a result after visiting a page
        counts = df_day.groupby(session)[visit_page].apply(lambda x: 1 in set(x))
        result.append(counts.mean())
    
    # Builds a dataframe with the daily CTR
    daily_ctr = pd.DataFrame(columns=['Days','Click_Through_Rate'])
    daily_ctr['Days'] = date_day
    daily_ctr['Click_Through_Rate'] = result
    # Returns a dataframe with the dates and the CTR
    return daily_ctr.set_index('Days')



def tried_first_result(df,timeline,session,visit_page,result_position,products):
    # Copies the dataframe
    df = df.copy()
    # Gets date
    df[timeline] = df[timeline].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d-%H:%M:%S").date())
    
    # Filters the dataframe by variables
    mask = [timeline,session,result_position]
    result = df[df[visit_page]==1][mask]
    # Grouping by date and result position
    result = result.groupby([timeline,result_position]).count()
    result = result.sort_values([timeline,session],ascending=True)
    # Creates a pivot dataframe with the first desired results
    result.reset_index(inplace=True)
    result = result[result[result_position]<=products].pivot(index=timeline,
                                                             columns=result_position,
                                                             values=session)
    result.columns = [result_position+'[0'+str(el)+']' for el in range(1,products+1)]
    
    # Returns a dataframe with the daily tried firsts results
    return result



def zero_result_rate(df,timeline,search,n_results):
    # Copies the dataframe
    df = df.copy()
    # Gets date
    df[timeline] = df[timeline].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d-%H:%M:%S").date())
    # Computes the number of searches
    mask = df[search]==1
    searches = df[mask].groupby(timeline)[[n_results]].count()
    # Computes the number of searches that yielded zero result
    mask = ((df[search]==1)&(df[n_results]==0))
    result = df[mask].groupby(timeline)[[n_results]].count()
    # Computes the zero result rate
    result['Zero_Result_Rate'] = result[n_results]/searches[n_results]
    result.drop(n_results,axis=1,inplace=True)
    # Returns a dataframe with the daily zero result rate
    return result


def t_student_test(x,y):
    stat, p = ttest_ind(x, y)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        print('Probably the same distribution.')
    else:
        print('Probably different distributions.')



def session_length(df,timeline,session):
    # Copies the dataframe
    df = df.copy()
    # Converts to datetime
    df[timeline] = pd.to_datetime(df[timeline])
    
    #Computes the session length by session id
    result = df.groupby(session)[timeline].apply(lambda x: (max(x) - min(x)).total_seconds())
    result = result.reset_index().rename(columns={timeline: 'Session_Length'})
    
    # Returns a dataframe with the session length
    return result


def session_length_agg(df,timeline,session,variable):
    # Copies the dataframe
    df = df.copy()
    session_length_result = session_length(df,timeline,session)
    # Converts to datetime
    df[timeline] = pd.to_datetime(df[timeline])
    #Computes the session length by session id
    result = df.groupby(session)[variable].sum()
    result = result.reset_index()
    # Returns a dataframe with the session length
    return pd.merge(result,session_length_result)
        
    
    
def outlier_removal(df):
    #Copies the dataframe
    df = df.copy()
    #Outliers removal
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
    return df    
    



























# Exponential Smoothing model
def Time_Series(df):
    # Copies the dataframe
    df = df.copy()
    
    #Does a minmax scaling
    if any(df)>1:
        df = (df-min(df))/max(df)
        
    
    #Exponential Smoothing
    def Exponential_Smoothing(train,params):
        seasonal_periods, initial_level, initial_trend, initial_seasonal, smoothing_level, smoothing_trend, smoothing_seasonal = params
        model = ExponentialSmoothing(train, trend='add',
                                     seasonal='add',
                                     seasonal_periods=int(seasonal_periods),
                                     initialization_method='known',
                                     initial_level=initial_level,
                                     initial_trend=initial_trend,
                                     initial_seasonal=initial_seasonal,
                                     missing='none')
                
        
        model = model.fit(smoothing_level=smoothing_level,
                          smoothing_trend=smoothing_trend,
                          smoothing_seasonal=smoothing_seasonal,
                          optimized=False,
                          remove_bias=False)
        return model
    
    # Mean Squared Error to minimize
    def metric(params):
        model = Exponential_Smoothing(df,params)
        y_valid = [el[0] for el in df.values]
        mse = ((model.fittedvalues.values - y_valid) ** 2).mean()
        return np.sqrt(mse)
    
    
    # This is the optimizer of the model
    boundary = [(2,7),(0,1),(0,1),(0,1),(0,10),(0,10),(0,10)]
    # Initial guess point
    x0 = [np.mean(el) for el in boundary]
    
    # Minimizer using the Basin-Hoping algorithm
    # Uses the method L-BFGS-B because the problem is smooth and bounded
    minimizer_kwargs = dict(method="L-BFGS-B", bounds=boundary)
    opt = basinhopping(metric, x0, minimizer_kwargs=minimizer_kwargs)
    
    #Defines the optimized model
    model = Exponential_Smoothing(df,opt.x)
    
    # Returns the optimized model
    return model




