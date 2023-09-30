import pandas as pd
import numpy as np
import datetime
import math
from pytz import timezone
import pytz
from dateutil import parser
import scipy.stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.sandbox.regression.predstd import wls_prediction_std

import plotly
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import matplotlib as mpl
mpl.use('tkagg')    #YAAA!!  this finally makes the Damn thing work
import matplotlib.pyplot as plt
#matplotlib inline
plt.rcParams['figure.figsize'] = (5, 5) # set default size of plots

import seaborn as sns

# directory for storing graphs generated
# import os
# graph_dir = os.path.join('drive/My Drive/', root_path)+'graphs_SI/'

from datetime import date
today = date.today()
fdate = date.today().strftime('%m%d%Y')    # append the data today when exporting the graphs



def f_test(x, y):
    x = np.array(x)
    y = np.array(y)
    f = np.var(x, ddof=1)/np.var(y, ddof=1) #calculate F test statistic 
    dfn = x.size-1 #define degrees of freedom numerator 
    dfd = y.size-1 #define degrees of freedom denominator 
    p = 1-scipy.stats.f.cdf(f, dfn, dfd) #find p-value of F test statistic 
    return f, p

# plot the parity chart
def parity_plot(ax, plot_data, Operator, plot_color, force_intercept_origin=0, plot_interval=['confidence'],
                plot_lim = [0,8000], legend_loc='upper left'):

    """
plot parity chart: scatter plot with a parity line, a regression line, and a confidence interval

INPUTS
- ax is the subplot ax to plot on
- plot_data is the processed data
- force_intercept_origin decides which regression to use
- plot_interval can be ['confidence','prediction'] or either one of those in the list
- plot_lim is the limit of the x and y axes
- legend_loc is the location of the legend

OUTPUT
- ax is the plotted parity chart
  """

  # set up plot
  #ax.set_xlabel('Methane release rate [kgh]',fontsize=10)
  #ax.set_ylabel('Reported release rate [kgh]',fontsize=10)
    ax.set_xlim(plot_lim)
    ax.set_ylim(plot_lim)

    # parity line
    x_lim = np.array([0,8000])
    y_lim = np.array([0,8000])
    ax.plot(x_lim,y_lim,color='black',linewidth=1, label = 'Parity line')

    x = plot_data['cr_kgh_CH4_mean60'].values
    y = plot_data['FacilityEmissionRate'].fillna(0).values

    # regression
    if force_intercept_origin == 0:
        n,slope,intercept,r_value, f, p ,x_lim,y_pred,lower_CI,upper_CI,lower_PI,upper_PI,residual = linreg_results(x,y)
    elif force_intercept_origin == 1:
        n,slope,intercept,r_value, f, p ,x_lim,y_pred,lower_CI,upper_CI,lower_PI,upper_PI,residual = linreg_results_no_intercept(x,y)

    # scatter plots

    yerr = np.array(list(zip(
    abs(plot_data['cr_kgh_CH4_mean60'] - plot_data['cr_kgh_CH4_lower60']),
    abs(plot_data['cr_kgh_CH4_mean60'] - plot_data['cr_kgh_CH4_upper60'])))).T * 1.96
    xerr = np.array(list(zip(
    abs(plot_data['FacilityEmissionRate'] - np.float64(plot_data['FacilityEmissionRateLower'])),
    abs(plot_data['FacilityEmissionRate'] - np.float64(plot_data['FacilityEmissionRateUpper']))))).T * 1.96
    #xerr = np.float64(xerr)
    #ax.scatter(x,y,s = 10, color='#8c1515',alpha = 0.2,label='$n$ = %d' %(n))
    ax.errorbar(x, y, xerr,
            yerr,
            fmt='o', markersize=2, color=plot_color,ecolor=plot_color, elinewidth=1, capsize=0,alpha=0.2);

    # plot regression line
    if force_intercept_origin == 0:
        if intercept<0:   # label differently depending on intercept
            ax.plot(x,y_pred,'-',color=plot_color,linewidth=1,
                label = 'Best fit $R^{2}$ = %0.2f \n$y$ = %0.2f$x$%0.2f' % (r_value**2,slope,intercept))
        elif intercept>=0:
            ax.plot(x,y_pred,'-',color=plot_color,linewidth=1,
                label = 'Best fit $R^{2}$ = %0.2f \n$y$ = %0.2f$x+$%0.2f' % (r_value**2,slope,intercept))
    elif force_intercept_origin ==1:
        ax.plot(x,y_pred,'-',color=plot_color,linewidth=1,
            label = 'Best fit $R^{2}$ = %0.2f \n$y$ = %0.2f$x$' % (r_value**2,slope))

    # plot intervals
    if 'confidence' in plot_interval:
        ax.plot(np.sort(x),upper_CI,':',color='black', label='95% CI')
        ax.plot(np.sort(x),lower_CI,':',color='black')
        ax.fill_between(np.sort(x), lower_CI, upper_CI, color='black', alpha=0.05)
    if 'prediction' in plot_interval:
        ax.plot(np.sort(x),upper_PI,':',color='#8c1515', label='95% PI')
        ax.plot(np.sort(x),lower_PI,':',color='#8c1515')
        ax.fill_between(np.sort(x), lower_PI, upper_PI, color='black', alpha=0.05)

    # ax.legend(loc=legend_loc, bbox_to_anchor=(1.6, 0.62),fontsize=12)   # legend box on the right
    # ax.legend(loc=legend_loc,fontsize=8)   # legend box within the plot
    #ax.set_yticks(np.arange(0,plot_lim[1],4000))
    #ax.set_yticklabels(np.arange(0,plot_lim[1]+1000,1000).astype('int'),fontsize=10, fontname = 'Arial')
    #ax.set_xticks(np.arange(0,plot_lim[1],4000))
    #ax.set_xticklabels(np.arange(0,plot_lim[1]+1000,1000).astype('int'),fontsize=10, fontname = 'Arial')

    labels = [item.get_text() for item in ax.get_xticklabels()]
    empty_string_labels = [''] * len(labels)
    ax.set_xticklabels(empty_string_labels)
    labels = [item.get_text() for item in ax.get_yticklabels()]
    empty_string_labels = [''] * len(labels)
    ax.set_yticklabels(empty_string_labels)

    return ax, n, r_value,slope,intercept, f, p

      
def detection_rate_by_bin(data,n_bins,threshold):  

  """
  INPUT
  - data is the processed data after data selection
  - n_bins is the number of bins 
  - threshold is the highest release rate in kgh/mps to show in the detection threshold graph; if wind_normalization is set to 0, then this number is measured in kgh methane
  - wind_to_use is the type of wind to use
  - wind_normalization is whether or not to apply wind normalization in presenting values on the x axis
  OUTPUT
  - detection is a dataframe of wind-normalized release rate of each data point and whether each release was detected by Kairos
  - detection_prob is a dataframe that has n_bins number of rows recording the detection rate of each bin
  """

  '''
  Yulia data:
     - CH4_release_kgh -> total_release_kgh
     - CH4_release_kghmps -> DELETE
     - 

  '''

  # find whether each pass was detected
  detection = pd.DataFrame()
  detection['released'] = data.release_applied_kgh!=0
  detection['detected'] = ~np.isnan(data.performer_release_kgh)
  # detection[wind_to_use] = data[wind_to_use]
  detection['release_rate'] = data.release_applied_kgh
  # detection['release_rate_wind_normalized'] = data.CH4_release_kghmps

  # if wind_normalization ==1:
  #     detection = detection.loc[detection.release_rate_wind_normalized <= threshold]
  # elif wind_normalization ==0:
  detection = detection.loc[detection.release_rate <= threshold]

  # find the median wind of passes below min detection
  # median_wind = detection[wind_to_use].median()

  # initiate the bins 
  bins = np.linspace(0,threshold,n_bins+1)
  detection_probability = np.zeros(n_bins)
  # detection_probability_highwind = np.zeros(n_bins)
  # detection_probability_lowwind = np.zeros(n_bins)
  bin_size, bin_num_detected = np.zeros(n_bins).astype('int'), np.zeros(n_bins).astype('int')
  # bin_size_highwind, bin_num_detected_highwind = np.zeros(n_bins).astype('int'), np.zeros(n_bins).astype('int')
  # bin_size_lowwind, bin_num_detected_lowwind = np.zeros(n_bins).astype('int'), np.zeros(n_bins).astype('int')
  bin_median = np.zeros(n_bins)
  # bin_two_sigma, bin_two_sigma_highwind, bin_two_sigma_lowwind = np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins)
  bin_two_sigma = np.zeros(n_bins)
  two_sigma_upper, two_sigma_lower = np.zeros(n_bins),np.zeros(n_bins)
  # two_sigma_upper_highwind, two_sigma_lower_highwind = np.zeros(n_bins),np.zeros(n_bins)
  # two_sigma_upper_lowwind, two_sigma_lower_lowwind = np.zeros(n_bins),np.zeros(n_bins)

  # for each bin, find number of data points and detection prob
  # for each bin, find number of data points and detection prob
  for i in range(n_bins):
      bin_min = bins[i]
      bin_max = bins[i+1]
      bin_median[i] = (bin_min+bin_max)/2
      
      # if wind_normalization==1:
      #   binned_data = detection.loc[detection.release_rate_wind_normalized<bin_max].loc[detection.release_rate_wind_normalized>=bin_min]
      # elif wind_normalization==0:
      binned_data = detection.loc[detection.release_rate<bin_max].loc[detection.release_rate>=bin_min]

      ################# all data within bin ######################
      bin_num_detected[i] = binned_data.detected.sum()
      n = len(binned_data)
      bin_size[i] = n
      p = binned_data.detected.sum()/binned_data.shape[0]
      detection_probability[i] = p
      
      # std of binomial distribution
      sigma = np.sqrt(p*(1-p)/n)
      bin_two_sigma[i] = 2*sigma

      # find the lower and uppder bound defined by two sigma
      two_sigma_lower[i] = 2*sigma
      two_sigma_upper[i] = 2*sigma
      if 2*sigma + p > 1:
        two_sigma_upper[i] = 1-p
      if p - 2*sigma < 0 :
        two_sigma_lower[i] = p
  
  detection_prob = pd.DataFrame({
    "bin_median": bin_median,
    "detection_prob_mean": detection_probability,
    "detection_prob_two_sigma_upper": two_sigma_upper,
    "detection_prob_two_sigma_lower": two_sigma_lower,
    "n_data_points": bin_size,
    "n_detected": bin_num_detected})
  
  return detection, detection_prob 





# plot the minimum detection limit
def detection_rate_bin_plot(ax, data, 
              n_bins, 
              threshold,
              path_export,
              operator):
              # by_wind_speed = 0,
              # wind_normalization = 1,
              # wind_to_use = "WS_windGust_logged_mps"):
  """ 
  INPUT
  - ax is the subplot to be plotted
  - data is the processed data after data selection
  - n_bins: number of bins
  - threshold: max wind-normalized release rate to include in the plot
  - by_wind_speed is a binary that decides whether to split the data into low/high wind when plotting
  - wind_to_use is the type of wind speed to use to normalize the release rate
  - wind_normalization is binary determining whether to apply wind normalization to numbers on the x-axis
  OUTPUT
  - ax is the subplot to show the minium detection
  """
  detection, detection_prob = detection_rate_by_bin(data,
      n_bins = n_bins,
      threshold = threshold)
  w = threshold/n_bins/2.5    # bin width in the plot

  # if by_wind_speed==0:
  for i in range(n_bins):
    ax.annotate('%d / %d' %(detection_prob.n_detected[i],detection_prob.n_data_points[i]),
                  [detection_prob.bin_median[i]-w/1.6,0.03],fontsize=13)

  # for plotting purpose, we don't want a small hypen indicating zero uncertainty interval
  detection_prob.detection_prob_two_sigma_lower[detection_prob.detection_prob_two_sigma_lower==0]=np.nan  
  detection_prob.detection_prob_two_sigma_upper[detection_prob.detection_prob_two_sigma_upper==0]=np.nan
  detection_prob.detection_prob_mean[detection_prob.detection_prob_mean==0]=np.nan

  # plot the bars and the detection points
  ax.bar(detection_prob.bin_median,detection_prob.detection_prob_mean,
        yerr=[detection_prob.detection_prob_two_sigma_lower,detection_prob.detection_prob_two_sigma_upper],
        error_kw=dict(lw=2, capsize=3, capthick=1,alpha=0.3),
        width=threshold/n_bins-0.5,alpha=0.6,color='#8c1515',ecolor='black', capsize=2)
    # if wind_normalization == 1:
    #   ax.scatter(detection.release_rate_wind_normalized,np.multiply(detection.detected,1),
    #           edgecolor="black",facecolors='none')
    # elif wind_normalization == 0:
  ax.scatter(detection.release_rate,np.multiply(detection.detected,1),
              edgecolor="black",facecolors='none')
  

  # elif by_wind_speed == 1:   # split data into two sets based on wind speed
  
  #   # find median wind
  #   median_wind = detection[wind_to_use].median()
  #   detection_lowwind = detection[detection[wind_to_use]<median_wind]
  #   detection_highwind = detection[detection[wind_to_use]>=median_wind]
  #   n_below_median_wind = detection_lowwind.shape[0]
  #   n_above_median_wind = detection_lowwind.shape[0]
  
  #   # width and position of the bars
  #   x = detection_prob.bin_median
  #   w = threshold/n_bins/2.5

  #   for i in range(n_bins):
  #     offset_annotation = -0.3
  #     dist_annotation = 1.7
  #     # low wind
  #     ax.annotate("%d" %detection_prob_lowwind.n_detected[i],[x[i]-w/dist_annotation+offset_annotation,0.11],fontsize=15)
  #     ax.annotate("—",[detection_prob_lowwind.bin_median[i]-w/dist_annotation-0.2+offset_annotation,0.081],fontsize=15)
  #     ax.annotate("%d"%detection_prob_lowwind.n_data_points[i],[x[i]-w/dist_annotation+offset_annotation,0.04],fontsize=15) 
  #     # high wind
  #     ax.annotate("%d" %detection_prob_highwind.n_detected[i],[x[i]+w/dist_annotation+offset_annotation,0.11],fontsize=15)
  #     ax.annotate("—",[detection_prob_highwind.bin_median[i]+w/dist_annotation-0.2+offset_annotation,0.081],fontsize=15)
  #     ax.annotate("%d"%detection_prob_highwind.n_data_points[i],[x[i]+w/dist_annotation+offset_annotation,0.04],fontsize=15) 

  #   # for plotting purpose, we don't want a small hypen indicating zero uncertainty interval
  #   detection_prob_lowwind.detection_prob_two_sigma_lower[detection_prob_lowwind.detection_prob_two_sigma_lower==0]=np.nan  
  #   detection_prob_lowwind.detection_prob_two_sigma_upper[detection_prob_lowwind.detection_prob_two_sigma_upper==0]=np.nan
  #   detection_prob_lowwind.detection_prob_mean[detection_prob_lowwind.detection_prob_mean==0]=np.nan
  #   detection_prob_highwind.detection_prob_two_sigma_lower[detection_prob_highwind.detection_prob_two_sigma_lower==0]=np.nan  
  #   detection_prob_highwind.detection_prob_two_sigma_upper[detection_prob_highwind.detection_prob_two_sigma_upper==0]=np.nan
  #   detection_prob_highwind.detection_prob_mean[detection_prob_highwind.detection_prob_mean==0]=np.nan

  #   # low wind: plot the bars and the detection points
  #   ax.bar(x - w/2, detection_prob_lowwind.detection_prob_mean,
  #         yerr=[detection_prob_lowwind.detection_prob_two_sigma_lower,
  #               detection_prob_lowwind.detection_prob_two_sigma_upper],
  #         error_kw=dict(lw=2, capsize=3, capthick=1,alpha=0.3),
  #         width=w,alpha=0.6,color='#00505c',ecolor='black', capsize=2,align='center',
  #         label='below %.2f mps n=%d'%(median_wind,n_below_median_wind))
  #   ax.scatter(detection_lowwind.release_rate_wind_normalized,np.multiply(detection_lowwind.detected,1),
  #             edgecolor="black",facecolors='#00505c',alpha=0.3)

  #   # high wind: plot the bars and the detection points
  #   ax.bar(x + w/2, detection_prob_highwind.detection_prob_mean,
  #         yerr=[detection_prob_highwind.detection_prob_two_sigma_lower,
  #               detection_prob_highwind.detection_prob_two_sigma_upper],
  #         error_kw=dict(lw=2, capsize=3, capthick=1,alpha=0.3),
  #         width=w,alpha=0.6,color='#5f574f',ecolor='black', capsize=2,align='center',
  #         label='above %.2f mps n=%d'%(median_wind,n_above_median_wind))
  #   ax.scatter(detection_highwind.release_rate_wind_normalized,np.multiply(detection_highwind.detected,1),
  #             edgecolor="black",facecolors='#5f574f',alpha=0.3)
    
  ax.legend(loc='upper right',fontsize=11)

  # set more room on top for annotation
  ax.set_ylim([-0.05,1.22])
  ax.set_xlim([0,threshold])
  ax.set_yticks([0.0,0.2,0.4,0.6,0.8,1.0])
  ax.set_yticklabels([0.0,0.2,0.4,0.6,0.8,1.0],fontsize=13)
  ax.set_xticklabels(np.arange(0,threshold+threshold/n_bins,threshold/n_bins).astype('int'),
                    fontsize=13)
  ax.set_xlabel('Methane release rate [kgh]',fontsize=22)
  ax.set_ylabel('Proportion detected',fontsize=22)
  
  # fig.savefig(path_export  + operator + '_mindetect.jpg')
  
  plt.close()

  return ax

def linreg_results(x,y):

    """
Regression to output all values to be potentially used in plotting:
n = number of points in scatter plot;
pearson_corr = peason's correlation coefficient;
slope, intercept = regression parameters;
r_value = R (not R_squared);
x_lim = (min&max) of x;
y_pred = (min&max) of y_predction computed with slope, intercept, and x_lim;
lower_CI, upper_CI are bounds of 95% confidence interval for the fit line;
lower_PI, upper_CI are bounds of 95% prediction interval for predictions;
see reference: http://www2.stat.duke.edu/~tjl13/s101/slides/unit6lec3H.pdf
    """

    n = len(x)
    #pearson_corr, _ = scipy.stats.pearsonr(x, y)    # Pearson's correlation coefficient
    #slope, intercept, r_value, p_value, std_err = scipy.stats.stats.linregress(x,y)

    x_lim = np.array([0,max(x)])

    x_Reg = sm.add_constant(x)
    model = sm.OLS(y,x_Reg)
    result = model.fit()
    slope = result.params[1]
    intercept = result.params[0]
    #slope, intercept, r_value, p_value, std_err = scipy.stats.stats.linregress(x,y)
    r_squared = result.rsquared
    # std_err = result.bse[0]
    r_value = math.sqrt(r_squared)
    f, p = result.fvalue, result.f_pvalue

    
    y_pred = intercept + slope*x
    residual = y - (intercept+slope*x)
    dof = n - 2                               # degree of freedom
    t_score = scipy.stats.t.ppf(1-0.025, df=dof)    # one-sided t-test

    # sort x from smallest to largest for ease of plotting
    df = pd.DataFrame()
    df['x'] = x
    df['y'] = y
    df = df.sort_values('x')
    x = df.x.values
    y = df.y.values

    y_hat = intercept + slope*x
    x_mean = np.mean(x)
    S_yy = np.sum(np.power(y-y_hat,2))      # total sum of error in y
    S_xx = np.sum(np.power(x-x_mean,2))     # total sum of variation in x



    # find lower and upper bounds of CI and PI
    lower_CI = y_hat - t_score * np.sqrt(S_yy/dof * (1/n+np.power(x-x_mean,2)/S_xx))
    upper_CI = y_hat + t_score * np.sqrt(S_yy/dof * (1/n+np.power(x-x_mean,2)/S_xx))
    lower_PI = y_hat - t_score * np.sqrt(S_yy/dof * (1/n+1+np.power(x-x_mean,2)/S_xx))
    upper_PI = y_hat + t_score * np.sqrt(S_yy/dof * (1/n+1+np.power(x-x_mean,2)/S_xx))

    return n,slope,intercept,r_value, f, p , x_lim,y_pred,lower_CI,upper_CI,lower_PI,upper_PI,residual



def linreg_results_no_intercept(x,y):

    """
Regression to output all values to be potentially used in plotting:
n = number of points in scatter plot;
pearson_corr = peason's correlation coefficient;
slope = regression parameters;
r_value = R (not R_squared);
x_lim = (min&max) of x;
y_pred = (min&max) of y_predction computed with slope, intercept, and x_lim;
lower_CI, upper_CI are bounds of 95% confidence interval for the fit line;
lower_PI, upper_CI are bounds of 95% prediction interval for predictions;
see reference: http://www2.stat.duke.edu/~tjl13/s101/slides/unit6lec3H.pdf
    """

# An intercept is not included by default and should be added by the user
# https://stackoverflow.com/questions/20701484/why-do-i-get-only-one-parameter-from-a-statsmodels-ols-fit


    n = len(x)

    model = sm.OLS(y,x)
    result = model.fit()
    slope = result.params[0]
    intercept = 0
    #slope, intercept, r_value, p_value, std_err = scipy.stats.stats.linregress(x,y)
    r_squared = result.rsquared
    # std_err = result.bse[0]
    r_value = math.sqrt(r_squared)

    x_lim = np.array([0,max(x)])
    y_pred = slope*x
    residual = y - y_pred
    dof = n - 1                               # degree of freedom
    t_score = scipy.stats.t.ppf(1-0.025, df=dof)    # one-sided t-test

    # sort x from smallest to largest for ease of plotting
    df = pd.DataFrame()
    df['x'] = x
    df['y'] = y
    df = df.sort_values('x')
    x = df.x.values
    y = df.y.values

    y_hat = slope*x
    x_mean = np.mean(x)
    S_yy = np.sum(np.power(y-y_hat,2))      # total sum of error in y
    S_xx = np.sum(np.power(x-x_mean,2))     # total sum of variation in x

    # find lower and upper bounds of CI and PI
    lower_CI = y_hat - t_score * np.sqrt(S_yy/dof * (1/n+np.power(x-x_mean,2)/S_xx))
    upper_CI = y_hat + t_score * np.sqrt(S_yy/dof * (1/n+np.power(x-x_mean,2)/S_xx))
    lower_PI = y_hat - t_score * np.sqrt(S_yy/dof * (1/n+1+np.power(x-x_mean,2)/S_xx))
    upper_PI = y_hat + t_score * np.sqrt(S_yy/dof * (1/n+1+np.power(x-x_mean,2)/S_xx))

    f, p = result.fvalue, result.f_pvalue

    return n,slope,intercept,r_value, f, p ,x_lim,y_pred,lower_CI,upper_CI,lower_PI,upper_PI,residual



