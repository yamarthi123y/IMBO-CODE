# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 11:59:08 2022

@author: Desktop
"""

import matplotlib.pyplot as plt
import numpy as np
fig,ax = plt.subplots(figsize =(7, 6)) 
x1=[0.00204499,
0.044893313,
0.144943473,
0.175354658,
0.181341719,
0.222087176,
0.236292781,
0.237906908,
0.245829636,
0.259990225,
0.288452882,
0.95

] 
y1=[0.003144654,
0.044025157,
0.119496855,
0.248427673,
0.320754717,
0.396226415,
0.449685535,
0.660377358,
0.786163522,
0.86163522,
0.943396226,
0.962264151

]
x2 = [0,
0.030514077,
0.036475415,
0.054719553,
0.083066456,
0.105001865,
0.133560983,
0.149856593,
0.168184332,
0.186479917,
0.221231881,
0.225186814,
0.247565948,
1


]
      

y2 =[
   0.003144654,
0.075471698,
0.160377358,
0.238993711,
0.377358491,
0.650943396,
0.685534591,
0.716981132,
0.754716981,
0.808176101,
0.814465409,
0.965408805,
0.937106918,
0.965408805


]
x3=[
  0.00408998,
0.27624082,
0.955100256


]
# first plot with X and Y data
y3=[ 
   0,
0.91509434,
0.952830189]
x4=[0.002032128,
0.044893313,
0.057047498,
0.069246698,
0.06901519,
0.07507942,
0.083150056,
0.084944245,
0.090950599,
0.125612532,
0.164422323,
0.168460856,
0.195000707,
0.198917056,
0.225392599,
0.227263958,
0.243559568,
0.276260112,
0.99803218



]
y4=[
0.003144654,
0.044025157,
0.100628931,
0.135220126,
0.248427673,
0.283018868,
0.336477987,
0.459119497,
0.522012579,
0.572327044,
0.594339623,
0.619496855,
0.641509434,
0.726415094,
0.779874214,
0.864779874,
0.896226415,
0.905660377,
0.959119497


]


x6=[
0.006128539,
0.046893288,
0.065111703,
0.085349385,
0.13836478,
0.173013852,
0.178820851,
0.196942805,
0.229424702,
0.280369384,
0.96


]
y6=[     
0,
0.066037736,
0.157232704,
0.261006289,
0.336477987,
0.393081761,
0.553459119,
0.691823899,
0.808176101,
0.896226415,
0.952830189


]
x7=[0.0061414,
0.05297038,
0.079510231,
0.111046803,
0.243611015,
0.292677908,
0.306954251,
0.472591992,
0.492932567,
0.97


]
y7=[0,
0.094339623,
0.116352201,
0.694968553,
0.871069182,
0.877358491,
0.896226415,
0.899371069,
0.952830189,
0.959119497


]

# second plt with x1 and y1 data
plt.plot(x2,y2,c='#006600',Lw = 2, linestyle='solid',label='Proposed')
plt.plot(x1,y1,c='#B34700',Lw = 2, linestyle='solid',label='DNN')

plt.plot(x3,y3,c='#5C85D6',Lw = 2, linestyle='solid',label='CNN')
plt.plot(x4,y4,c='#ff00ff',Lw = 2, linestyle='solid',label='XGBoost')
#plt.plot(x5,y5,c='#1A0033',Lw = 2, linestyle='dashed',label='Deep STRCF')
plt.plot(x6,y6,c='#806000',Lw = 2, linestyle='solid',label='RF')
plt.plot(x7,y7,c='#99003D',Lw = 2, linestyle='solid',label='I-SiamIDS')

plt.minorticks_on()

# Customize the major grid
plt.grid(which='major', linestyle='-', linewidth='0.1', color='grey')
# Customize the minor grid
plt.grid(which='minor', linestyle=':', linewidth='0.1', color='grey')

# Turn off the display of all ticks.
plt.tick_params(which='both', # Options for both major and minor ticks
                top='off', # turn off top ticks
                left='off', # turn off left ticks
                right='off',  # turn off right ticksS
                bottom='off') # turn off bottom ticks

# giving a title to my graph
#plt.title('Some cool customization

ax.spines['right'].set_linewidth(2)
ax.spines['left'].set_linewidth(2) 
ax.spines['top'].set_linewidth(2)  # setting up Y-axis tick color to red
ax.spines['bottom'].set_linewidth(2)
#plt.plot(x1, y1,c='#009933',label='Ice',Lw = 4)

plt.xlim(0,1)
plt.ylim(0,1)
plt.yticks(np.linspace(0,1,11))
plt.xticks(np.linspace(0,1,11))


plt.xlabel("False Positive Rate",fontweight ='bold', fontsize = 15,fontname="Times New Roman")
plt.ylabel("True Positive Rate",fontweight ='bold', fontsize = 15,fontname="Times New Roman")
#plt.title('Success plots of OPE - Illumination Variation', fontweight ='bold', fontsize = 17,fontname="Times New Roman")

plt.legend(loc="lower right")
plt.show()
#--------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
fig,ax = plt.subplots(figsize =(7, 6)) 
x1=[0.00425454,
0.010551822,
0.023146387,
0.033444628,
0.069720919,
0.129678937,
0.139850387,
0.169730781,
0.207993463,
1

] 
y1=[0.006564952,
0.536149501,
0.595318597,
0.690679458,
0.720165392,
0.719968161,
0.844934703,
0.867862728,
0.933526337,
0.957391206
]
x2 = [0.002113182,
0.055139963,
0.98

]
      

y2 =[
   0.006571996,
0.624818619,
0.95723624

]
x3=[
  0.006395898,
0.006297282,
0.04465858,
0.070185819,
0.108645732,
0.116802615,
0.137821732,
0.165419889,
0.214417537,
0.97


]
# first plot with X and Y data
y3=[ 
  0.006557908,
0.529584548,
0.572221518,
0.611611231,
0.63122156,
0.726589465,
0.818625586,
0.874455856,
0.933505205,
0.950664244,
]
x4=[0.006395898,
0.006649479,
0.027992618,
0.051350323,
0.076581717,
0.082625417,
0.96

]
y4=[
0.006557908,
0.447346548,
0.463723708,
0.509699506,
0.618169139,
0.706965048,
0.957250328

]


x6=[
0.00212727,
0.013073553,
0.051336235,
0.082822648,
0.101686319,
0.2450305,
0.360579294,
0.375470183,
0.403265571,
0.443697787,
0.501472183,
0.694166209,
0.732668385,
0.97


]
y6=[     
0.003282476,
0.447325416,
0.512989026,
0.660911768,
0.756244453,
0.785378189,
0.804734937,
0.827712269,
0.837489258,
0.896566784,
0.906245157,
0.912190243,
0.921932012,
0.95723624


]
x7=[0.00425454,
0.006438161,
0.020976854,
0.074355832,
0.11237902,
0.156995337,
0.186819379,
0.201794796,
0.214544328,
0.242396067,
0.25945649,
0.95

]
y7=[0.006564952,
0.496689348,
0.601904681,
0.637913303,
0.759498753,
0.841588831,
0.877674936,
0.880915149,
0.903899525,
0.900518434,
0.916909683,
0.95394672

]

# second plt with x1 and y1 data

plt.plot(x1,y1,c='#006600',Lw = 2, linestyle='solid',label='Proposed')
plt.plot(x2,y2,c='#B34700',Lw = 2, linestyle='solid',label='DNN')
plt.plot(x3,y3,c='#5C85D6',Lw = 2, linestyle='solid',label='CNN')
plt.plot(x4,y4,c='#ff00ff',Lw = 2, linestyle='solid',label='XGBoost')
#plt.plot(x5,y5,c='#1A0033',Lw = 2, linestyle='dashed',label='Deep STRCF')
plt.plot(x6,y6,c='#806000',Lw = 2, linestyle='solid',label='RF')
plt.plot(x7,y7,c='#99003D',Lw = 2, linestyle='solid',label='I-SiamIDS')

plt.minorticks_on()

# Customize the major grid
plt.grid(which='major', linestyle='-', linewidth='0.1', color='grey')
# Customize the minor grid
plt.grid(which='minor', linestyle=':', linewidth='0.1', color='grey')

# Turn off the display of all ticks.
plt.tick_params(which='both', # Options for both major and minor ticks
                top='off', # turn off top ticks
                left='off', # turn off left ticks
                right='off',  # turn off right ticksS
                bottom='off') # turn off bottom ticks

# giving a title to my graph
#plt.title('Some cool customization

ax.spines['right'].set_linewidth(2)
ax.spines['left'].set_linewidth(2) 
ax.spines['top'].set_linewidth(2)  # setting up Y-axis tick color to red
ax.spines['bottom'].set_linewidth(2)
#plt.plot(x1, y1,c='#009933',label='Ice',Lw = 4)

plt.xlim(0,1)
plt.ylim(0,1)
plt.yticks(np.linspace(0,1,11))
plt.xticks(np.linspace(0,1,11))


plt.xlabel("False Positive Rate",fontweight ='bold', fontsize = 15,fontname="Times New Roman")
plt.ylabel("True Positive Rate",fontweight ='bold', fontsize = 15,fontname="Times New Roman")
#plt.title('Success plots of OPE - Illumination Variation', fontweight ='bold', fontsize = 17,fontname="Times New Roman")

plt.legend(loc="lower right")
plt.show()
#----------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
fig,ax = plt.subplots(figsize =(7, 6)) 

x1=[0.004171346,
0.648791379,
0.96


] 
y1=[0.006410771,
0.931353281,
0.961408364

]
x2 = [0,
0.004057704,
1

]
      

y2 =[
  0,
0.951762126,
0.961421734


]
x3=[
  0,
0.007380074,
0.011444462,
0.0217525,
0.032080593,
0.05286379,
0.048598856,
0.067329804,
0.065143858,
0.081775763,
0.090058292,
0.100466602,
0.11499278,
0.208540564,
0.208400182,
0.222953099,
0.22489839,
0.216535644,
0.226877106,
0.235152949,
0.239250762,
0.25377694,
0.353548318,
0.413812236,
0.430464196,
0.436661051,
0.97


]
# first plot with X and Y data
y3=[ 
  0,
0.450191187,
0.495220333,
0.537054121,
0.569241671,
0.572523932,
0.623957163,
0.614371089,
0.665811006,
0.665864485,
0.681968287,
0.675570886,
0.68847933,
0.691995561,
0.759519226,
0.75956602,
0.823880956,
0.846362105,
0.87211883,
0.891438045,
0.920390128,
0.933298572,
0.943265683,
0.956321194,
0.946728435,
0.966040965,
0.948546714

]
x4=[0,
0.0015442,
0.164360394,
0.363943259,
0.98


]
y4=[
0.003222097,
0.742766993,
0.942657361,
0.943299107,
0.951782181


]


x6=[
0.002072303,
0.001697952,
0.625962618,
0.636324135,
0.709095406,
0.715298946,
0.746510509,
0.761030002,
0.79638617,
0.97


]
y6=[     
0.003215413,
0.816721482,
0.911987272,
0.928097759,
0.925116316,
0.941213434,
0.928452056,
0.944575913,
0.93825873,
0.951775496

]
x7=[0.004157976,
0.001704637,
0.058532542,
0.154145944,
0.463834964,
0.657140756,
0.709082036,
0.719503717,
0.98


]
y7=[0,
0.819936895,
0.845854056,
0.855807797,
0.895388791,
0.915302957,
0.931547142,
0.918718915,
0.945337986

]

# second plt with x1 and y1 data
plt.plot(x2,y2,c='#006600',Lw = 2, linestyle='solid',label='Proposed')
plt.plot(x1,y1,c='#B34700',Lw = 2, linestyle='solid',label='DNN')

plt.plot(x3,y3,c='#5C85D6',Lw = 2, linestyle='solid',label='CNN')
plt.plot(x4,y4,c='#ff00ff',Lw = 2, linestyle='solid',label='XGBoost')
#plt.plot(x5,y5,c='#1A0033',Lw = 2, linestyle='dashed',label='Deep STRCF')
plt.plot(x6,y6,c='#806000',Lw = 2, linestyle='solid',label='RF')
plt.plot(x7,y7,c='#99003D',Lw = 2, linestyle='solid',label='I-SiamIDS')

plt.minorticks_on()

# Customize the major grid
plt.grid(which='major', linestyle='-', linewidth='0.1', color='grey')
# Customize the minor grid
plt.grid(which='minor', linestyle=':', linewidth='0.1', color='grey')

# Turn off the display of all ticks.
plt.tick_params(which='both', # Options for both major and minor ticks
                top='off', # turn off top ticks
                left='off', # turn off left ticks
                right='off',  # turn off right ticksS
                bottom='off') # turn off bottom ticks

# giving a title to my graph
#plt.title('Some cool customization

ax.spines['right'].set_linewidth(2)
ax.spines['left'].set_linewidth(2) 
ax.spines['top'].set_linewidth(2)  # setting up Y-axis tick color to red
ax.spines['bottom'].set_linewidth(2)
#plt.plot(x1, y1,c='#009933',label='Ice',Lw = 4)

plt.xlim(0,1)
plt.ylim(0,1)
plt.yticks(np.linspace(0,1,11))
plt.xticks(np.linspace(0,1,11))


plt.xlabel("False Positive Rate",fontweight ='bold', fontsize = 15,fontname="Times New Roman")
plt.ylabel("True Positive Rate",fontweight ='bold', fontsize = 15,fontname="Times New Roman")
#plt.title('Success plots of OPE - Illumination Variation', fontweight ='bold', fontsize = 17,fontname="Times New Roman")
# set the spacing between subplots
# using padding
plt.margins(x=0.1, y=0.5)
plt.legend(loc="lower right")
plt.show()
#-----------------------------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
fig,ax = plt.subplots(figsize =(7, 6)) 

x1=[0.006508036,
0.054111763,
0.048919958,
0.05029469,
0.040920192,
0.043494157,
1

] 
y1=[0,
0.000175498,
0.931353281,
0.381919359,
0.547428229,
0.95284233,
0.972958743

]
x2 = [0.004299691,
0.002413092,
0.021381459,
0.018807494,
0.038287728,
0.040130453,
0.068195446,
0.124398555,
0.13726838,
0.230296737,
0.230399111,
0.249806221,
0.308203051,
0.309958027,
0.32075113,
0.320634131,
0.335741551,
0.355221785,
0.433098849,
0.95

]
      

y2 =[
 0.00676397,
0.557416968,
0.560875733,
0.655461632,
0.655527443,
0.729857993,
0.746844699,
0.763926467,
0.790996973,
0.801446393,
0.77779809,
0.794755546,
0.805087968,
0.899688492,
0.906481712,
0.933508343,
0.943694517,
0.943760329,
0.954158562,
0.959445428

]
x3=[
 0.006493412,
0.136888135,
0.15192243,
0.162774032,
0.162627784,
0.179914299,
0.9589


]
# first plot with X and Y data
y3=[ 
0,
0.878833526,
0.905911344,
0.899191249,
0.932974538,
0.939789695,
0.962823757

]
x4=[0.006449537,
0.059025696,
0.98


]
y4=[
0.010149612,
0.365056964,
0.95268877


]


x6=[
0.006478787,
0.001842725,
0.044035275,
0.182078769,
0.98

]
y6=[     
0.003392954,
0.574323237,
0.827844158,
0.939797008,
0.959445428

]
x7=[0.006508036,
0.171300291,
0.99

]
y7=[0.003363704,
0.929625459,
0.966202085

]

# second plt with x1 and y1 data

plt.plot(x1,y1,c='#006600',Lw = 2, linestyle='solid',label='Proposed')
plt.plot(x2,y2,c='#B34700',Lw = 2, linestyle='solid',label='DNN')
plt.plot(x3,y3,c='#5C85D6',Lw = 2, linestyle='solid',label='CNN')
plt.plot(x4,y4,c='#ff00ff',Lw = 2, linestyle='solid',label='XGBoost')
#plt.plot(x5,y5,c='#1A0033',Lw = 2, linestyle='dashed',label='Deep STRCF')
plt.plot(x6,y6,c='#806000',Lw = 2, linestyle='solid',label='RF')
plt.plot(x7,y7,c='#99003D',Lw = 2, linestyle='solid',label='I-SiamIDS')

plt.minorticks_on()

# Customize the major grid
plt.grid(which='major', linestyle='-', linewidth='0.1', color='grey')
# Customize the minor grid
plt.grid(which='minor', linestyle=':', linewidth='0.1', color='grey')

# Turn off the display of all ticks.
plt.tick_params(which='both', # Options for both major and minor ticks
                top='off', # turn off top ticks
                left='off', # turn off left ticks
                right='off',  # turn off right ticksS
                bottom='off') # turn off bottom ticks

# giving a title to my graph
#plt.title('Some cool customization

ax.spines['right'].set_linewidth(2)
ax.spines['left'].set_linewidth(2) 
ax.spines['top'].set_linewidth(2)  # setting up Y-axis tick color to red
ax.spines['bottom'].set_linewidth(2)
#plt.plot(x1, y1,c='#009933',label='Ice',Lw = 4)

plt.xlim(0,1)
plt.ylim(0,1)
plt.yticks(np.linspace(0,1,11))
plt.xticks(np.linspace(0,1,11))


plt.xlabel("False Positive Rate",fontweight ='bold', fontsize = 15,fontname="Times New Roman")
plt.ylabel("True Positive Rate",fontweight ='bold', fontsize = 15,fontname="Times New Roman")
#plt.title('Success plots of OPE - Illumination Variation', fontweight ='bold', fontsize = 17,fontname="Times New Roman")
# set the spacing between subplots
# using padding
plt.margins(x=0.1, y=0.5)
plt.legend(loc="lower right")
plt.show()
#-----------------------------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
fig,ax = plt.subplots(figsize =(7, 6)) 

x1=[0.00204499,
0.010551822,
0,
0.006508036,
0.033444628,
0.175354658,
1

] 
y1=[0,
0.003144654,
0.536149501,
0.00641077,
0.000175498,
0.931353281,
0.972958743

]
x2 = [0,
0.055139963,
0.064057704,
0.038287728,
0.208540564,
0.040130453,
0.083066456,
0.124398555,
0.13726838,
0.230296737,
0.230399111,
0.249806221,
0.168184332,
0.309958027,
0.9,


]
      

y2 =[
  0,
0.106571996,
0.268184332,
0.2308993711,
0.755527443,
0.424818619,
0.5555527443,
0.601446393,
0.76981132,
0.804755546,
0.846844699,
0.861421734,
0.895087968,
0.89688492,
0.95

]
x3=[
 0.006493412,
0.136888135,
0.15192243,
0.162774032,
0.162627784,
0.179914299,
0.9589


]
# first plot with X and Y data
y3=[ 
0,
0.878833526,
0.905911344,
0.899191249,
0.932974538,
0.939789695,
0.962823757

]
x4=[0.006449537,
0.059025696,
0.98


]
y4=[
0.010149612,
0.365056964,
0.95268877


]


x6=[
0.006478787,
0.001842725,
0.044035275,
0.182078769,
0.98

]
y6=[     
0.003392954,
0.574323237,
0.827844158,
0.939797008,
0.959445428

]
x7=[0.006508036,
0.171300291,
0.99

]
y7=[0.003363704,
0.929625459,
0.966202085

]

# second plt with x1 and y1 data

plt.plot(x1,y1,c='#006600',Lw = 2, linestyle='solid',label='Proposed')
plt.plot(x2,y2,c='#B34700',Lw = 2, linestyle='solid',label='DNN')
plt.plot(x3,y3,c='#5C85D6',Lw = 2, linestyle='solid',label='CNN')
plt.plot(x4,y4,c='#ff00ff',Lw = 2, linestyle='solid',label='XGBoost')
#plt.plot(x5,y5,c='#1A0033',Lw = 2, linestyle='dashed',label='Deep STRCF')
plt.plot(x6,y6,c='#806000',Lw = 2, linestyle='solid',label='RF')
plt.plot(x7,y7,c='#99003D',Lw = 2, linestyle='solid',label='I-SiamIDS')

plt.minorticks_on()

# Customize the major grid
plt.grid(which='major', linestyle='-', linewidth='0.1', color='grey')
# Customize the minor grid
plt.grid(which='minor', linestyle=':', linewidth='0.1', color='grey')

# Turn off the display of all ticks.
plt.tick_params(which='both', # Options for both major and minor ticks
                top='off', # turn off top ticks
                left='off', # turn off left ticks
                right='off',  # turn off right ticksS
                bottom='off') # turn off bottom ticks

# giving a title to my graph
#plt.title('Some cool customization

ax.spines['right'].set_linewidth(2)
ax.spines['left'].set_linewidth(2) 
ax.spines['top'].set_linewidth(2)  # setting up Y-axis tick color to red
ax.spines['bottom'].set_linewidth(2)
#plt.plot(x1, y1,c='#009933',label='Ice',Lw = 4)

plt.xlim(0,1)
plt.ylim(0,1)
plt.yticks(np.linspace(0,1,11))
plt.xticks(np.linspace(0,1,11))


plt.xlabel("False Positive Rate",fontweight ='bold', fontsize = 15,fontname="Times New Roman")
plt.ylabel("True Positive Rate",fontweight ='bold', fontsize = 15,fontname="Times New Roman")
#plt.title('Success plots of OPE - Illumination Variation', fontweight ='bold', fontsize = 17,fontname="Times New Roman")
# set the spacing between subplots
# using padding
plt.margins(x=0.1, y=0.5)
plt.legend(loc="lower right")
plt.show()
#----------------------------------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
fig,ax = plt.subplots(figsize =(7, 6)) 
  
x = [0.002066596,
0.046653406,
0.091304797,
0.127741469,
0.160096613,
0.186510294,
0.193898375,
0.210172819,
0.225426881,
0.245498695,
0.275748495,
0.998140064,


]

y =[0.009480509,
0.066751053,
0.127160239,
0.200175661,
0.27316525,
0.301813438,
0.491733616,
0.504494846,
0.767249619,
0.849655137,
0.938454186,
0.952519956,

]
  
# first plot with X and Y data

  
x1 = [0.00410736,
0.024153341,
0.023972514,
0.037715378,
0.037612048,
0.033478856,
0.036901656,
0.040879853,
0.051083671,
0.055087701,
0.063069928,
0.132171734,
0.132081321,
0.185141174,
0.199026117,
0.217018419,
0.239130996,
0.955232363

]

y1 = [0.009467593,
0.079266875,
0.123569528,
0.256567901,
0.281883702,
0.294515771,
0.455929839,
0.481271473,
0.481336054,
0.500348738,
0.544703056,
0.614760662,
0.636911989,
0.637247811,
0.735436956,
0.827322983,
0.909741417,
0.964906616

]
x2=[0,
0.30252383,
0.906266953,

]
y2=[0.003164475,
0.878497068,
0.961432151

]
x3=[0.002066596,
0.038038284,
0.062462866,
0.136343675,
0.284040712,
0.94097285

]
y3=[0.009480509,
0.17745602,
0.193433391,
0.592635168,
0.906861099,
0.958487252

]
x4=[0.004094443,
0.018173129,
0.017940637,
0.023675441,
0.033595102,
0.046291752,
0.255547519,
0.467735269,
0.47570458,
0.934850559

]
y4=[0.006303118,
0.044418899,
0.101379453,
0.196352458,
0.266035494,
0.655356359,
0.88769342,
0.901694609,
0.949213402,
0.958448503

]
x5=[0.004029862,
0.04263646,
0.058407171,
0.104905582,
0.155730929,
0.157668363,
0.167872181,
0.165766836,
0.173787812,
0.193084653,
0.239453902,
0.26974245,
0.938945003

]
y5=[0.015848209,
0.050902844,
0.187078608,
0.294967839,
0.342757872,
0.36808659,
0.368151172,
0.383960631,
0.418821524,
0.691095554,
0.830629537,
0.909935161,
0.95530986

]
# second plot with x1 and y1 data
plt.plot(x, y,c='#006600',Lw = 2, linestyle='solid',label='Proposed')
plt.plot(x1, y1,c='#B34700',Lw = 2, linestyle='solid',label='DNN')
plt.plot(x2, y2,c='#5C85D6',Lw = 2, linestyle='solid',label='CNN')
plt.plot(x3, y3,c='#ff00ff',Lw = 2, linestyle='solid',label='XGBoost')
plt.plot(x4, y4,c='#806000',Lw = 2, linestyle='solid',label='RF')
plt.plot(x5, y5,c='#99003D',Lw = 2, linestyle='solid',label='I-SiamIDS')
#plt.plot(x1, y1,c='#009933',label='Ice',Lw = 4)

plt.minorticks_on()

# Customize the major grid
plt.grid(which='major', linestyle='-', linewidth='0.1', color='grey')
# Customize the minor grid
plt.grid(which='minor', linestyle=':', linewidth='0.1', color='grey')

# Turn off the display of all ticks.
plt.tick_params(which='both', # Options for both major and minor ticks
                top='off', # turn off top ticks
                left='off', # turn off left ticks
                right='off',  # turn off right ticksS
                bottom='off') # turn off bottom ticks

# giving a title to my graph
#plt.title('Some cool customization

ax.spines['right'].set_linewidth(2)
ax.spines['left'].set_linewidth(2) 
ax.spines['top'].set_linewidth(2)  # setting up Y-axis tick color to red
ax.spines['bottom'].set_linewidth(2)
#plt.plot(x1, y1,c='#009933',label='Ice',Lw = 4)

plt.xlim(0,1)
plt.ylim(0,1)
plt.yticks(np.linspace(0,1,11))
plt.xticks(np.linspace(0,1,11))


plt.xlabel("False Positive Rate",fontweight ='bold', fontsize = 15,fontname="Times New Roman")
plt.ylabel("True Positive Rate",fontweight ='bold', fontsize = 15,fontname="Times New Roman")
#plt.title('Success plots of OPE - Illumination Variation', fontweight ='bold', fontsize = 17,fontname="Times New Roman")
# set the spacing between subplots
# using padding
plt.margins(x=0.1, y=0.5)
plt.legend(loc="lower right")
plt.show()
