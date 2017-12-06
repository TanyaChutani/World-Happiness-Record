#Importing packages
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from wordcloud import WordCloud
from collections import Counter
import plotly.graph_objs as go
#from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.offline
#import geopy


#1
df=pd.read_csv('2015.csv')
dff=pd.read_csv('2016.csv')
df['Year']=2015
dff['Year']=2016
dat = pd.concat([df, dff])

#2015
print(df.head(2))

print('The Top 20 Happy Countries of 2015:')
top20_df_15=df['Country'].head(20)
print(top20_df_15)
print('The Top 20 Happy Countries of 2016:')
top20_df_16=dff['Country'].head(20)
print(top20_df_16)

#Max happiness in 2015
print(df[df['Happiness Score'] == df['Happiness Score'].max()][['Country','Happiness Score', 'Happiness Rank']])
#SwitzerLand
#Max happiness in 2015
print(dff[dff['Happiness Score'] == dff['Happiness Score'].max()][['Country','Happiness Score', 'Happiness Rank']])
#Denmark
#Min Happiness in 2015
print(df[df['Happiness Score'] == df['Happiness Score'].min()][['Country','Happiness Score', 'Happiness Rank']])
#Min Happiness in 2016
print(dff[dff['Happiness Score'] == dff['Happiness Score'].min()][['Country','Happiness Score', 'Happiness Rank']])

#sns.heatmap(df.corr(),annot=True,cmap='Oranges')
#plt.show()
plt.figure(figsize=(16,9))
sns.heatmap(df.corr()[['Happiness Score']].sort_values('Happiness Score'), annot=True)
#sns.heatmap(df.corr(),annot=True,cmap='Blues')
#plt.show()
plt.figure(figsize=(16,9))
sns.heatmap(dff.corr()[['Happiness Score']].sort_values('Happiness Score'), annot=True)

plt.figure(figsize=(12,6))
sns.boxplot('Region', 'Happiness Score', data = df)
plt.xticks(rotation = 60)
plt.show()
sns.barplot(x=df['Happiness Score'],y='Region',data=df)
plt.show()
sns.barplot(x='Happiness Score',y='Region',hue='Year',hue_order=[2015, 2016],data=dat)
plt.show()

plt.figure(figsize=(10,6))
list = df.sort_values(by=['Happiness Rank'],ascending=True)['Region'].head(30).value_counts()
list.plot(kind = 'bar', color = 'grey')
plt.show()
plt.figure(figsize=(10,6))
list = dff.sort_values(by=['Happiness Rank'],ascending=True)['Region'].head(30).value_counts()
list.plot(kind = 'bar', color = 'grey')
plt.show()

plt.figure(figsize=(14,13))
sns.pointplot(x='Year', y='Happiness Score', hue ='Region', data=dat,size=4, aspect=.7)
#print(dff.head(2))
plt.grid(axis= 'both')
plt.show()

top10=dat['Country'].head(10)
top1=df['Country'].head(10)
top=dff['Country'].head(10)

plt.title('2015')
plt.xlim(7, 7.7)
sns.barplot(x=df['Happiness Score'],y=top1,data=df, palette= 'Greens_d')
plt.show()
plt.title('2016')
plt.xlim([7, 7.7])
sns.barplot(x=dff['Happiness Score'],y=top,data=df, color= 'salmon')
plt.show()
plt.xlim([7, 7.7])
sns.barplot(x='Happiness Score',y=top10,hue='Year',hue_order=[2015, 2016],data=dat, palette= 'Blues_d')
plt.show()

plt.title('2015')
df['Region'].value_counts().plot.pie(subplots=True, figsize=(8, 8), autopct='%.2f')
plt.show()
plt.title('2016')
dff['Region'].value_counts().plot.pie(subplots=True, figsize=(8, 8), autopct='%.2f')
plt.show()

plt.figure(figsize= (10,10))
#plt.subplot(1, 2, 1)
sns.kdeplot(df['Happiness Score'], c= 'b')
plt.xticks([2, 3, 4, 5, 6, 7, 8])
#plt.subplot(1, 2, 2)
sns.kdeplot(dff['Happiness Score'], c= 'k')
plt.title('2015-Blue\n2016-Black')
plt.show()

g= sns.lmplot(x="Happiness Score", y="Health (Life Expectancy)", hue="Year",
               truncate=True, size=5, data=dat)


g= sns.lmplot(x="Happiness Score", y="Economy (GDP per Capita)", hue="Year",
               truncate=True, size=5, data=dat)

g= sns.lmplot(x="Happiness Score", y="Family", hue="Year",
               truncate=True, size=5, data=dat)

g= sns.lmplot(x="Happiness Score", y="Freedom", hue="Year",
               truncate=True, size=5, data=dat)

g= sns.lmplot(x="Happiness Score", y="Trust (Government Corruption)", hue="Year",
               truncate=True, size=5, data=dat)

#sns.pairplot(data = df.drop(['Country', 'Region', 'Happiness Rank', 'Standard Error', 'Dystopia Residual'], axis = 1))

dt=dat.drop(['Standard Error','Lower Confidence Interval','Upper Confidence Interval'],axis=1)
sns.set(style="ticks")
sns.pairplot(dt, hue="Year")

common = dff.loc[:, ['Country', 'Region']].merge(df.loc[:, ['Country']],on=['Country'])
list_new_countries= np.array(dff[(~dff.Country.isin(common.Country))].Country)
new_l= []
for i in range(len(list_new_countries)):
    new_l.append(list_new_countries.reshape(6, -1)[i][0])
    print(new_l[i])

ar= []
ar.append([i, (-dff[dff['Country']==i].index.values.astype(int)[0]+df[df['Country']==i].index.values.astype(int)[0])] for i in dff.Country if i not in new_l)
#, dff.loc[dff[dff['Country']==i].index.values.astype(int)[0]], ['Region']
emp_l= []
for x in ar:
    for y in x:
        emp_l.append(y)
c= [i[0] for i in sorted(emp_l,key=lambda x: (x[1]), reverse= True)]
reg= [dff['Region'][dff['Country']==i] for i in c]
reg_1= []
for i in reg:
    for k in i:
        reg_1.append(k)
#print(reg_1)
        
#Countries= sorted(emp_l,key=lambda x: (x[1]), reverse= True)
rise_fall= [i[1] for i in sorted(emp_l,key=lambda x: (x[1]), reverse= True)]
#regions= [i[2] for i in sorted(emp_l,key=lambda x: (x[1]), reverse= True)]
df_1= pd.DataFrame({'Country': c, 'Rise_Fall': rise_fall, 'Regions': reg_1})

df_1_2= pd.concat([df_1.head(), df_1.tail()])
df_1_2= df_1_2.reset_index()
#print(df_1_2)

fig, ax = plt.subplots(figsize= (10, 10))
rects1 = ax.bar(range(len(df_1_2)), df_1_2.Rise_Fall)
ax.set_ylabel("Change in Positions")
ax.set_title("TOP MOVERS IN POSITIONS FROM 2015-2016")
plt.xticks(range(len(df_1_2)), df_1_2.Country, rotation= 'vertical')
#ax.legend((rects1[0], rects2[0]), ('Men', 'Women'))
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')
autolabel(rects1)
plt.show()

e= str()
x= [((str(dff.Country[i])+' ')*10) if i<10 else ((str(dff.Country[i])+' ')*5) if i<30 else ((str(dff.Country[i])+' ')*3) if i<50 else (str(dff.Country[i])+' ') for i in range(len(dff))]
for i in x:
    e+= i
#print(np.array(e))
e= str(e)
def word_count(string):
    my_string = string.lower().split()
    my_dict = {}
    for item in my_string:
        if item in my_dict:
            my_dict[item] += 1
        else:
            my_dict[item] = 1
    return my_dict
dicti= word_count(e)
#print(dicti)
plt.figure(figsize= (10, 10))
word_cloud = WordCloud().generate_from_frequencies(dicti)
plt.imshow(word_cloud)
plt.axis("off")
plt.show()

#print(df.columns)
'''
for i in dff['Country'].head(20):
    list.append(dff[dff['Country'== i]], dff[dff['Country']==i].index.values.astype(int)[0]-df[df['Country']==i].index.values.astype(int)[0])
'''
#dff[dff['Country'==i]]['Country'], 
#dff['Shift from 2015']= df.
top_2015= df['Country'].head(20)
top_2016= df['Country'].head(20)
'''
arr= {}
for i in top_2015:
    arr[i]= [df[df['Country'== i]].Country, dff[dff['Country'== i]]['Happiness Score']-df[df['Country'== i]]['Happiness Score']]
'''

data = dict(type = 'choropleth', locations = df['Country'], locationmode = 'country names', z = df['Happiness Rank'], text = df['Country'], colorbar = {'title':'Happiness 2015'})
layout = dict(title = 'Global Happiness', geo = dict(showframe = False, projection = {'type': 'Mercator'}))
choromap3 = go.Figure(data = [data], layout=layout)
plotly.offline.plot(choromap3, auto_open= False, filename='C:/Users/HP/Desktop/first/Happiness_2015.png')

data = dict(type = 'choropleth', locations = dff['Country'], locationmode = 'country names', z = dff['Happiness Rank'], text = dff['Country'], colorbar = {'title':'Happiness 2016'})
#print(data)
layout = dict(title = 'Global Happiness',  geo = dict(showframe = False, projection = {'type': 'Mercator'}))
choromap3 = go.Figure(data =[data], layout=layout)
plotly.offline.plot(choromap3, auto_open= False, filename='C:/Users/HP/Desktop/first/Happiness_2016.png')

print('2015')
X= np.array(df.drop(['Happiness Score', 'Country', 'Region'], 1))
y = np.array(df['Happiness Score'])
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
print('No. of training examples: ',x_train.shape[0],'\nNo. of testing examples: ',x_test.shape[0])
'''
rf = RandomForestRegressor(criterion= 'entropy', n_estimators=20, random_state= 14)
rf.fit(x_train, y_train)
y_pred= rf.predict(x_test)
importances = rf.feature_importances_
sc= StandardScaler()
importances= sc.fit_transform(importances)
print(importances)
ind= np.argsort(importances)

plt.title('Feature Importances')
plt.barh(range(len(ind)), importances[ind], color='b', align='center')
plt.yticks(range(len(ind)), df.columns)
print(rf.feature_importances_)

plt.subplot(2, 1, 2)
print('2016')
'''
X= np.array(df.drop(['Happiness Score', 'Country', 'Region'], 1))
y = np.array(df['Happiness Score'])
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
print('No. of training examples: ',x_train.shape[0],'\nNo. of testing examples: ',x_test.shape[0])

plt.figure(figsize= (7,5))
rf = RandomForestRegressor(n_estimators=20, random_state= 14)
rf.fit(x_train, y_train)
y_pred= rf.predict(x_test)
importances = rf.feature_importances_
ind= np.argsort(importances)
plt.title('Feature Importances 2015')
plt.xticks(range(len(ind)), df.columns[ind], rotation= 90)
plt.yscale("log")
plt.xlabel('Features')
plt.ylabel('Importances')
plt.plot(ind[:],importances[ind[:]],'k.')
plt.show()
#print(rf.feature_importances_)

Predict_rf=rf.predict(x_test)
result_rf=pd.DataFrame({
    'Actual':y_test,
    'Predict':Predict_rf,
    'diff':y_test-Predict_rf
})

plt.figure(figsize= (10, 10))
sns.pointplot(x='Actual',y='Predict',data=result_rf)
plt.xticks(rotation= 75)
plt.show()

X= np.array(dff.drop(['Happiness Score', 'Country', 'Region'], 1))
y = np.array(dff['Happiness Score'])
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
print('No. of training examples: ',x_train.shape[0],'\nNo. of testing examples: ',x_test.shape[0])

rf = RandomForestRegressor(n_estimators=20, random_state= 14)
rf.fit(x_train, y_train)
y_pred= rf.predict(x_test)
importances = rf.feature_importances_
ind= np.argsort(importances)
plt.title('Feature Importances 2016')
plt.xticks(range(len(ind)), dff.columns[ind], rotation= 90)
plt.yscale("log")
plt.xlabel('Features')
plt.ylabel('Importances')
plt.plot(ind[:],importances[ind[:]],'k.')
plt.show()

Predict_rf=rf.predict(x_test)
result_rf=pd.DataFrame({
    'Actual':y_test,
    'Predict':Predict_rf,
    'diff':y_test-Predict_rf
})

plt.figure(figsize= (10, 10))
sns.pointplot(x='Actual',y='Predict',data=result_rf)
plt.xticks(rotation= 75)
plt.show()

from sklearn import svm
svr= svm.SVR(kernel='linear')
svr.fit(x_train, y_train)
y_pred= svr.predict(x_test)
print(svr.score(x_test, y_test))

p_svr=svr.predict(x_test)
result_svr=pd.DataFrame({
    'Actual':y_test,
    'Predict':p_svr,
    'diff':y_test-p_svr
})
plt.figure(figsize= (10, 10))
sns.pointplot(x='Actual',y='Predict',data=result_svr)
plt.xticks(rotation= 75)
plt.show()

X= np.array(df.drop(['Country', 'Region', 'Happiness Score', 'Happiness Rank'], 1))
y= np.array(df['Happiness Score'])
pca = PCA(n_components=2)
pca.fit(X)
transform = pca.transform(X)
plt.figure(figsize= (6,5))
plt.scatter(transform[:,0],transform[:,1], s=y*3, c= y, cmap = "brg")
plt.colorbar()
plt.clim(0,9)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title('2015')
plt.show()
#print(pca.components_)
#print(pca.explained_variance_)
#print(X.shape, x.shape)

X= np.array(dff.drop(['Country', 'Region', 'Happiness Score', 'Happiness Rank'], 1))
y= np.array(dff['Happiness Score'])
pca = PCA(n_components=2)
pca.fit(X)
transform = pca.transform(X)
plt.figure(figsize= (6,5))
plt.scatter(transform[:,0],transform[:,1], s=y*3, c= y, cmap = "jet")
plt.colorbar()
plt.clim(0,9)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title('2016')
plt.show()

'''
# plot data
plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    draw(pca.mean_, pca.mean_ + v)
plt.axis('equal');
X_new = pca.inverse_transform(x)
plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8)
plt.axis('equal')
plt.show()

km= KMeans(n_clusters= 4)
X_clus= km.fit_predict(x)
cmap = {0:'r', 1:'g',2 : 'b', 3: 'k', 4:'y'}
label_col= [cmap[i] for i in X_clus]
plt.figure(figsize= (10, 10))
plt.scatter(x[:,0], x[:, 1], c= label_col, alpha= 0.7)
plt.show()
df = pd.DataFrame(x)
df['X_clus']= X_clus
print(df['X_clus'])

sns.pairplot(df, hue='X_clus')
plt.show()
'''