import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
url='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_mod1.csv'
df=pd.read_csv(url)
print(df.head())
##finding missing values
missing_data=df.isnull()
print(missing_data.head())
for columns in missing_data.columns.values.tolist():
    print(columns)
    print(missing_data[columns].value_counts())
    print("")

##replacing missing values using mean
mean=df['Weight_kg'].mean()
df.replace(np.nan,mean,inplace=True)
print(df.iloc[29]['Weight_kg'])

##data formatting
df = df[(df['Screen_Size_cm'] != '?') & (df['Weight_kg'] != '?')]
df['Screen_Size_cm'] = df['Screen_Size_cm'].astype('float')
df['Weight_kg'] = df['Weight_kg'].astype('float')
print(df.dtypes)

##data standardization
df["Weight_kg"] = df["Weight_kg"]*2.205
df.rename(columns={'Weight_kg':'Weight_pounds'}, inplace=True)
df["Screen_Size_cm"] = df["Screen_Size_cm"]/2.54
df.rename(columns={'Screen_Size_cm':'Screen_Size_inch'}, inplace=True)
print(df.head())

##data normalization
df['CPU_frequency']=df['CPU_frequency']/df['CPU_frequency'].max()
print(df['CPU_frequency'])

##binning
bins=np.linspace(min(df['Price']),max(df['Price']),4)
group_names=['low','medium','high']
df['Price-binned']=pd.cut(df['Price'],bins,labels=group_names,include_lowest=True)
print(df.head())

##plotting
plt.bar(group_names,df['Price-binned'].value_counts())
plt.xlabel("Price")
plt.ylabel("Count")
plt.title("Price-bins")
plt.show()

##indicator variables
dummy_variable_1 = pd.get_dummies(df["Screen"])
dummy_variable_1.rename(columns={'IPS Panel':'Screen-IPS_panel', 'Full HD':'Screen-Full_HD'}, inplace=True)
df = pd.concat([df, dummy_variable_1], axis=1)
df.drop("Screen", axis = 1, inplace=True)
print(df.head())
print(df.dtypes)

########EDA########
##correlations
sns.regplot(x='CPU_frequency',y='Price',data=df)
plt.ylim(0,)
plt.show()
sns.regplot(x='Screen_Size_inch',y='Price',data=df)
plt.ylim(0,)
plt.show()
sns.regplot(x='Weight_pounds',y='Price',data=df)
plt.ylim(0,)
plt.show()
for param in ["CPU_frequency", "Screen_Size_inch","Weight_pounds"]:
    print(f"Correlation of Price and {param} is ", df[[param,"Price"]].corr())

##boxplots
sns.boxplot(x='Category',y='Price',data=df)
plt.show()
sns.boxplot(x='GPU',y='Price',data=df)
plt.show()
sns.boxplot(x='OS',y='Price',data=df)
plt.show()
sns.boxplot(x='CPU_core',y='Price',data=df)
plt.show()
sns.boxplot(x='RAM_GB',y='Price',data=df)
plt.show()
sns.boxplot(x='Storage_GB_SSD',y='Price',data=df)
plt.show()

##desciptive statistics
print(df.describe(include='object'))

##groupby and pivot tables
df_gptest = df[['GPU','CPU_core','Price']]
grouped_test1 = df_gptest.groupby(['GPU','CPU_core'],as_index=False).mean()
print(grouped_test1)
grouped_pivot = grouped_test1.pivot(index='GPU',columns='CPU_core')
print(grouped_pivot)
plt.pcolor(grouped_pivot,cmap='RdBu')
plt.show()

##simple linear model
lm=LinearRegression()
X=df[['CPU_frequency']]
y=df['Price']
lm.fit(X,y)
yhat=lm.predict(X)
print(yhat)

##distribution plot
sns.kdeplot(df['Price'], color='r', label='Actual Value', shade=True)
sns.kdeplot(yhat, color='y', label='Fitted Values', shade=True)
plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price')
plt.ylabel('Proportion of laptops')
plt.legend(['Actual Value', 'Predicted Value'])
plt.show()

##meansquared error and R-squared value
ans1=mean_squared_error(y,yhat)
print(ans1)
print(lm.score(X,y))

##multiple linear regression
Y=df[['CPU_frequency','RAM_GB','Storage_GB_SSD','OS','CPU_core','GPU','Category']]
lm.fit(Y,y)
yhat1=lm.predict(Y)
print(yhat1)

##distribution plot
sns.kdeplot(df['Price'], color='r', label='Actual Value', shade=True)
sns.kdeplot(yhat1, color='y', label='Fitted Values', shade=True)
plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price')
plt.ylabel('Proportion of laptops')
plt.legend(['Actual Value', 'Predicted Value'])
plt.show()

##meansquared error and R-squared value
ans2=mean_squared_error(y,yhat1)
print(ans2)
print(lm.score(Y,y))

##polynomial regression
x=df['CPU_frequency']
f1 = np.polyfit(x,y, 1)
p1 = np.poly1d(f1)

f3 = np.polyfit(x, y, 3)
p3 = np.poly1d(f3)

f5 = np.polyfit(x, y, 5)
p5 = np.poly1d(f5)

##plots
def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(independent_variable.min(),independent_variable.max(),100)
    y_new = model(x_new)
    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title(f'Polynomial Fit for Price ~ {Name}')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of laptops')
    plt.show()

PlotPolly(p1,x,y,'CPU_frequency')
PlotPolly(p3,x,y,'CPU_frequency')
PlotPolly(p5,x,y,'CPU_frequency')

##Pipelines
Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]
pipe=Pipeline(Input)
Z = Y.astype(float)
pipe.fit(Z,y)
ypipe=pipe.predict(Z)
print(ypipe)

##training and test data sets
y_data=df['Price']
x_data=df.drop('Price',axis=1)
x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.1,random_state=0)
print(x_train,'\n',x_test,'\n',y_train,'\n',y_test)

##rscores
lre=LinearRegression()
lre.fit(x_train[['CPU_frequency']],y_train)
print(lre.score(x_train[['CPU_frequency']],y_train))
print(lre.score(x_test[['CPU_frequency']],y_test))
scores=cross_val_score(lre,x_data[['CPU_frequency']],y_data,cv=4)
print(scores)
print(scores.mean(),scores.std())

##overfitting
x_train2,x_test2,y_train2,y_test2=train_test_split(x_data,y_data,test_size=0.5,random_state=0)
print(x_train2,'\n',x_test2,'\n',y_train2,'\n',y_test2)
Rsqu_test = []
order = [1, 2, 3, 4, 5]
for n in order:
    pr = PolynomialFeatures(degree=n)
    x_train_pr = pr.fit_transform(x_train[['CPU_frequency']])
    x_test_pr = pr.fit_transform(x_test[['CPU_frequency']])    
    lre.fit(x_train_pr, y_train)
    Rsqu_test.append(lre.score(x_test_pr, y_test))
print(Rsqu_test)
plt.plot(order, Rsqu_test)
plt.xlabel('order')
plt.ylabel('R^2')
plt.title('R^2 Using Test Data')
plt.show()

##Ridge regression
pr=PolynomialFeatures(degree=2)
x_train_pr=pr.fit_transform(x_train[['CPU_frequency', 'RAM_GB', 'Storage_GB_SSD', 'CPU_core', 'OS', 'GPU', 'Category']])
x_test_pr=pr.fit_transform(x_test[['CPU_frequency', 'RAM_GB', 'Storage_GB_SSD', 'CPU_core', 'OS', 'GPU', 'Category']])
print(x_train,'\n',x_test)

##grid search
parameters = [{'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10]}]
RR = Ridge()
Grid = GridSearchCV(RR, parameters, cv=4)
Grid.fit(x_train_pr, y_train)  
BestRR = Grid.best_estimator_
print(BestRR.score(x_test_pr, y_test))