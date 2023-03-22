# Ex02-Outlier

You are given bhp.csv which contains property prices in the city of banglore, India. You need to examine price_per_sqft column and do following,

(1) Remove outliers using IQR 

(2) After removing outliers in step 1, you get a new dataframe.

(3) use zscore of 3 to remove outliers. This is quite similar to IQR and you will get exact same result

(4) for the data set height_weight.csv find the following

    (i) Using IQR detect weight outliers and print them

    (ii) Using IQR, detect height outliers and print them
    
  #EXPLANATION

An Outlier is an observation in a given dataset that lies far from the rest of the observations. That means an outlier is vastly larger or smaller than the remaining values in the set. An outlier is an observation of a data point that lies an abnormal distance from other values in a given population. (odd man out).Outliers badly affect mean and standard deviation of the dataset. These may statistically give erroneous results.Most machine learning algorithms do not work well in the presence of outlier. So it is desirable to detect and remove outliers.Outliers are highly useful in anomaly detection like fraud detection where the fraud transactions are very different from normal transactions.

#ALGORITHM

##STEP 1

Read the given Data.

##STEP 2

Get the information about the data.

##STEP 3

Detect the Outliers using IQR method and Z score.

##STEP 4

Remove the outliers.

##STEP 5

Plot the datas using Box Plot.

#PROGRAM

Developed by : HEMASONICA.P

Registration Number : 212222230048

 ```
 
import pandas as ps
import numpy as np
import seaborn as sns

df=ps.read_csv("bhp.csv")
df

df.head()
df.describe()
df.info()
df.isnull().sum()
df.shape

sns.boxplot(x="price_per_sqft",data=df)

q1=df['price_per_sqft'].quantile(0.35)
q3=df['price_per_sqft'].quantile(0.65)
print("First Quantile =",q1,"Second quantile =",q3)

IQR=q3-q1 #INTERQUARTILE RANGE
ul =q3+0.5*IQR
ll =q1-1.5*IQR

df1=df[((df['price_per_sqft']<=l1)&(df['price_per_sqft']>u1))]
df1

df1.shape

sns.boxplot(x='price_per_sqft',data=df1)

from scipy import stats
z=np.abs(stats.zscore(df['price_per_sqft']))
df2=df[(z<3)]
df2

print(df2.shape)

sns.boxplot(x='price_per_sqft',data=df2)

df3=ps.read_csv('height_weight.csv')
df3

df3.head()
df3.info()
df3.describe()
df3.isnull().sum()
df3.shape

sns.boxplot(x='weight',data=df3)

q1=df3['weight'].quantile(0.25)
q3=df3['weight'].quantile(0.75)
print('First Quantile =',q1,'Second Quantile =',q3)

IQR=q3-q1
u1=q3+1.5*IQR
l1=q1-1.5*IQR

df4 =df3[((df3['height']>=l1)&(df3['height']<=u1))]
df4

df4.shape

sns.boxplot(x='height',data=df4)

```

#OUTPUT

DATASET FOR BHP_CSV

![image](https://user-images.githubusercontent.com/118361409/226977685-d923deeb-a5a9-4b82-a95e-2b08ffdeb257.png)

DATASET HEAD(BHP)

![image](https://user-images.githubusercontent.com/118361409/226978228-6b75dae5-513a-407b-bfcb-a147b40969ff.png)

DATASET DESCRIBE(BHP)

![image](https://user-images.githubusercontent.com/118361409/226978355-650b3677-f6ca-4d98-b542-4da017d64c14.png)

DATASET INFO(BHP)

![image](https://user-images.githubusercontent.com/118361409/226978533-62355395-5a4d-40e7-bad4-4913375b526a.png)

DATASET NULL VALUES(BHP)

![image](https://user-images.githubusercontent.com/118361409/226978600-679740a2-1e8d-4ab3-8ab2-dbee7293cd60.png)

DATASET SHAPE WITH OUTLIERS(BHP)

![image](https://user-images.githubusercontent.com/118361409/226978848-a3e3ee20-6fd1-4850-b1a0-46ec3e4a7d48.png)

DATASET BOXPLOT WITH OUTLIERS(BHP)

![image](https://user-images.githubusercontent.com/118361409/226979709-4c150b39-3852-4864-8e08-91680a984c71.png)


DATASET WITHOUT OUTLIERS(BHP)

![image](https://user-images.githubusercontent.com/118361409/226979841-c0a9f411-5c77-403e-b164-8de960f7687f.png)


![image](https://user-images.githubusercontent.com/118361409/226979903-dbe06397-01b6-4655-8861-71cccb134000.png)



DATASET SHAPE WITHOUT OUTLIERS(BHP)

![image](https://user-images.githubusercontent.com/118361409/226980032-902d8aad-a51a-49d2-a9fa-601c218e7170.png)


DATASET BOXPLOT WITHOUT OUTLIERS(BHP)

![image](https://user-images.githubusercontent.com/118361409/226980217-0d741770-65e3-4ef4-97e8-9047dbb8a0d8.png)


DATASET AFTER REMOVAL OF OUTLIERS USING Z-SCORE(BHP)

![image](https://user-images.githubusercontent.com/118361409/226980332-3d86cfdf-ecbb-41e9-be6a-eea801a8fb82.png)



DATASET SHAPE AFTER REMOVAL OF OUTLIERS USING Z-SCORE(BHP)

![image](https://user-images.githubusercontent.com/118361409/226980420-cdd675b8-a27a-4603-abee-8545d8643cf1.png)


DATASET BOXPLOT AFTER REMOVAL OF OUTLIERS USING Z-SCORE(BHP)

![image](https://user-images.githubusercontent.com/118361409/226980525-38956c15-04dd-4ffd-8773-c6eeabac0a4b.png)


DATASET FOR WEIGHT_HEIGHT_CSV

![image](https://user-images.githubusercontent.com/118361409/226980654-b5d05c7a-6e9a-4a8d-8b01-60485983dc9b.png)


DATASET HEAD(WEIGHT_HEIGHT)

![image](https://user-images.githubusercontent.com/118361409/226980736-1a8f7341-6513-4d81-82d6-8120a03b1c03.png)


DATASET INFO(WEIGHT_HEIGHT)

![image](https://user-images.githubusercontent.com/118361409/226980826-9a1c33ef-102a-4c6a-a9f7-ecb4ed42b6fa.png)


DATASET DESCRIBE(WEIGHT_HEIGHT)

![image](https://user-images.githubusercontent.com/118361409/226980889-89e7eff3-f0ae-48a4-9fba-e14230eb7927.png)


DATASET NULL VALUES(WEIGHT_HEIGHT)

![image](https://user-images.githubusercontent.com/118361409/226980969-f8e068ee-185d-4a1e-9162-9c621baa0352.png)


DATASET BOXPLOT WITH OUTLIERS(WEIGHT_HEIGHT)

![image](https://user-images.githubusercontent.com/118361409/226981049-b0a304ad-a20e-4859-81df-cb6dd5af55e2.png)


DATASET AFTER REMOVING OUTLIERS USING IQR METHOD(WEIGHT_HEIGHT)

![image](https://user-images.githubusercontent.com/118361409/226981200-4e59f131-9bd2-4806-b14a-0366919e6757.png)

![image](https://user-images.githubusercontent.com/118361409/226981295-d928c3be-7aa2-4c45-95ac-a70bac6b20b0.png)


DATASET SHAPE(WEIGHT_HEIGHT)

![image](https://user-images.githubusercontent.com/118361409/226981369-cafbdaff-de98-4515-8bb1-8998b696eebc.png)


DATASET BOXPLOT AFTER REMOVING OUTLIERS USING IQR METHOD(WEIGHT_HEIGHT)

![image](https://user-images.githubusercontent.com/118361409/226981468-73ab8009-9749-49e6-8218-d21456954f74.png)


RESULT

The given datasets are read and outliers are detected and are removed using IQR and z-score methods.

