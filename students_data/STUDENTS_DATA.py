### Multinomial Regression #### 

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

student_data = pd.read_csv(r"E:\DESKTOPFILES\suraj\assigments\multinominal regressions\Datasets_Multinomial\mdata.csv")

student_data.head(10)

student_data.columns

#### Removing unwanted columns 

student_data = student_data.drop('Unnamed: 0', axis=1)
student_data = student_data.drop('id', axis=1)

student_data.describe()

student_data.columns

#### Converting Categorical Columns into Numerical Columns
student_data = pd.get_dummies(student_data, columns = ["female","schtyp","honors"], drop_first=True)

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()

student_data['ses']  = lb.fit_transform(student_data['ses'])

student_data.columns

######## Rearranging columns
student_data = student_data[['prog','ses','read','write','math','science','female_male','schtyp_public','honors_not enrolled']]

student_data.prog.value_counts() ### Each category count in the variable prog

# Boxplot of independent variable distribution for each category of prog 
sns.boxplot(x = "prog", y = "ses", data = student_data)
sns.boxplot(x = "prog", y = "read", data = student_data)
sns.boxplot(x = "prog", y = "write", data = student_data)
sns.boxplot(x = "prog", y = "math", data = student_data)
sns.boxplot(x = "prog", y = "science", data = student_data)
sns.boxplot(x = "prog", y = "female_male", data = student_data)
sns.boxplot(x = "prog", y = "schtyp_public", data = student_data)
sns.boxplot(x = "prog", y = "honors_not enrolled", data = student_data)


# Scatter plot for each categorical programe of student
sns.stripplot(x = "prog", y = "ses", jitter = True, data = student_data)
sns.stripplot(x = "prog", y = "read", jitter = True, data = student_data)
sns.stripplot(x = "prog", y = "write", jitter = True, data = student_data)
sns.stripplot(x = "prog", y = "math", jitter = True, data = student_data)
sns.stripplot(x = "prog", y = "science", jitter = True, data = student_data)
sns.stripplot(x = "prog", y = "female_male", jitter = True, data = student_data)
sns.stripplot(x = "prog", y = "schtyp_public", jitter = True, data = student_data)
sns.stripplot(x = "prog", y = "honors_not enrolled", jitter = True, data = student_data)


# Scatter plot between each possible pair of independent variable and also histogram for each independent variable 
sns.pairplot(student_data) # Normal
sns.pairplot(student_data, hue = "prog") # With showing the category of each prog in the scatter plot

# Correlation values between each independent features
a=student_data.corr()

train, test = train_test_split(student_data, test_size = 0.2)

# ‘multinomial’ option is supported only by the ‘lbfgs’ and ‘newton-cg’ solvers
model = LogisticRegression(multi_class = "multinomial", solver = "newton-cg").fit(train.iloc[:, 1:], train.iloc[:, 0])
help(LogisticRegression)

test_predict = model.predict(test.iloc[:, 1:]) # Test predictions

# Test accuracy 
accuracy_score(test.iloc[:,0], test_predict) #0.575

train_predict = model.predict(train.iloc[:, 1:]) # Train predictions 
# Train accuracy 
accuracy_score(train.iloc[:,0], train_predict)#0.66875

# Test accuracy and Train accuracy is little bit far so, we cannot accept this model