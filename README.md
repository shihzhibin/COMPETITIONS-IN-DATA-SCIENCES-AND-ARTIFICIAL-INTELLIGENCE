# HW-4

#Load data 
train = pd.read_csv('sales_train.csv')
test = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')
items = pd.read_csv('items.csv')
item_categories = pd.read_csv('item_categories.csv')
shops = pd.read_csv('shops.csv')

#check nulls
print("Check for Nulls:")
print(train.isnull().sum())
print(test.isnull().sum())

#Check what items are outliers
print("*** Item outlier ***")
for i in range(1,len(train_grouped_i)):
    if train_grouped_i.iloc[i,1] >=25000: 
        print(train_grouped_i.iloc[i,0] , " -> " , items.iloc[i,0])

#Check what stores were outliers
print("*** Biggest shop ***")
for i in range(1,len(train_grouped_s)):
    if train_grouped_s.iloc[i,1] >=250000: 
        print(train_grouped_s.iloc[i,0] , " -> " , shops.iloc[i,0])
![image](https://user-images.githubusercontent.com/73217181/121851281-82adae00-cd20-11eb-97bc-b803fc0b0127.png)
![image](https://user-images.githubusercontent.com/73217181/121851301-86d9cb80-cd20-11eb-90b5-a9b7564c2937.png)

