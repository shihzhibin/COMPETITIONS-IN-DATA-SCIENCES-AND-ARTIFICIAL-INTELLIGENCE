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
確認商店 月份  品項受出狀況
![image](https://user-images.githubusercontent.com/73217181/121851705-04054080-cd21-11eb-9733-b3183e52e1f1.png)

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
確認為合理的分布  我們留下outlier

訓練集及 測試集商店  ID![image](https://user-images.githubusercontent.com/73217181/121851377-9c4ef580-cd20-11eb-9508-a099564830f5.png)
訓練集中的商店ID有 60 个 
測試集中的商店ID有 42 个
(1)刪除未使用的商店
(2)合併相同的商店
#drop Shop that are not in the test set
#[0, 1, 32, 33, 8, 9, 40, 11, 43, 13, 17, 51, 20, 54, 23, 27, 29, 30]
stg_dsi = train_grouped_d_s.copy()
for i in dif1_a:
    stg_dsi.drop(stg_dsi[stg_dsi['shop_id'] == i].index, inplace = True)
    

![image](https://user-images.githubusercontent.com/73217181/121851457-bd174b00-cd20-11eb-83a4-f5edbc6d03a5.png)
#solds every shops
![image](https://user-images.githubusercontent.com/73217181/121851590-e506ae80-cd20-11eb-9765-6a724a6ad3f5.png)

using  Light GBM
以下為GBM 參數
![image](https://user-images.githubusercontent.com/73217181/121851811-2dbe6780-cd21-11eb-8a0b-f6a6018c3fb5.png)

得出RMSE
![image](https://user-images.githubusercontent.com/73217181/121851835-357e0c00-cd21-11eb-9017-fd89ba080a55.png)


