# HW-4

#Load data 
train = pd.read_csv('sales_train.csv')
test = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')
items = pd.read_csv('items.csv')
item_categories = pd.read_csv('item_categories.csv')
shops = pd.read_csv('shops.csv')

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#check nulls

print("Check for Nulls:")
print(train.isnull().sum())
print(test.isnull().sum())

確認商店 月份  品項售出狀況


![image](https://user-images.githubusercontent.com/73217181/121851705-04054080-cd21-11eb-9733-b3183e52e1f1.png)

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Check what items are outliers

print("*** Item outlier ***")
for i in range(1,len(train_grouped_i)):
    if train_grouped_i.iloc[i,1] >=25000: 
        print(train_grouped_i.iloc[i,0] , " -> " , items.iloc[i,0])
        
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Check what stores were outliers

print("*** Biggest shop ***")
for i in range(1,len(train_grouped_s)):
    if train_grouped_s.iloc[i,1] >=250000: 
        print(train_grouped_s.iloc[i,0] , " -> " , shops.iloc[i,0])
        

確認品項

![image](https://user-images.githubusercontent.com/73217181/121905430-ba3a4b80-cd5c-11eb-97ac-f85c18d1bb2e.png)


確認商店


![image](https://user-images.githubusercontent.com/73217181/121905474-c58d7700-cd5c-11eb-9b5e-fc24117ddf12.png)

確認商店31 與 品項20949  彼此間的關係


![image](https://user-images.githubusercontent.com/73217181/121905546-d9d17400-cd5c-11eb-8010-094dc7370d9b.png)



確認為合理的分布  我們留下outlier

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

訓練集及 測試集商店  
訓練集中的商店ID有 60 个 
測試集中的商店ID有 42 个
(1)刪除未使用的商店
(2)合併相同的商店



#drop Shop that are not in the test set
#[0, 1, 32, 33, 8, 9, 40, 11, 43, 13, 17, 51, 20, 54, 23, 27, 29, 30]
stg_dsi = train_grouped_d_s.copy()
for i in dif1_a:
    stg_dsi.drop(stg_dsi[stg_dsi['shop_id'] == i].index, inplace = True)
    
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

確認商店標籤  及 售出狀況
![image](https://user-images.githubusercontent.com/73217181/121851457-bd174b00-cd20-11eb-83a4-f5edbc6d03a5.png)
#solds every shops
![image](https://user-images.githubusercontent.com/73217181/121851590-e506ae80-cd20-11eb-9765-6a724a6ad3f5.png)

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

以品項種類的部分  我們除了放入一些比較常見的資料別



![image](https://user-images.githubusercontent.com/73217181/121904383-b78b2680-cd5b-11eb-8844-0f4b664ba248.png)


這邊比較特別的  有發現  商品依主遊戲、配件、周邊商品等名稱 長度不同  
我們多分出一個名稱長度  的column  



![image](https://user-images.githubusercontent.com/73217181/121904286-a2ae9300-cd5b-11eb-9b4e-c9ac232c8a9c.png)

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

以商店的部分   我們是先看 各個商店 賣出品項的種類有多少 以這張圖呈現



![image](https://user-images.githubusercontent.com/73217181/121904476-ceca1400-cd5b-11eb-9a30-e3565ec9525b.png)



那這邊 特別的是我們也加入一個 名為city 的column  因為 每家商店 都有自己所屬的城市


![image](https://user-images.githubusercontent.com/73217181/121904850-24062580-cd5c-11eb-9239-6d4095665de6.png)



![image](https://user-images.githubusercontent.com/73217181/121904798-1781cd00-cd5c-11eb-9744-d802d0790867.png)


詳情請看  *ppt*

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

using  Light GBM
以下為GBM 參數
![image](https://user-images.githubusercontent.com/73217181/121851811-2dbe6780-cd21-11eb-8a0b-f6a6018c3fb5.png)

得出RMSE
![image](https://user-images.githubusercontent.com/73217181/121851835-357e0c00-cd21-11eb-9017-fd89ba080a55.png)


 report link *ppt*
https://drive.google.com/file/d/1drIItUpGYscIWmAhlbLiIhh4Znayx32N/view?usp=sharing
