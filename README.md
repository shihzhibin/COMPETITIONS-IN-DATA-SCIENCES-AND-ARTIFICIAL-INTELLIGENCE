# HW-4  1092_資料科學與人工智慧競技  COMPETITIONS IN DATA SCIENCES AND ARTIFICIAL INTELLIGENCE



![image](https://user-images.githubusercontent.com/73217181/122633732-9b490a00-d10c-11eb-9a39-bcf4c9a3cabc.png)


P96091092施智臏   |   NF6094019洪志宇 
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
流程


![image](https://user-images.githubusercontent.com/73217181/122633303-7358a700-d10a-11eb-829e-6255098a8a90.png)



--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
資料預處理

取出四個特徵值進行繪圖
( X軸 : date_black_num   |   Shop ID   |   Item_ID  ，   Y軸 : item_cnt_day  ) 


![image](https://user-images.githubusercontent.com/73217181/122633368-df3b0f80-d10a-11eb-95f6-2b1c110a28e9.png)


圖1 ： X軸為date_black_num，Y軸為item_cnt_day，可看出在12月的銷    售狀況相較其他來的突出。

圖2 ： X軸為 Shop ID，Y軸為 item_cnt_day，此圖呈現 Item20949在眾多品項中銷售狀況最為突出

圖3 ：X軸為 Item_ID，Y軸為 item_cnt_day，此圖呈現了Shop31 眾多商店中銷售狀況更為突出

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#決定是否剔除異常值


![image](https://user-images.githubusercontent.com/73217181/121851705-04054080-cd21-11eb-9733-b3183e52e1f1.png)


圖. 4  每個店家在每個月的銷售總量圖

＃圖示說明 : 

以「前視圖 （月）」看過去，12月 同為兩年銷售最為突出的月份，且幾乎各個店家皆有此趨勢。

以「右視圖（店家ID）」看過去，可觀察到不同店家隨著月份的銷售總量變化。




![image](https://user-images.githubusercontent.com/73217181/122633397-14476200-d10b-11eb-96cf-481761451985.png)
        
        
        
圖5 ： X軸為date_black_num，Y軸為item_cnt_day，可看出每年的12月份存在季節性，皆有突出的銷售概況。

圖6 ： X軸為 Shop ID，Y軸為 item_cnt_day，呈現了 Item20949 在銷售上存在一定的趨勢性。

圖7 ：X軸為 Item_ID，Y軸為 item_cnt_day，呈現了Shop31 在銷售上存在一定的趨勢性。


在各個圖皆呈現 每年度裡的12月銷售皆為峰值
以及確認商店31 及 品項20949各自及交互作用下的狀態

結論 :  確認為合理的分布 ，決定留下outlier

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Shop  預處理

經整理過後，發現訓練集中的商店ID有 60 個  |  測試集中的商店ID有 42 個 

![image](https://user-images.githubusercontent.com/73217181/122633426-3b059880-d10b-11eb-979d-76fa546fd41e.png)



表2 ： 由於部分店家店名重複，因此視為同一間店家，將銷售總額合併同時刪除未使用的商店。



![image](https://user-images.githubusercontent.com/73217181/122633449-507ac280-d10b-11eb-8352-a80fc7c20f9a.png)

圖. 8   合併店家後的各店家銷售量分佈圖 



新增一項特徵
由於發現商店名稱裡包含了所在城市，因此從 Shop_name裡分離並新增一欄名為City的欄位進行分析，同時也做Encoding的動作。


![image](https://user-images.githubusercontent.com/73217181/122633461-6c7e6400-d10b-11eb-8b66-aec46b6957ea.png)

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Item  預處理

首先，繪製出 (品項種類) 與 各項 (品項) 歸類加總後的長條圖。

![image](https://user-images.githubusercontent.com/73217181/122633477-8029ca80-d10b-11eb-8962-e780bb32adad.png)

圖9 .  商品種類與各項品項歸類加總後的長條圖，其中 Item_category 40 較其他熱銷


新增一項特徵
由於發現商品依主遊戲、配件、周邊商品等名稱長度不同，而這意外地與銷量有著關係，因此多分出一個名稱長度 的column進行分析。

![image](https://user-images.githubusercontent.com/73217181/122633791-fbd84700-d10c-11eb-9706-f62cd16ef32d.png)


--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

確認整理後的資料 並繪製  各間   shop_ID 售出狀況  是否線性
 
![image](https://user-images.githubusercontent.com/73217181/121851457-bd174b00-cd20-11eb-83a4-f5edbc6d03a5.png)


#solds every shops


![image](https://user-images.githubusercontent.com/73217181/121851590-e506ae80-cd20-11eb-9765-6a724a6ad3f5.png)



----------------------------------------------SHOP AND ITEM CATEGORY CLUSTERING-------------------------------
這邊商店和商品類別我們根據其銷售概況 進行cluster

![image](https://user-images.githubusercontent.com/73217181/122633544-bcf5c180-d10b-11eb-90fb-c58c995ebbf6.png)


繪製商店和商品類別的主成分分析和聚類結果，依照它們在 PCA 維度上的分數繪製單個項目，並根據它們的集群分配著色，主成分圖顯示 3 個類別在這方面是異常值。


![image](https://user-images.githubusercontent.com/73217181/122633554-caab4700-d10b-11eb-8861-4b4678f16d88.png)


每間商店按每個商品的總銷售額聚集

![image](https://user-images.githubusercontent.com/73217181/122633563-dac32680-d10b-11eb-8451-6dedbb77ad79.png)



主成分圖顯示商店的主要銷售量不同，商店 31 因其銷售量而成為異常值。商店 12 和 55 是正交維度上的異常值，因為它們與其他店不同，他們銷售的是（網路）商品。


![image](https://user-images.githubusercontent.com/73217181/122633583-ef072380-d10b-11eb-838f-0870ce096080.png)



--------------------------------------------------------------------------------#時間序列分析#----------------------------------------------

![image](https://user-images.githubusercontent.com/73217181/122633619-12ca6980-d10c-11eb-9719-f4c4e5d984c8.png)


選擇將 商品每月銷量、商品總銷量、商品每月平均價格、商品ID分類每月平均價格、依城市GROUP BY的商品ID分類每月平均價格、不同商店的不同商品的平均銷售等等參數進行3個月的lag，同時新增欄位以增加特徵。



--------------------------------------------------------------------------------#模型配適#------------------------------------------------------
Linear Regression

![image](https://user-images.githubusercontent.com/73217181/122633646-3beafa00-d10c-11eb-8503-b6caec70806d.png)


如前面提到將新增出來的欄位選進來，並作為分析的特徵，如: city、item_name_lengh、cluster 及  三個月的lag，進行預測得出RMSE 為0.75。

![image](https://user-images.githubusercontent.com/73217181/122633656-473e2580-d10c-11eb-8a2b-24c4273d59ac.png)

 
####    Kaggle分數為1.01174



LightGBM－第一次實驗結果 :



![image](https://user-images.githubusercontent.com/73217181/122633670-57560500-d10c-11eb-93f0-46e035d90b39.png)


參考網路上的參數設定得出RMSE為 0.787355，但或許是參數設置讓他RMSE 變得比較高，於是做了參數上的調整。



LightGBM－第二次實驗結果 :



![image](https://user-images.githubusercontent.com/73217181/122633691-689f1180-d10c-11eb-95bb-a638bd42aa4f.png)


除了增加了一些參數，同時前一個實驗的參數進行微調 :
1.	將leaves增加
2.	增加子帶
3.	學習率保持不變
4.	其他
得出RMSE 為 0.7849   下降了 0.003
####   Kaggle分數為 0.90145





 report link *ppt*
https://drive.google.com/file/d/1vJgIU1UwcYdHh8ehhuLiY1OMxPEgx4HW/view?usp=sharing
