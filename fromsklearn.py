# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder #str to num
# from sklearn.impute import SimpleImputer #null
# from sklearn.metrics import mean_squared_error, r2_score
# import pandas as pd
# import numpy as np
# f=pd.read_csv("cutoff.csv")    # min 5000 rows, max 10,000 rows
# p1=LabelEncoder()   # top_colleges.csv==college_name
# f["cast"]=p1.fit_transform(f["cast"])   # cutoff.csv==college_code,cast,cutoff,year
# p2=SimpleImputer(strategy="mean")
# f[f.select_dtypes(include='number').columns]=p2.fit_transform(f.select_dtypes(include='number'))
# x=f.drop(["college_code","year"],axis=1)
# y=f["college_code"]
# xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=0)
# model=LinearRegression()
# model.fit(xtrain,ytrain)
# ypre=model.predict(xtest)
# cast=input("enter cast:")
# cast=p1.transform([cast])[0]
# maths=float(input("enter maths marks:"))
# phy=float(input("enter phy marks:"))
# chem=float(input("enter chem marks:"))
# cutoff=maths+(phy/2)+(chem/2)
# print("cutoff=",cutoff)
# df1=pd.DataFrame([[cast,cutoff]],columns=['cast','cutoff'])
# df2=pd.DataFrame([[cast,(cutoff-1)]],columns=['cast','cutoff'])
# df3=pd.DataFrame([[cast,(cutoff-2)]],columns=['cast','cutoff'])
# df4=pd.DataFrame([[cast,(cutoff-3)]],columns=['cast','cutoff'])
# input_df=pd.concat([df1,df2,df3,df4],ignore_index=True)
# pre=model.predict(input_df)
# rank=pd.read_csv("top_colleges.csv")  
# sorted_indices=rank["college_code"].tolist()
# predicted_list=(np.argsort(pre)[::-1]).tolist()
# predicted=[int(x) for x in predicted_list]
# predic=[x for x in predicted if x in sorted_indices]
# sorted=sorted(predic,key=lambda x:sorted_indices.index(x))
# for code in sorted:
#     match=rank[rank["college_code"]==code]
#     if not match.empty:
#         print(f"{code}:{match.iloc[0]['college_name']}")
# # for i, prob in enumerate(pre):
# #     print(f"{model.classes_[i]} : {prob:.4f}")
# ypre_rounded = np.round(ypre).astype(float)
# print("MSE:", mean_squared_error(ytest, ypre))   # 
# print("R2 score:", r2_score(ytest, ypre)*100)   # 
# print("test score:",model.score(xtest,ytest)*100)   # 
# print("train score:",model.score(xtrain,ytrain)*100)    # 







from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder #str to num
import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer as st
import faiss
import pandas as pd
from ollama import Client
def get_tier(caste, cutoff):
    caste = caste.upper()
    
    tiers = {
        'OC':   [190, 180, 170, 150],
        'BC':   [188, 178, 165, 145],
        'BCM':  [185, 175, 160, 140],
        'MBC':  [186, 175, 160, 140],
        'SC':   [180, 170, 155, 130],
        'SCA':  [177, 167, 150, 125],
        'ST':   [175, 165, 150, 125]
    }

    limits = tiers.get(caste, [190, 180, 170, 150])  # Default to OC

    if cutoff >= limits[0]:
        return 1
    elif cutoff >= limits[1]:
        return 2
    elif cutoff >= limits[2]:
        return 3
    elif cutoff >= limits[3]:
        return 4
    else:
        return 5


message=input("enter str:")
if any(char.isdigit() for char in message):
    cast_match = re.search(r"\b(oc|bc|bcm|mbc|sc|sca|st)\b",message,re.IGNORECASE)
    maths_match=re.search(r"math[s]?\s*(\d{1,3})",message)
    phy_match=re.search(r"phy[sics]*\s*(\d{1,3})",message)
    chem_match=re.search(r"chem[istry]*\s*(\d{1,3})",message)
    year_match=re.search(r"year\s*(\d{4})",message)
    f=pd.read_csv("cutoff_clg.csv")    # min 5000 rows, max 10,000 rows
    p1=LabelEncoder()   # top_colleges.csv==college_name
    f["cast"]=p1.fit_transform(f["cast"])   # cutoff.csv==college_code,cast,cutoff,year
    x=f.drop(["college_code","year"],axis=1)
    y=f["college_code"]
    xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.1,random_state=0)
    model=RandomForestClassifier(n_estimators=100,random_state=42)
    model.fit(xtrain,ytrain)
    ypre=model.predict(xtest)
    cast=cast_match.group(1).upper()
    maths=float(maths_match.group(1))
    phy=float(phy_match.group(1))
    chem=float(chem_match.group(1))
    cast_encoded=p1.transform([cast])[0]
    cutoff=maths+(phy/2)+(chem/2)
    tier=get_tier(cast,cutoff)
    print("cutoff:",cutoff)
    input_df=pd.DataFrame([[cast_encoded,cutoff,tier]],columns=["cast","cutoff","tier"])
    probas=model.predict_proba(input_df)[0]
    college_proba_pairs=list(zip(model.classes_,probas))
    non_zero_pairs=[(code,prob)for code,prob in college_proba_pairs if prob>0]
    rank=pd.read_csv("top_colleges.csv")   
    ranked_list=rank["college_code"].tolist()
    college_code_mode,mode_prob=max(non_zero_pairs,key=lambda x:x[1])
    start_idx=ranked_list.index(int(college_code_mode))
    final_college_codes=ranked_list[start_idx:]
    colleges=rank[rank["college_code"].isin(final_college_codes)]
    new_row={"college_code":int(college_code_mode),"cast":cast,"cutoff":cutoff,"year":int(year_match.group(1)),"tier":tier}
    cutoff_df=pd.read_csv("cutoff_clg.csv")
    cutoff_df=pd.concat([cutoff_df,pd.DataFrame([new_row])],ignore_index=True)
    cutoff_df.to_csv("cutoff_clg.csv",index=False)
    pd.set_option('display.max_colwidth',None)  
    pd.set_option('display.max_rows',None)      
    print(colleges[["college_code","college_name"]])
else:
    df=pd.read_csv("data.csv")
    questions=df["question"].tolist()
    ans=df["answer"].tolist()
    model=st("all-MiniLM-L6-v2")
    indexed_data=faiss.IndexFlatL2(384)
    indexed_data.add(model.encode(questions))
    d,i=indexed_data.search(model.encode([message]),k=1)
    print(ans[i[0][0]])
    client=Client()
    response=client.chat(model='llama3',messages=[
        {"role":"system","content":f"You are a helpful assistant.Use this context to help the user:{ans[i[0][0]]}"},
        {"role":"user","content":message[0]}
    ])
    print(response['message']['content'])