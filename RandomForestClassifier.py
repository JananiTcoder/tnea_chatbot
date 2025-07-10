
from flask import Flask, request, jsonify, render_template
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder #str to num
import re,pandas as pd
from sentence_transformers import SentenceTransformer as st
import faiss
from ollama import Client

import firebase_admin
from firebase_admin import auth,credentials
cred=credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred)
import random, smtplib, requests
details={}
em=[' ']

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
EMAIL_ADDRESS=" @gmail.com"
EMAIL_PASSWORD=" "  # not your normal Gmail password!

# from pymongo import MongoClient
# client = MongoClient("mongodb+srv://2023cs0309:<your_password>@chatbot-tnea.xxxxx.mongodb.net/")
# chatdb = client['tnea-chatbot']  # your DB name
# chatlogs = chatdb['chatlogs']  # your collection name

def get_tier(caste,cutoff):
    caste=caste.upper()
    tiers={'OC':[190,180,170,150],'BC':[188,178,165,145],'BCM':[185,175,160,140],'MBC':[186,175,160,140],'SC':[180,170,155,130],'SCA':[177,167,150,125],'ST':[175,165,150,125]}
    limits=tiers.get(caste,[190,180,170,150]) 
    if cutoff>=limits[0]:return 1
    elif cutoff>=limits[1]:return 2
    elif cutoff>=limits[2]:return 3
    elif cutoff>=limits[3]:return 4
    else:return 5
app=Flask(__name__)

@app.route('/frontend', methods=['GET'])
def show_frontend():
    return render_template('frontend.html')  
@app.route("/ask",methods=["POST"])
def ask():
    data=request.get_json() 
    message=data.get("message", "").lower()
    if any(char.isdigit() for char in message):
        cast_match = re.search(r"\b(oc|bc|bcm|mbc|sc|sca|st)\b",message,re.IGNORECASE)
        maths_match=re.search(r"math[s]?\s*(\d{1,3})",message)
        phy_match=re.search(r"phy[sics]*\s*(\d{1,3})",message)
        chem_match=re.search(r"chem[istry]*\s*(\d{1,3})",message)
        year_match=re.search(r"year\s*(\d{4})",message)
        f=pd.read_csv("cutoff_clg.csv")   
        p1=LabelEncoder()   
        f["cast"]=p1.fit_transform(f["cast"])   
        x=f.drop(["college_code","year"],axis=1)
        y=f["college_code"]
        xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.1,random_state=0)
        model=RandomForestClassifier(n_estimators=100,random_state=42)
        model.fit(xtrain,ytrain)
        ypre=model.predict(xtest)
        rank=pd.read_csv("top_colleges.csv")
        ranked_list=rank["college_code"].tolist()
        if not(cast_match):
            return jsonify({"error":"Please include caste"})
        if not(maths_match):
            return jsonify({"error":"Please include marks for maths"})
        if not(phy_match):
            return jsonify({"error":"Please include marks for physics"})
        if not(chem_match):
            return jsonify({"error":"Please include marks for chemistry"})
        if not(year_match):
            return jsonify({"error":"Please include years"})
        cast=cast_match.group(1).upper()
        maths=float(maths_match.group(1))
        phy=float(phy_match.group(1))
        chem=float(chem_match.group(1))
        cast_encoded=p1.transform([cast])[0]
        cutoff=maths+(phy/2)+(chem/2)
        tier=get_tier(cast,cutoff)
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
        return jsonify({
            "message": f"Based on your cutoff of {cutoff:.2f}, here is a list:",
            "colleges": colleges[["college_code","college_name"]].to_dict(orient="records")
        })
    else:
        df=pd.read_csv("data.csv")
        questions=df["question"].tolist()
        ans=df["answer"].tolist()
        model=st("all-MiniLM-L6-v2")
        indexed_data=faiss.IndexFlatL2(384)
        indexed_data.add(model.encode(questions))
        d,i=indexed_data.search(model.encode([message]),k=1)
        client=Client()
        response=client.chat(model='llama3',messages=[
            {"role":"system","content":f"You are a helpful assistant.Use this context to help the user:{ans[i[0][0]]}"},
            {"role":"user","content":message}
        ])
        # chatlogs.insert_one({
        #     "question":message,
        #     "context_used":ans[i[0][0]],
        #     "response":response['message']['content']
        # })
        return jsonify({
            "message": response['message']['content'],
            "colleges": []  
        })    
    
@app.route('/signup', methods=['GET'])
def show_signup():
    return render_template('signup.html')
@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    otp_input = data.get('otp')  # This is safe. Will be None on first step.

    # First Step: No OTP yet => Send OTP
    if not otp_input:
        try:
            if not isinstance(password, str) or len(password) < 6:
                return jsonify({"error": "Password must be at least 6 characters."})

            otp = str(random.randint(100000, 999999))
            details[email] = {"password": password, "otp": otp}  # Temporarily save

            # Send OTP via email
            subject = "Verify your email for TNEA Chatbot"
            body = f"Hi,\n\nYour OTP is:\n\n{otp}\n\nThank you!"
            msg = MIMEMultipart()
            msg["From"] = EMAIL_ADDRESS
            msg["To"] = email
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
                server.sendmail(EMAIL_ADDRESS, email, msg.as_string())

            return jsonify({"message": "OTP sent"})
        except Exception as e:
            return jsonify({"error": str(e)})

    # Second Step: Verify OTP
    if email in details and otp_input == details[email]["otp"]:
        try:
            if not isinstance(password, str) or len(password) < 6:
                return jsonify({"error": "Password must be at least 6 characters."})

            auth.create_user(email=email, password=password)
            return jsonify({"message": "User created"})
        except Exception as e:
            return jsonify({"error": str(e)})

    return jsonify({"error": "Invalid OTP or email"})


    



@app.route('/login', methods=['GET'])
def show_login():
    return render_template('login.html')
@app.route('/login',methods=['POST'])
def login():
    data=request.get_json()
    email=data['email']
    password=data['password']
    payload={
        "email":email,
        "password":password,
        "returnSecureToken":True
    }
    API_KEY=" "
    url=f" "
    response=requests.post(url,json=payload)
    if response.status_code==200:
        res=response.json()
        return jsonify({"message": "Login successful", "uid": res["localId"]})
    else:
        return jsonify({"error": "Login failed. Check email or password."})

# @app.route('/history',methods=['GET'])
# def history():
#     logs=chatlogs.find().sort('_id',-1).limit(20) 
#     history=[]
#     for log in logs:
#         history.append({
#             "question":log.get("question"),
#             "response":log.get("response")
#         })
#     return jsonify(history)

@app.route("/")
def home():
    return render_template("frontpg.html")  

if __name__ == "__main__":
    app.run(debug=True)