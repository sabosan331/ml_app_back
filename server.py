##################################
# front : vue
# back  : flask ← here
# ml    : scikit-learn ← here
# db    : mongodb ← here
##################################

from flask import Flask, jsonify, make_response, request, Response
from flask_cors import CORS # ここを追記
import get_result

app = Flask(__name__)
cors = CORS(app)

@app.route('/get_exp_data/')
def get_exp_data():
    print(exp_data)
    return jsonify(exp_data)

@app.route('/get_exp_result/')
def get_exp_result():

    print(request.args.get('condition')) # 実験条件をrequestとして受け取る
    res = { "tp": int(cm[0,0]),"fp": int(cm[1,0]),
            "fn" : int(cm[0,1]), "tn" : int(cm[1,1] ),
             "acc": acc, "rec" : rec, "prec": prec, "f1":f1} 
    print(res)
    return jsonify(res)

###########################################################################################################

exp_data, cm,acc,rec,prec,f1 = get_result.resnet_test(dataset_id=0,model_id=0)

app.run(port=5000)
