from flask import Flask, render_template, request, redirect
import sys
import os
from pymongo import MongoClient

def connect():
# Substitute the 5 pieces of information you got when creating
# the Mongo DB Database (underlined in red in the screenshots)
# Obviously, do not store your password as plaintext in practice
    client = MongoClient("localhost", 27017)
    #handle = connection["data"]
    #handle.authenticate("robert", "fckgw")
    #return handle
    db = client['data']
    return db
    #client.the_database.authenticate('robert', 'fckgw', source='data', mechanism='SCRAM-SHA-1')

app = Flask(__name__)
handle = connect()

# Bind our index page to both www.domain.com
@app.route("/index", methods=['GET'])
@app.route("/", methods=['GET'])
def index():
    userinputs = [x for x in handle.mycollection.find()]
    return render_template('index.html', userinputs=userinputs)

@app.route("/write", methods=['POST'])
def write():
    userinput = request.form.get("userinput")
    oid = handle.mycollection.insert({"message":userinput})
    return redirect ("/")

@app.route("/deleteall", methods=['GET'])
def deleteall():
    handle.mycollection.remove()
    return redirect ("/")

# Remove the "debug=True" for production
if __name__ == '__main__':
    # Bind to PORT if defined, otherwise default to 5000.
    #port = int(os.environ.get('PORT', 8080))

    app.run(host='0.0.0.0', debug=True)#port=port, debug=True)

