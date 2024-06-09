from pymongo.mongo_client import MongoClient
from langchain.tools import tool
from pydantic import BaseModel

uri = "mongodb+srv://riturajdutta400:Y12pcTpIgpfWcN17@cluster0.wafyl7x.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

def get_required_files(source_hcm:str, target_hcm:str):

    client = MongoClient(uri)

    db = client['adp']
    collection = db['manthan']

    query = {"sourceHcm": source_hcm, "targetHcm": target_hcm}
    documents = collection.find(query)

    required_files = []
    for doc in documents:
        required_files.append(doc['reqFiles'])

    if not required_files:
        return f"No required files found for sourceHcm {source_hcm} and targetHcm {target_hcm}."

    result = ", ".join([file for sublist in required_files for file in sublist])
    return result

