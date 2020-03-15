import requests
url = 'http://localhost:5000/'
r = requests.post(url+'predict',json={'exp':1.8,})
print(r.json())
d = {
    "features":[3.,4.,5.,6.],
    "labels": [2000,4000,8000,16000]
}
r = requests.post(url+'fit',json=d)
r = requests.post(url+'predict',json={'exp':4,})
print(r.json())