#!/usr/bin/env python
# coding: utf-8

# ### A*

# In[47]:


op=set()
cl=set()
d={}
p={}
graph = {
    'A': [('C', 12), ('D', 6)],
    'B': [('C', 10),('E',7)],
    'C': [('G', 5)],
    'D': [('G', 16)],
    'E': [('C',2)],
    'S': [('A', 4),('B',3)]    
}
def h(n):
    H_dist = {
            'A': 12,
            'B': 11,
            'C': 4,
            'D': 11,
            'E': 6,
            'S': 14,
            'G': 0,
            }
    return H_dist[n]
def neigh(n):
    if n in graph:
        return graph[n]
    else:
        return None


# In[48]:


def astar(start,stop):
    op=set(start)
    d[start]=0
    p[start]=start
    while len(op)>0:
        n=None
        for i in op:
            if n==None or d[i]+h(i)<d[n]+h(n):
                n=i
        if n==stop or graph[n]==None:
            pass
        else:
            for (m,w) in neigh(n):
                if m not in op and m not in cl:
                    op.add(m)
                    p[m]=n
                    d[m]=d[n]+w
                else:
                    if d[m]>d[n]+w:
                        d[m]=d[n]+w
                        p[m]=n
                        if m in cl:
                            cl.remove(m)
                            op.add(m)
        if n==None:
            print('path does not exist')
            return None
        if n==stop:
            path=[]
            while p[n]!=n:
                path.append(n)
                n=p[n]
            path.append(start)
            path.reverse()
            print('path found {}'.format(path))
            return path
        op.remove(n)
        cl.add(n)
    print('path does not exist ')
    return None


# In[49]:


astar('S','G')


# In[46]:


graph={'S':[('A',1),('G',12)],
      'A':[('B',3),('C',1)],
      'B':[('D',3)],
      'C':[('D',1),('G',2)],
      'D':[('G',3)],
      'G':None}
heu = {'S':4,'A':2,'B':6,'C':2,'D':3,'G':0}


# ### AO*

# In[1]:


class Graph:
    def __init__(self,graph,heu,sn):
        self.graph = graph
        self.h=heu
        self.start=sn
        self.par={}
        self.sta={}
        self.sg={}
    def applyaostar(self):
        self.aostar(self.start,False)
    def getneigh(self,v):
        return self.graph.get(v,'')
    def getsta(self,v):
        return self.sta.get(v,0)
    def setsta(self,v,val):
        self.sta[v]=val
    def gethval(self,n):
        return self.h.get(n,0)
    def sethval(self,n,val):
        self.h[n]=val
    def printsol(self):
        print("for graph soln traverse the graph",self.start)
        print("------")
        print(self.sg)
        print("------------------")
    def computemincost(self,v):
        mincost=0
        costtochild={}
        costtochild[mincost]=[]
        flag=True
        for nodeinfo in self.getneigh(v):
            cost=0
            nodelist=[]
            for (c,weight) in nodeinfo:
                cost+=self.gethval(c)+weight
                nodelist.append(c)
            if flag==True:
                mincost=cost
                costtochild[mincost]=nodelist
                flag=False
            else:
                if mincost>cost:
                    mincost=cost
                    costtochild[mincost]=nodelist
        return mincost,costtochild[mincost]
    def aostar(self,v,backtracking):
        print("heuristic value",self.h)
        print("solution graph",self.sg)
        print("processing node:",v)
        print("-----------------")
        if self.getsta(v)>=0:
            mincost,childnodelist=self.computemincost(v)
            self.sethval(v,mincost)
            self.setsta(v,len(childnodelist))
            solved=True
            for child in childnodelist:
                self.par[child]=v
                if self.getsta(child)!=-1:
                    solved = False
        if solved == True:
            self.setsta(v,-1)
            self.sg[v]=childnodelist
        if v!=self.start:
            self.aostar(self.par[v],True)
        if backtracking == False:
            for child in childnodelist:
                self.setsta(child,0)
                self.aostar(child,False)
h1={'A':1,'B':6,'C':2,'D':12,'E':2,'F':1,'G':5,'H':7,'I':7,'J':1}
graph1={'A':[[('B',1),('C',1)],[('D',1)]],
       }
g1=Graph(graph1,h1,'A')
g1.applyaostar()
g1.printsol()
print("heuristic value",g1.h)
print("solution graph",g1.sg)


# In[4]:


class Graph:
    def __init__(s,g,h,st):
        s.h=h
        s.g=g
        s.st=st
        s.sg={}
        s.p={}
        s.sta={}
    def applyao(s):
        s.aostar(s.st,False)
    def aostar(s,v,back):
        print(s.h)
        print(s.sg)
        print(v)
        if s.getsta(v)>=0:
            mcost,childnode=s.mincost(v)
            s.seth(v,mcost)
            s.setsta(v,len(childnode))
            solved=True
            for i in childnode:
                s.p[i]=v
                if s.getsta(i)!=-1:
                    solved=solved&False
        if solved ==True:
            s.setsta(v,-1)
            s.sg[v]=childnode
        if v!=s.st:
            s.aostar(s.p[v],True)
        if back==False:
            for i in childnode:
                s.setsta(i,0)
                s.aostar(i,False)
    def mincost(s,v):
        mcost=0
        childnode={}
        childnode[mcost]=[]
        flag=True
        for i in s.getneigh(v):
            cost=0
            nodelist=[]
            for c,w in i:
                cost+=s.geth(c)+w
                nodelist.append(c)
            if flag==True:
                mcost=cost
                childnode[mcost]=nodelist
                flag=False
            else:
                if mcost>cost:
                    mcost=cost
                    childnode[mcost]=nodelist
        return mcost,childnode[mcost]
    def printsol(s):
        print(s.st)
        print(s.sg)
    def getneigh(s,v):
        return s.g.get(v,'')
    def getsta(s,v):
        return s.sta.get(v,0)
    def setsta(s,v,val):
        s.sta[v]=val
    def geth(s,v):
        return s.h.get(v,0)
    def seth(s,v,val):
        s.h[v]=val
h1= {'A': 1, 'B': 6, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 5, 'H': 6, 'I': 7, 'J': 1}
graph1={ 'A': [[('B', 1), ('C', 1)], [('D', 1)]],
 'B': [[('G', 1)], [('H', 1)]],
 'C': [[('J', 1)]],
 'D': [[('E', 1), ('F', 1)]],
 'G': [[('I', 1)]]

       }
g1=Graph(graph1,h1,'A')
g1.applyao()
g1.printsol()
print("heuristic value",g1.h)
print("solution graph",g1.sg)
            
    


# In[ ]:


class Graph:
    def __init__(self,graph,heu,sn):
        self.graph=graph
        self.h=heu
        self.start=sn
        self.sg={}
        self.par={}
        self.sta={}
    def computemincost(self,v):
        mincost=0
        costtochild={}
        costtochild[mincost]=[]
        flag=True
        for nodeinfo in self.getneigh(v):
            cost=0
            nodelist=[]
            for (c,w) in nodeinfo:
                cost+=self.gethval(c)+w
                nodelist.append(c)
            if flag==True:
                mincost=cost
                costtochild[mincost]=nodelist
                flag=False
            else:
                if mincost>cost:
                    mincost=cost
                    costtochild[mincost]=nodelist
        return mincost,costtochild[mincost]
    def aostar(self,v,b):
        if self.getsta(v)>0:
            mincost,childnode=self.computemincost(v)
            self.sethval(v,mincost)
            self.setsta(v,len(childnode))
            solved=True
            for node in childnode:
                self.par[child]=v
                if self.getsta(v)!=-1:
                    solved =False
            if solved == True:
                self.setsta(v,-1)
                self.sg[v]=nodelist
            if v!=self.start:
                self.aostar(par[v],True)
            if b==False:
                for node in childnode:
                    self.setsta(node,0)
                    self.aostar(node,False)


# ### CE

# In[34]:


['&']*6


# In[38]:


g=[['&' for i in range(6)]for i in range(6)]
g


# In[31]:


import csv


# In[32]:


file = open('lab2.csv')


# In[33]:


data=list(csv.reader(file))[1:]
data


# In[34]:


con=[]
tar=[]
for i in data:
    con.append(i[:-1])
    tar.append(i[-1])
con,tar


# In[35]:


sp=['0']*len(con[0])
ge=[['?' for i in range(len(sp))]for i in range(len(sp))]
sp,ge


# In[36]:


for i,ins in enumerate(con):
    if tar[i]=='yes':
        for x in range(len(sp)):
            if sp[x]=='0':
                sp[x]=ins[x]
            elif ins[x]!=sp[x]:
                sp[x]='?'
                ge[x][x]='?'
    if tar[i]=='no':
        for x in range(len(sp)):
            if ins[x]!=sp[x]:
                ge[x][x]=sp[x]
            else:
                ge[x][x]='?'


# In[37]:


'''for i in indi:
    ge.remove(['?']*len(sp))'''
g=[]
for i in ge:
    if i != ['?']*len(sp):
        g.append(i)
g


# In[39]:


sp,g


# ### ID3

# In[35]:


import csv
import math
from pprint import pprint
def major_class(data, attributes, target):
    freq = {}
    index = attributes.index(target)
    for t in data:
        if t[index] in freq:
            freq[t[index]] += 1
        else:
            freq[t[index]] = 1
    m = 0
    major = ""
    for key in freq.keys():
        if freq[key] > m:
            m = freq[key]
            major = key
    return major

def entropy(attributes, data, targetAttr):
    freq = {}
    data_entropy = 0.0
    i = 0
    for entry in attributes:
        if targetAttr == entry:
            break
        i += 1
    for entry in data:
        if entry[i] == 'PlayTennis':
            pass
        else:
            if entry[i] in freq:
                freq[entry[i]] += 1.0
            else:
                freq[entry[i]] = 1.0
    for f in freq.values():
        data_entropy += (-f/len(data)) * math.log(f/len(data), 2)
    return data_entropy

def info_gain(data, attributes, targetAttr, attr):
    freq = {}
    subset_entropy = 0.0
    i = attributes.index(attr)
    for entry in data:
        if entry[i] == attr:
            pass
        else:
            if entry[i] in freq:
                freq[entry[i]] += 1.0
            else:
                freq[entry[i]] = 1
    for val in freq.keys():
        p = sum(freq.values())
        val_prob = freq[val] / (p)
        data_subset = [entry for entry in data if entry[i] == val]
        subset_entropy += val_prob * entropy(attributes, data_subset, targetAttr)
    data_subset = [entry for entry in data if entry[0] != 'Outlook']
    Info_gain=entropy(attributes, data_subset, targetAttr) - subset_entropy
    return Info_gain

def attr_choose(data, attributes, target):
    best = attributes[0]
    max_gain = 0
    for attr in attributes:
        if attr != target:
            new_gain = info_gain(data, attributes, target, attr)
            if new_gain > max_gain:
                max_gain = new_gain
                best = attr
    return best

def get_values(data, attributes, attr):
    i = attributes.index(attr)
    values = []
    for entry in data:
        if entry[i] == attr:
            pass
        else:
            if entry[i] not in values:
                values.append(entry[i])
    return values

def get_data(data, attributes, best, val):
    new_data = []
    i = attributes.index(best)
    for entry in data:
        if entry[i] == val:
            new_entry = []
            for j in range(len(entry)):
                if j != i:
                    new_entry.append(entry[j])
            new_data.append(new_entry)
    return new_data

def build_tree(data, attributes, target):
    vals = [record[attributes.index(target)] for record in data]
    default = major_class(data, attributes, target)
    if not data or (len(attributes) - 1) <= 0:
        return default
    elif vals.count(vals[0]) == len(vals):
        return vals[0]
    else:
        best = attr_choose(data, attributes, target)
        tree = {best: {}}
        for val in get_values(data, attributes, best):
            new_data = get_data(data, attributes, best, val)
            new_attr = attributes[:]
            new_attr.remove(best)
            subtree = build_tree(new_data, new_attr, target)
            tree[best][val] = subtree
    return tree

def test(attributes, instance, tree):
    attribute = next(iter(tree))
    i = attributes.index(attribute)
    if instance[i] in tree[attribute].keys():
        result = tree[attribute][instance[i]]
        if isinstance(result, dict):
            return test(attributes, instance, result)
        else:
            return result
    else:
        return 'NULL'
    
def execute_decision_tree():
    data = []
    with open('PlayTennis.csv') as tsv:
        for line in csv.reader(tsv):
            data.append(tuple(line))
    attributes = list(data[0])
    target = attributes[-1]
    training_set = [x for i, x in enumerate(data)]
    print("DATA SET IS:")
    #pprint(training_set)
    print()

    tree = build_tree(training_set, attributes, target)

    print('Decision Tree is as below: \n')
    pprint(tree)
    instance = ['Rainy','Hot','Normal','TRUE']

    print("***************")
    print('Testing instance is: ', instance)
    result = test(attributes, instance, tree)
    print('The Target value for the testing instance is: ')
    pprint(result)

execute_decision_tree()


# 

# ### LWR

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
xtrain=np.array(list(range(3,35))).reshape(32,1)
ytrain=np.sin(xtrain)+xtrain**0.75
xtest=np.array([i/10 for i in range(400)]).reshape(400,1)
ytest=[]
#plt.plot(xtrain.squeeze(),ytrain,'o')
#plt.plot(xtest.squeeze(),ytest,'-') 
for r in range(len(xtest)):
    w=np.diag(np.exp(-np.sum((xtrain-xtest[r])**2,1)/(2*0.5**2)))
    #print(xtrain)
    f1=np.linalg.inv(xtrain.T.dot(w).dot(xtrain))
    params=f1.dot(xtrain.T).dot(w).dot(ytrain)
    pred=xtest[r].dot(params)
    ytest.append(pred)
plt.plot(xtrain.squeeze(),ytrain,'ro')
plt.plot(xtest.squeeze(),ytest,'b-') 


# ### KNN

# In[10]:


from sklearn.datasets import load_iris 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
iris_dataset=load_iris() 
X_train, X_test, y_train, y_test = train_test_split(iris_dataset["data"], iris_dataset["target"], random_state=0) 
kn = KNeighborsClassifier()
kn.fit(X_train, y_train)
prediction = kn.predict(X_test)
import sklearn.metrics as sm
print('ACCURACY of KNN:',sm.accuracy_score(y_test,prediction))
print('Confusion Matrix for KNN:\n',sm.confusion_matrix(y_test,prediction))   
plt.plot(X_test,y_test,'ro')        
plt.plot(X_test,prediction,'b+')    
print("Classification Results are:\n")
for i in range(len(X_test)):
    print("Sample:", str(X_test[i]), " Actual label:", str(y_test[i])," Predicted label:", str(prediction[i]))


# ### K-means vs EM

# In[12]:


import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd 
import numpy as np

np.random.seed(2)       #the start number of the random number generator with 2.
iris=load_iris() 
x=pd.DataFrame(iris.data)
y=pd.DataFrame(iris.target)
colormap=np.array(['red','blue','green'])
print(x.head())
from sklearn.cluster import KMeans 
kmeans=KMeans(n_clusters=3).fit(x) 
plt.subplot(1,2,2) 
plt.title("KMeans") 
plt.scatter(x[2],x[3],c=colormap[kmeans.labels_])
KM_Cluster=kmeans.predict(x)
print(KM_Cluster)
print(y)
#print('x2        ',x[2])
import sklearn.metrics as sm 
print('K Means Accuracy:',sm.accuracy_score(y,KM_Cluster)) 
print('Confusion Matrix for KMeans:\n',sm.confusion_matrix(y,KM_Cluster))

from sklearn.mixture import GaussianMixture     #if it didn't work, replace GaussianMixture with GMM
gm=GaussianMixture(n_components=3).fit(x) 
ycluster=gm.predict(x) 

plt.subplot(1,2,1) 
plt.title("EM") 
plt.scatter(x[2],x[3],c=colormap[ycluster]) 
print('EM Accuracy:',sm.accuracy_score(y,ycluster)) 
print('Confusion Matrix for EM:\n',sm.confusion_matrix(y,ycluster))


# ### NCB

# In[4]:


# load the daibetis dataset
import csv
file=open('Diabetis_data.csv')
data=list(csv.reader(file))
X=[]
y=[]
# store the feature matrix (X) and response vector (y)
for row in data:
    X.append(row[:-1])
    y.append(row[-1])
print(len(X))
print(len(y))

# splitting X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

print("Number of training Instances:",len(X_train))
print("Number of testing Instances:",len(y_test))

# training the model on training set
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# making predictions on the testing set
y_pred = gnb.predict(X_test)

# comparing actual response values (y_test) with predicted response values (y_pred)
from sklearn import metrics
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)
#print(metrics.classification_report(y_test, y_pred))
metrics.confusion_matrix(y_test, y_pred)


# ### BP

# In[2]:


from random import random,seed,randint
from pprint import pprint
def initialize(n_inputs,n_hidden,n_output):
    network=[]
    hidden_layer=[{'w':[random() for i in range(n_inputs+1)]} for i in range(n_hidden)] #Initialize Weights and Biases for hidden layers
    network.append(hidden_layer)
    output_layer=[{'w':[random() for i in range(n_hidden+1)]} for i in range(n_output)] #Initialize Weights and Biases for output layer
    network.append(output_layer)
    return network
def activate(w,i):
    activation=w[-1] #Bias value is -1
    for x in range(len(w)-1):
        activation+=w[x]*i[x] #WX similar to WiXi + Bias ie activation=activation + Wixi
        return activation #WX+B
from math import exp
def sigmoid(a):
    return 1/(1+exp(-a))
def forward_prop(network,row):
    inputs=row
    for layer in network:
        new_inputs=[]
        for neuron in layer:
            activation=activate(neuron['w'],inputs) #Compute Activations
            neuron['output']=sigmoid(activation) #Compute Sigmoid
            new_inputs.append(neuron['output']) #Adds it to the output layer
        inputs=new_inputs #new_inputs values now becomes the input
    return inputs
def sigmoid_derivative(output):
    return output * (1-output) #Derivative of 1/(1+e^-x)
def backprop(network,expected): #expected is our expected output value we'd use to compute the error
    for i in reversed(range(len(network))): #Prints the list ie "Network" in reversed order
        layer=network[i] #network contains what? see below
        errors=[] #initialize error values to an empty list
        if i!=len(network)-1: #Output Layer
            for j in range(len(layer)):
                error=0 #Assign error values to 0
                for neuron in network[i+1]:
                    error+=(neuron['w'][j]*neuron['delta'])#Calculates and Updates the error
                errors.append(error)
        else:
                for j in range(len(layer)):
                    neuron=layer[j]
                    errors.append(expected[j]-neuron['output']) #Calculates and appends the errors
        for j in range(len(layer)):
                neuron=layer[j]
                neuron['delta']=errors[j]*sigmoid_derivative(neuron['output']) #Compute Gradients
def update_weights(network,row,lrate): #Gradient Descent
    for i in range(len(network)):
        inputs=row[:-1] #Takes all except last row 
        if i!=0: 
            inputs=[neuron['output'] for neuron in network[i-1]]
            for neuron in network[i]:
                for j in range(len(inputs)):
                    neuron['w'][j]+=lrate*neuron['delta']*inputs[j] #Weights update similar to w5 + n*Edy*xi
                    neuron['w'][-1]+=lrate*neuron['delta'] #Bias is -1 and its updated
def train_network(network,train,lrate,epochs,n_output):
    for epoch in range(epochs):
        sum_err=0
        for row in train:
            outputs=forward_prop(network,row)
            expected=[0 for i in range(n_output)]
            expected[row[-1]]=1
            sum_err+=sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])#Computes the error
            backprop(network,expected)#Calls backpropagation
            update_weights(network,row,lrate)#Finally weights are updated
        print('epoch=%d, lrate=%.3f,error=%.3f'%(epoch,lrate,sum_err))
seed(1)
data=[[2.7810836,2.550537003,0],
      [1.465489372,2.362125076,0],
      [3.396561688,4.400293529,0],
      [1.38807019,1.850220317,0],
      [3.06407232,3.005305973,0],
      [7.627531214,2.759262235,1],
      [5.332441248,2.088626775,1],
      [6.922596716,1.77106367,1],
      [8.675418651,-0.242068655,1],
      [7.673756466,3.508563011,1]] #This dataset contain 2 input inits and 1 output unit
n_inputs=len(data[0])-1
n_outputs=len(set(row[-1] for row in data))
network=initialize(n_inputs,2,n_outputs)
pprint(network)
train_network(network,data,0.5,20,n_outputs)
for layer in network:
    pprint(layer)


# In[4]:


import pandas as pd
import math
from collections import Counter
from pprint import pprint
def entropy(prbs):
    return sum([-prb*math.log(prb,2) for prb in prbs])
def entropylist(alist):
    cnt = Counter(x for x in alist)
    num = len(alist)*1.0
    prbs = [x/num for x in cnt.values()]
    return entropy(prbs)
def infogain(df,split,target):
    dfsplit = df.groupby(split)
    num = len(df.index)*1.0
    oldentropy = entropylist(df[target])
    dfagg=dfsplit.agg({target:[entropylist,lambda x:len(x)/num]})
    dfagg.columns=['entropy','observed']
    newentropy = sum(dfagg['entropy']*dfagg['observed'])
    return oldentropy-newentropy
def id3(df,target,attrname,defaultclass = None):
    cnt = Counter(x for x in df[target])
    if len(cnt) == 1:
        return next(iter(cnt))
    elif df.empty or (not attrname):
        return defaultclass
    else:
        defaultclass = max(cnt.keys())
        gain = [infogain(df,attr,target) for attr in attrname]
        idx = gain.index(max(gain))
        best = attrname[idx]
        tree = {best:{}}
        rem = [x for x in attrname if x!=best]
        for val,data in df.groupby(best):
            subtree = id3(data,target,rem,defaultclass)
            tree[best][val]=subtree
        return tree
def test(df,inst,tree):
    att = next(iter(tree))
    i = df.index(att)
    if inst[i] in tree[att].keys():
        res=tree[att][inst[i]]
        if isinstance(res,dict):
            return test(df,inst,res)
        else:
            return res
    else:
        return 'NULL'
df=pd.read_csv('Playgolf_data.csv')
print(df)
attrname=list(df.columns)
attrname.remove('PlayGolf')
tree = id3(df,'PlayGolf',attrname)
pprint(tree)


# In[ ]:




