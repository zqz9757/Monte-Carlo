#求定积分
import time
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
def f(x):
    return x**2
def n(N):
    start_time=time.time()
    a=1/3.0
    x=np.random.uniform(0,1,N)
    c=f(x)
    s=0
    for i in c:
        s+=i
    j=s*1.00/N
    print('蒙特卡洛估值为：',j)
    print('与真实值之间的误差为：',abs(a-j))
    end_time=time.time()
    result=round(end_time-start_time,3)
    print('耗时为：',result)
n(100)

####蒙特卡洛模拟路径
import matplotlib.pyplot as plt
np.random.seed(3)
def callMonteCarlo(N,S0,T,K,sigma,r):
    start=time.time()
    m = 50
    dt=T/m
    S=np.zeros((m+1,N))
    S[0]=S0
    for t in range(1,m+1):
        z=np.random.normal(0,1,N)
        #z=np.random.randn(N)
        S[t]=S[t-1]*np.exp((r-0.5*sigma**2)*dt+sigma*np.sqrt(dt)*z)
    end=time.time()
    price=np.maximum(S[-1]-K,0)*np.exp(-r*T)
    error=np.std(price)
    plt.figure(figsize=(15,10))
    x=np.linspace(0,1,m+1)
    for i in range(N):
        plt.plot(x,S[:,i])
    plt.show()
    return np.sum(np.maximum(S[-1]-K,0)*np.exp(-r*T))/N, error, end-start

    #z=np.random.rand(N)
    #ST=S0*np.exp((r-0.5*sigma**2)*T+sigma*z*np.sqrt(T))
    #return sum(np.maximum(ST-K,0))*np.exp(-r*T)/N
c=callMonteCarlo(10000,100,1.0,90, 0.3, 0.05)
print(c)


####CIR路径模拟
r0=0.05
sigma=0.2
kappa=0.5
theta=0.05

N=5000
m=100
T=1.0
dt=T/m
r=np.zeros((m+1,N))
r[0]=r0
for t in range(1,m+1):
    z=np.random.standard_normal(N)
    r[t]=r[t-1]+kappa*(theta-r[t-1])*dt+sigma*np.sqrt(r[t-1])*np.sqrt(dt)*z
#plt.figure(figsize=(10,10))
#x=np.linspace(0,1,m+1)
#plt.plot(x,r[:,:20])  #绘制前10条路径
#plt.show()

####Vasicek路径模拟
r1=np.zeros((m+1,N))
r1[0]=r0
for t in range(1,m+1):
    z=np.random.standard_normal(N)
    r1[t]=r1[t-1]+kappa*(theta-r1[t-1])*dt+sigma*np.sqrt(dt)*z
plt.figure(figsize=(10,10))
x=np.linspace(0,1,m+1)
plt.plot(x,r[:,:2])
plt.plot(x,r1[:,:2],color='grey')  #绘制前10条路径
plt.show()

####方差下降

##对偶变量法
import time
def callMonteCarlo_Anti(N,S0,T,K,sigma,r):
    start=time.time()
    m = 50
    dt=T/m
    S1=np.zeros((m+1,N))
    S2=np.zeros((m+1,N))
    S=np.zeros((m+1,N))
    S1[0]=S0
    S2[0]=S0
    S[0]=S0
    for t in range(1,m+1):
        z=np.random.normal(0,1,N)
        #z=np.random.randn(N)
        S1[t]=S1[t-1]*np.exp((r-0.5*sigma**2)*dt+sigma*np.sqrt(dt)*z)
        S2[t]=S2[t-1]*np.exp((r-0.5*sigma**2)*dt-sigma*np.sqrt(dt)*z)
    end=time.time()
    return ((np.sum(np.maximum(S1[-1]-K,0)*np.exp(-r*T))/N)+\
        (np.sum(np.maximum(S2[-1]-K,0)*np.exp(-r*T))/N))/2, end-start
print(callMonteCarlo_Anti(10000, 100, 1.0, 90, 0.3, 0.05))

##控制变量法
#欧式期权

def callMonteCarlo_Control(N,S0,T,K,sigma,r):
    start=time.time()
    m = 50
    dt=T/m
    S=np.zeros((m+1,N))
    S[0]=S0
    for t in range(1,m+1):
        z=np.random.normal(0,1,N)
        #z=np.random.randn(N)
        S[t]=S[t-1]*np.exp((r-0.5*sigma**2)*dt+sigma*np.sqrt(dt)*z)

    price=np.maximum(S[-1]-K,0)*np.exp(-r*T)
    covariance=np.cov(price,S[-1])
    c=-covariance[0,1]/np.var(S[-1])
    ControlPrice=price+c*(S[-1]-np.exp(r*T)*S0)
    error=np.std(ControlPrice)
    end=time.time()
    return np.average(ControlPrice),error,end-start
print(callMonteCarlo_Control(10000, 100, 1.0, 90, 0.3, 0.05))

#亚式期权

def GeoAsianPrice(S0,K,sigma,r,T,is_call):  #算术亚式期权的闭解
    sigma_a = sigma * np.sqrt(T/3)
    q=r+sigma**2/6
    d1=(np.log(S0/K)+q*T/2)/sigma_a
    d2=d1-sigma_a
    Nd1=stats.norm.cdf(d1)
    Nd2=stats.norm.cdf(d2)
    if is_call==True:
        price=S0*np.exp(-q*T/2)*Nd1-K*np.exp(-r*T)*Nd2
    else:
        price=K*np.exp(-r*T)*(1-Nd2)-S0*np.exp(-q*T/2)*(1-Nd1)
    return price
GeoAsianPrice(100,90,0.3,0.05,2.,True)

np.random.seed(3)
def AsianMC(S0, K, T, r, sigma, N,is_call):
    m=20
    dt=T/m
    S=np.zeros((m+3,N))
    S[0]=S0
    for t in range(1,m+1):
        z = np.random.normal(0,1,N)
        S[t]=S[t-1]*np.exp((r-0.5*sigma**2)*dt+sigma*np.sqrt(dt)*z)
    for j in range(N):
        S[-2,j]=np.average(S[1:m+1,j])
        S[-1,j]=stats.gmean(S[1:m+1,j])
        if is_call == True:
            S[-2,j]=np.maximum(S[-2,j]-K,0)
            S[-1,j]=np.maximum(S[-1,j]-K,0)
        else:
            S[-2, j] = np.maximum(K-S[-2,j], 0)
            S[-1, j] = np.maximum(K-S[-1,j], 0)
    Avgpayoff=np.asarray(S[-2])
    Geopayoff=np.asarray(S[-1])
    ariAsian=np.average(S[-2])*np.exp(-r*T)
    geoAsian=np.average(S[-1])*np.exp(-r*T)

    covariance=np.cov(Avgpayoff,Geopayoff)
    Geostdev=np.std(Geopayoff)

    c=-covariance[0,1]/(Geostdev**2)
    Geoprice= GeoAsianPrice(S0,K,sigma,r,T,is_call)
    ControlPrice= Avgpayoff+c*(Geopayoff-Geoprice*np.exp(r*T))
    price=np.average(ControlPrice)*np.exp(-r*T)
    error=np.std(ControlPrice)/np.sqrt(N)
    return Avgpayoff, Geopayoff,ariAsian,geoAsian,price,error,c
AsianMC(100, 90, 2.0, 0.05, 0.3, 50000,True)

[Avgpayoff, Geopayoff,ariAsian,geoAsian,price,error,c]=AsianMC(100, 90, 2.0, 0.05, 0.3, 50000,True)
plt.plot(Avgpayoff, Geopayoff,'ro')
plt.xlabel('arithmetic MC price')
plt.ylabel('geometric MC price')
plt.show()




####重要性抽样

N=10000
y=np.random.normal(8,1,N)
a=[0 if i<=8 else np.exp(-8*i+0.5*8**2) for i in y]
np.mean(a)
from scipy import stats
1-stats.norm.cdf(8)
#欧式期权
def callMonteCarlo_Imp(S0,K,T,r,sigma,N):
    start=time.time()
    m=50
    dt=T/m
    S=np.zeros((m+1,N))
    S[0]=S0
    LR=0
    for t in range(1,m+1):
        z=np.random.normal(r,1,N)
        S[t]=S[t-1]*np.exp((r-0.5*sigma**2)*dt+sigma*np.sqrt(dt)*z)
        LR=LR+z
    price = np.maximum(S[-1] - K, 0) * np.exp(-r * T) * np.exp(-r*LR+0.5*m*r**2)
    end = time.time()
    error = np.std(price)
    return np.average(price),error, end-start
callMonteCarlo_Imp(100, 90, 1.0, 0.05, 0.3, 10000)

####重要性&控制变量
def callMonteCarlo_ImpCont(S0,K,T,r,sigma,N):
    start=time.time()
    m=50
    dt=T/m
    S=np.zeros((m+1,N))
    S[0]=S0
    S1=np.zeros((m+1,N))
    S1[0]=S0
    LR=0
    for t in range(1,m+1):
        z=np.random.normal(r,1,N)
        S1[t]=S1[t-1]*np.exp((r-0.5*sigma**2)*dt+sigma*np.sqrt(dt)*(z-r))
        S[t]=S[t-1]*np.exp((r-0.5*sigma**2)*dt+sigma*np.sqrt(dt)*z)
        LR=LR+z
    price = np.maximum(S[-1] - K, 0) * np.exp(-r * T) * np.exp(-r*LR+0.5*m*r**2)
    covariance=np.cov(price,S1[-1])
    c=-covariance[0,1]/np.var(S1[-1])
    ControlPrice=price-c*(S1[-1]-np.exp(r*T)*S0)
    end = time.time()
    error = np.std(ControlPrice)
    return  np.average(ControlPrice),error, end-start

print(callMonteCarlo(10000,100,1.0,90, 0.3, 0.05))
print(callMonteCarlo_Control(10000, 100, 1.0, 90, 0.3, 0.05))
print(callMonteCarlo_Imp(100, 90, 1.0, 0.05, 0.3, 10000))
print(callMonteCarlo_ImpCont(100,90,1.0,0.05,0.3,10000))




#### 条件期望


#圆周率
N=10000
n=0
x_list=[]
y_list=[]
for i in range(N+1):
    x=np.random.uniform(-1,1,size=1)
    y=np.random.uniform(-1,1,size=1)
    x_list.append(x)
    y_list.append(y)
    if np.add(x**2,y**2)<=1:
        n=n+1
pi=4*n/N

a=0
b=0
r=1
theta=np.arange(0,2*np.pi,0.01)
x=a+r*np.cos(theta)
y=b+r*np.sin(theta)

fig=plt.figure(figsize=(8,8))
axes=fig.add_subplot(111)
axes.plot(x,y)
axes.axis('equal')
plt.scatter(x_list,y_list)

plt.show()
