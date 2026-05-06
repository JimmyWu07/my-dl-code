import torch

#todo:1.定义函数，创建线性张量
def dm01():
    #创建指定范围线性张量
    t1 = torch.arange(0,10,2)
    print(f"t1:{t1},dtype = {type(t1)}")
    print('-'*30)

    t2 = torch.linspace(1,10,6)  #生成6个数 形成等差数列
    print(f"t2:{t2},dtype = {type(t2)}")

#todo:2.定义函数，演示随机张量
def dm02():
    #step1:设置随机种子。
    #torch.initial_seed()  #默认采用当前系统的时间戳作为随机种子
    torch.manual_seed(11)#设置随机种子


    #step2:创建随机张量。
    #场景1：均匀分布的（0，1）随机张量
    t1 = torch.rand(size=(2,3))
    print(f"t1:{t1},dtype = {type(t1)}")
    print('-' * 30)


    #场景2：符合正态分布的随机张量
    t2 = torch.randn(size=(2,3))
    print(f"t2:{t2},dtype = {type(t2)}")
    print('-' * 30)

    #场景3：创建随机整数张量
    t3 = torch.randint(low=1,high=10,size=(3,5))
    print(f"t3:{t3},dtype = {type(t3)}")

#todo:3.测试函数
if __name__ == '__main__':
    #dm01()
    dm02()
