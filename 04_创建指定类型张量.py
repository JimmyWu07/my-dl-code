import torch


# 场景1：直接创建指定类型的张量
t1 = torch.tensor([1,2,3,4,5],dtype=torch.float)
print(f"t1:{t1},(张量类型)：{type(t1)},(元素类型)：{type(t1)}")
print('-'*50)


# 场景2：创建好张量后 -->做类型转换
# 思路1：type()函数，推荐掌握
t2 = t1.type(torch.int16) #记住这个 自定义转换类型torch.int16
print(f"t2：{t2},(元素)类型:{t2.dtype},(张量)类型：{type(t2)}")
print('-'*50)

# 思路2:half()/double()/float()/short(/int()/long()
print(t2.half())
print(t2.double())
print(t2.float())#float,默认
print(t2.short())
print(t2.int())
print(t2.long())#int,默认