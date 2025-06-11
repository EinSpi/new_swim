import sys

line_pointer=1
target_sum=[]
coins=[]
next_title=1
coin_types=[]
coin_nums=[]
sub_types=[]
sub_nums=[]
for i,line in enumerate(sys.stdin):
    if i==0:
        exp_num=int(line.split()[0])
    elif i==next_title:
        #title line
        coin_number=int(line.split()[0])
        next_title+=(coin_number+1)
        target_sum.append(int(line.split()[1]))
    elif i==next_title-1:
        sub_types.append(int(line.split()[0]))
        sub_nums.append(int(line.split()[1]))
        coin_types.append(sub_types)
        coin_nums.append(sub_nums)
        sub_types=[]
        sub_nums=[]
    else:
        sub_types.append(int(line.split()[0]))
        
        sub_nums.append(int(line.split()[1]))
    




for i in range(len(target_sum)):
    res=target_sum[i]
    use_coins=0
    while res>0 and sum(coin_nums[i])>0:
        current_largest=0
        #what is the current largest?
        for j, coin in enumerate(coin_types[i]):
            if coin>current_largest and coin_nums[i][j]>0:
                current_largest=coin
                current_largest_index=j
        demand=res//current_largest+1
        budget=coin_nums[i][j]
        res-=min(demand,budget)*current_largest
        coin_nums[i][j]-=min(demand,budget)
        use_coins+=min(demand,budget)
    if res>0:
        print(-1)
    else:
        print(use_coins)