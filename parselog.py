result=open('test1.log').readlines()
allnum=0
rightnum=0
for i in range(len(result)):
   allnum+=1
   if result[i].split()[0]==result[i].split()[1]:
       rightnum+=1
print float(rightnum)/allnum

