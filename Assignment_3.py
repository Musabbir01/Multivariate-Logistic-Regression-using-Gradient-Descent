
import math
import csv
data = list(csv.reader(open("diabetes2.csv")))
testSet=[]
trainSet=[]

def find_mean(x):
   total=0
   for i in range(1,769):
       total=float(total)+float(data[i][x])
   total=float(total)/float(768)
   return total

def find_list(x):

   l=[]
   for i in range(1,769):
       l.append(float(data[i][x]))
   return l


Glucose_mean=find_mean(0)
Blood_mean=find_mean(1)
Thikness_mean=find_mean(2)
Insulin_mean=find_mean(3)
Bmi_mean=find_mean(4)
DF_mean=find_mean(5)
Age_mean=find_mean(6)

Glucose_list=find_list(0)
Glucose_max=max(Glucose_list)
Glucose_min=min(Glucose_list)

Blood_list=find_list(1)
Blood_max=max(Blood_list)
Blood_min=min(Blood_list)

Thikness_list=find_list(2)
Thikness_max=max(Thikness_list)
Thikness_min=min(Thikness_list)

Insulin_list=find_list(3)
Insulin_max=max(Insulin_list)
Insulin_min=min(Insulin_list)

Bmi_list=find_list(4)
Bmi_max=max(Blood_list)
Bmi_min=min(Blood_list)

Df_list=find_list(5)
DF_max=max(Df_list)
DF_min=min(Df_list)

Age_list=find_list(6)
Age_max=max(Age_list)
Age_min=min(Age_list)

#Mean List
mean_list=[]
mean_list.append(Glucose_mean)
mean_list.append(Blood_mean)
mean_list.append(Thikness_mean)
mean_list.append(Insulin_mean)
mean_list.append(Bmi_mean)
mean_list.append(DF_mean)
mean_list.append(Age_mean)

#Max List
max_list=[]
max_list.append(Glucose_max)
max_list.append(Blood_max)
max_list.append(Thikness_max)
max_list.append(Insulin_max)
max_list.append(Bmi_max)
max_list.append(DF_max)
max_list.append(Age_max)

print(max_list)

#Min List
min_list=[]
min_list.append(Glucose_min)
min_list.append(Blood_min)
min_list.append(Thikness_min)
min_list.append(Insulin_min)
min_list.append(Bmi_min)
min_list.append(DF_min)
min_list.append(Age_min)

print(min_list)



def modify(x):

    each_row=x
    #return each_row[7]
    new_row=[]
    for j in range(0,7):
        num=(float(each_row[j])-float(mean_list[j]))/(float(max_list[j])-float(min_list[j]))
        new_row.append(num)
    #new_row.append(float(each_row[7]))
    return new_row


modified_dataset=[]
print(data[1])

for i in range(1,769):
  #new_each_row = []
  #new_each_row=modify(data[i])
  modified_dataset.append(modify(data[i]))


for i in range(0,537):
    trainSet.append(modified_dataset[i])
for i in range(538,768):
    testSet.append(modified_dataset[i])

random_list=[1,2,8,5,3,2,6,5]
ran_train=[]

for i in range(0,537):
  trainSet[i].insert(0,1)


'''w_x=[]

def w_x_c(l):
   temp=[]
   num=0
   for j in range(0,8):
       num+=float(random_list[j])*float(l[j])
      # temp.append(float(num))
   return num

for i in trainSet:
   p=w_x_c(i)
   w_x.append(p)

print("This is w_x")
print(w_x)

#Finding h_theta(x)
h_theta=[]
num=0
for i in w_x:
    num=float(1)/float(float(1)+float(math.exp(i)))
    h_theta.append(num)'''



#Finding hypothisis value for each row
hypothisis_list=[]
for i in trainSet:
    i_num=float(random_list[0])*float(i[0])+float(random_list[1])*float(i[1])+float(random_list[2])*float(i[2])+float(random_list[3])*float(i[3])+float(random_list[4])*float(i[4])+float(random_list[5])*float(i[5])+float(random_list[6])*float(i[6])+float(random_list[7])*float(i[7])
    f_num=float(1/float(1+math.exp(-i_num)))
    hypothisis_list.append(f_num)
print(len(hypothisis_list))
#y_th list of train set
train_y=[]
test_y=[]
for i in range(1,538):
    train_y.append(int(data[i][7]))
for i in range(539,769):
    test_y.append(int(data[i][7]))


J_theta=0

for i in range(0,537):
    J_theta+=(float(train_y[i])*float(math.log(hypothisis_list[i])))+(float(1-float(train_y[i]))*float(math.log(1-hypothisis_list[i])))

J_theta=float(-1/537)*float(J_theta)
print("Old J_theta")
print(J_theta)
learning_rate=0.5

'''print("This")
print(trainSet[1][60])
print(hypothisis_list[536])
print(train_y[536])
print(len(train_y))
print(len(hypothisis_list))'''



while True:
    new_par_vector=[]
    num=0
    for i in range(0,8):
            for j in range(0,537):
                num+=float(float(hypothisis_list[j])-float(train_y[j]))*float(trainSet[j][i])
            num=float(random_list[i])-float((float(learning_rate)/float(537))*num)
            new_par_vector.append(num)

    hypothisis_list = []
    for i in trainSet:
        i_num = float(new_par_vector[0]) * float(i[0]) + float(new_par_vector[1]) * float(i[1]) + float(
            new_par_vector[2]) * float(i[2]) + float(new_par_vector[3]) * float(i[3]) + float(new_par_vector[4]) * float(
            i[4]) + float(new_par_vector[5]) * float(i[5]) + float(new_par_vector[6]) * float(i[6]) + float(
            new_par_vector[7]) * float(i[7])
        f_num = float(1 / float(1 + math.exp(-i_num)))
        hypothisis_list.append(f_num)

        New_J_theta = 0

    for i in range(0, 537):
            New_J_theta+= (float(train_y[i]) * float(math.log(hypothisis_list[i]))) + (
                        float(1 - float(train_y[i])) * float(math.log(1 - hypothisis_list[i])))

    New_J_theta = float(-1/537) * float(J_theta)
    if(J_theta-New_J_theta<0.05):
        break
    else:
        J_theta=New_J_theta

print("Finish New J_theta")
print(J_theta)
#Now have to work with the test set
print(new_par_vector)
print(len(testSet))

for i in range(0,230):
  testSet[i].insert(0,1)

#hypothesis for testList
test_hypothisis_list=[]
for i in testSet:
    i_num=float(new_par_vector[0])*float(i[0])+float(new_par_vector[1])*float(i[1])+float(new_par_vector[2])*float(i[2])+float(new_par_vector[3])*float(i[3])+float(new_par_vector[4])*float(i[4])+float(new_par_vector[5])*float(i[5])+float(new_par_vector[6])*float(i[6])+float(new_par_vector[7])*float(i[7])
    f_num=float(1/float(1+math.exp(-i_num)))
    test_hypothisis_list.append(f_num)

new_output_list=[]
for i in test_hypothisis_list:
    if i>0.5:
        new_output_list.append(int(1))
    else:
        new_output_list.append(int(0))

num_similar=0
for i in range(0,230):
    if new_output_list[i]==test_y[i]:
        num_similar+=1
print(num_similar)
#Now have to find the accuracy
accuracy=0.0
accuracy=(float(num_similar)/float(230))*100
print("accuracy is: "+ str("%.2f" % round(accuracy,2)) +"%")






















