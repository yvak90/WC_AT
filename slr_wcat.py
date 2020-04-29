# For reading data set
# importing necessary libraries
import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values
import matplotlib.pylab as plt #for different types of plots
import pickle

# reading csv file using pandas library
wcat = pd.read_csv("E:/All Projects 2.0/simple Linear Regression/Waist_AT/Dataset/wc-at.csv")

wcat.columns
wcat.describe()

#plt.hist(wcat.Waist)
#plt.boxplot(wcat.Waist)
#
#plt.hist(wcat.AT)
#plt.boxplot(wcat.AT)

#plt.scatter(x=wcat['Waist'], y=wcat['AT'],color='blue')# Scatter plot
#plt.xlabel("Waist")
#plt.ylabel("AT")

np.corrcoef(wcat.Waist, wcat.AT) #correlation
wcat["AT"].corr(wcat.Waist)

import statsmodels.formula.api as smf

model = smf.ols('AT ~ Waist', data=wcat).fit()
model.summary()

pred1 = model.predict(wcat['Waist'])

#plt.scatter(x=wcat['Waist'], y=wcat['AT'], color='red')# Scatter plot
#plt.plot(wcat['Waist'],pred1, color='black')
#plt.xlabel("Waist")
#plt.ylabel("AT")

np.corrcoef(wcat["AT"], pred1)
pred1.corr(wcat["AT"]) # accuracy of the model

print (model.conf_int(0.01)) # 99% confidence interval

res = wcat.AT - pred1
sqres = res*res
mse = np.mean(sqres)
rmse = np.sqrt(mse)


######### Model building on Transformed Data

# Log Transformation
# x = log(waist); y = at

np.corrcoef(np.log(wcat.Waist), wcat.AT) #correlation

model2 = smf.ols('AT ~ np.log(Waist)',data=wcat).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(wcat['Waist']))
pred2

#plt.scatter(x=np.log(wcat['Waist']),y=wcat['AT'],color='brown');
##plt.xlabel("Waist");
##plt.ylabel("AT")
#plt.plot(np.log(wcat['Waist']), pred2, color="blue")

pred2.corr(wcat["AT"]) # accuracy
print(model2.conf_int(0.01)) # 99% confidence level

res2 = wcat.AT - pred2
sqres2 = res2*res2
mse2 = np.mean(sqres2)
rmse2 = np.sqrt(mse2)

# Exponential transformation
#plt.scatter(x=wcat['Waist'], y=np.log(wcat['AT']),color='orange')

np.corrcoef(wcat.Waist, np.log(wcat.AT)) #correlation

model3 = smf.ols('np.log(AT) ~ Waist',data=wcat).fit()
pred_log = model3.predict(pd.DataFrame(wcat['Waist']))
pred_log
pred3 = np.exp(pred_log)

model3.params
model3.summary()

#plt.scatter(x=wcat["Waist"], y=wcat["AT"], color="blue")
#plt.plot(wcat["Waist"],pred3, color="black")
#pred3.corr(wcat["AT"])

print(model3.conf_int(0.01)) # 99% confidence level

res3 = wcat.AT - pred3
sqres3 = res3*res3
mse3 = np.mean(sqres3)
rmse3 = np.sqrt(mse3)


#Qudratic model
wcat["Waist_sq"] = wcat["Waist"]*wcat["Waist"]

model4 = smf.ols("np.log(AT)~Waist+Waist_sq", data=wcat).fit()
pred4 = np.exp(model4.predict(wcat))
model4.params
model4.summary()

pred4.corr(wcat["AT"])
#
plt.scatter(x=wcat["Waist"], y=wcat["AT"], color="red"); plt.xlabel="Waist"; plt.ylabel="AT"
plt.plot(wcat["Waist"],pred4, color = "blue")

res4 = wcat.AT - pred4
sqres4 = res4*res4
mse4 = np.mean(sqres4)
rmse4 = np.sqrt(mse4)


#polynomial model

wcat["Waist_cb"] = wcat["Waist"]*wcat["Waist"]*wcat["Waist"]


model5 = smf.ols('np.log(AT)~Waist+Waist_sq+Waist_cb', data=wcat).fit()
model5.params
model5.summary()
pred5 = np.exp(model5.predict(wcat))

wcat["AT_pred"] = pred5

pred5.corr(wcat["AT"])
#plt.scatter(x=wcat["Waist"], y=wcat["AT"], color="blue");plt.xlabel="Waist";plt.ylabel="AT"
#plt.plot(wcat["Waist"], pred5, color="red")

res5 = wcat.AT - pred5
sqres5 = res5*res5
mse5 = np.mean(sqres5)
rmse5 = np.sqrt(mse5)

#from sklearn.linear_model import LinearRegression


model5.save("slr_wcat.pkl")




from statsmodels.regression.linear_model import OLSResults
model = OLSResults.load("slr_wcat.pkl")

#type(new_results)
# saving model to disk

#pickle.dump(model5, open("slr_wcat.pkl","wb"))

# loading model to compare results
#slr_wcat = pickle.load(open("slr_wcat.pkl", "rb"))

x = np.exp(model.predict(pd.DataFrame([[36,1296,46656]], columns=["Waist", "Waist_sq", "Waist_cb"])))
print(float(round(x,2)))
#print(round(np.exp(model5.predict(pd.DataFrame([[80,6400,512000]], columns=["Waist", "Waist_sq", "Waist_cb"]))),2))


#print(np.exp(pd.DataFrame(model5.predict([[80,6400,512000]], columns=["Waist", "Waist_sq", "Waist_cb"]  ))))
#xyz = model.predict(pd.DataFrame([[75]], columns=["Waist"]))
#
#xyz = np.exp(model5.predict(pd.DataFrame([[80,6400,512000]], columns=["Waist", "Waist_sq", "Waist_cb"])))
#print(xyz)
#
#type(xyz)
#help(model5.predict())
#
#x = pd.DataFrame([[75,5625,421875]], columns=["Waist", "Waist_sq", "Waist_cb"])




