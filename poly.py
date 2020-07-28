# Load Data
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Fitting Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

# Visualize Linear Regression Results
plt.scatter(X,y, color="red")
plt.plot(X, lin_reg.predict(X))
plt.title("Linear Regression")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()

# Linear Regression prediction
lin_reg.predict([[6.5]])

# Convert X to polynomial format
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X)


# Passing X_poly to LinearRegression
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)

# Visualize Poly Regression Results
plt.scatter(X,y, color="red")
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)))
plt.title("Poly Regression - Degree 2")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()


# Polynomial Regression prediction
new_salary_pred = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
print('The predicted salary of a person at 6.5 Level is ',new_salary_pred)

# Change degree to 3 and run steps 5-8

# Convert X to polynomial format
poly_reg = PolynomialFeatures(degree=3)
X_poly = poly_reg.fit_transform(X)

 # Passing X_poly to LinearRegression
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)

# Visualize Poly Regression Results
plt.scatter(X,y, color="red")
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)))
plt.title("Poly Regression Degree 3")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()

# Polynomial Regression prediction
new_salary_pred = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
print('The predicted salary of a person at 6.5 Level is ',new_salary_pred)


# Change degree to 4 and run steps 5-8

# Convert X to polynomial format
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)

# Passing X_poly to LinearRegression
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)

# Visualize Poly Regression Results
plt.scatter(X,y, color="red")
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)))
plt.title("Poly Regression Degree 4")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()

# Polynomial Regression prediction
new_salary_pred = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
print('The predicted salary of a person at 6.5 Level is ',new_salary_pred)
