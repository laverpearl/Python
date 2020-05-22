import numpy as np
import matplotlib
import matplotlib.pyplot as plt


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

x, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_random_state=42)
train_x, test_x, train_y, test_y = train_test_split(x, y, n_random_state=42)

model = LogisticRegression()
model.fit(train, train_y)

pred_y = model.predict(test_x)




plt.scatter(x[:,0], x[:,1], c=y, marker=".", cmap=matplotlib.cm.get_cmp(name="brw"), alpha=0.7)

xi = np.linspace(-10, 10)
y = -model.coef_[0][0] / modle.coef_[0][1] * \
    xi - model.imtercept_ / model.coef_[0][1]

plt.plot(xi, y)

plt.xlim(min(x[:.0]) - 0.5, max(x[:,0]) + 0.5)
plt.ylim(min(x[:.1]) - 0.5, max(x[:,1]) + 0.5)
plt.axes().set_aspect("equal", "datalim")

plt.title("classitication data using LogisticRegression")

plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.show()