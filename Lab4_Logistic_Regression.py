# ==========================================
# BAI 1: LOGISTIC REGRESSION
# ==========================================

# Import thu vien
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# 1. Load du lieu Breast Cancer
cancer = load_breast_cancer()

X = cancer.data
y = cancer.target

print("Kich thuoc du lieu:", X.shape)
print("Ten cac lop:", cancer.target_names)

# 2. Chia du lieu Train va Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("So mau Train:", X_train.shape[0])
print("So mau Test:", X_test.shape[0])

# 3. Chuan hoa du lieu
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Tao mo hinh Logistic Regression
model = LogisticRegression(max_iter=10000)

# 5. Huan luyen mo hinh
model.fit(X_train_scaled, y_train)

print("Da huan luyen xong mo hinh Logistic Regression!")
# ==========================================
# BAI 2: DANH GIA MO HINH PHAN LOAI
# ==========================================

# Import cac thu vien danh gia
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report

# 1. Du doan tren tap Test
y_pred = model.predict(X_test_scaled)

# 2. Tinh Accuracy
accuracy = accuracy_score(y_test, y_pred)

# 3. Tinh Precision
precision = precision_score(y_test, y_pred)

# 4. Tinh Recall
recall = recall_score(y_test, y_pred)

# 5. In ket qua
print("\n===== KET QUA BAI 2 =====")

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

# 6. Classification Report
print("\nClassification Report:")

print(classification_report(
    y_test,
    y_pred,
    target_names=cancer.target_names
))
# ==========================================
# BAI 3: CONFUSION MATRIX
# ==========================================

# Import thu vien
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# 1. Tao ma tran nham lan
cm = confusion_matrix(y_test, y_pred)

print("\n===== CONFUSION MATRIX =====")
print(cm)

# 2. Lay cac gia tri TN, FP, FN, TP
tn, fp, fn, tp = cm.ravel()

print("\n===== CAC GIA TRI =====")
print("TN =", tn)
print("FP =", fp)
print("FN =", fn)
print("TP =", tp)

# 3. Hien thi ma tran nham lan
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=cancer.target_names
)

disp.plot(cmap="Blues", values_format="d")

plt.title("Confusion Matrix - Breast Cancer")
plt.show()

# 4. Giai thich ket qua
print("\n===== GIAI THICH =====")
print("TP: Du doan dung lop duong tinh")
print("TN: Du doan dung lop am tinh")
print("FP: Du doan nham am tinh thanh duong tinh")
print("FN: Du doan nham duong tinh thanh am tinh")
# ==========================================
# BAI 4: LOGISTIC REGRESSION TREN DIGITS
# ==========================================

# Import thu vien
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

import numpy as np
import matplotlib.pyplot as plt

# 1. Load du lieu Digits
digits = load_digits()

X_digits = digits.data
y_digits = digits.target

print("\n===== BAI 4: DIGITS =====")
print("Kich thuoc du lieu:", X_digits.shape)

# 2. Chia du lieu Train va Test
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    X_digits,
    y_digits,
    test_size=0.2,
    random_state=42,
    stratify=y_digits
)

print("So mau Train:", X_train_d.shape[0])
print("So mau Test:", X_test_d.shape[0])

# 3. Chuan hoa du lieu
scaler_digits = StandardScaler()

X_train_d_scaled = scaler_digits.fit_transform(X_train_d)
X_test_d_scaled = scaler_digits.transform(X_test_d)

# 4. Khoi tao mo hinh Logistic Regression
model_digits = LogisticRegression(max_iter=10000)

# 5. Huan luyen mo hinh
model_digits.fit(X_train_d_scaled, y_train_d)

print("Da huan luyen xong mo hinh Digits!")

# 6. Du doan
y_pred_d = model_digits.predict(X_test_d_scaled)

# 7. Danh gia mo hinh
accuracy_d = accuracy_score(y_test_d, y_pred_d)

print("\n===== KET QUA DIGITS =====")
print("Accuracy:", accuracy_d)

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test_d, y_pred_d))

# 8. Tim cac mau du doan sai
wrong_indices = np.where(y_test_d != y_pred_d)[0]

print("\n===== 5 MAU DU DOAN SAI DAU TIEN =====")

for i in range(min(5, len(wrong_indices))):

    idx = wrong_indices[i]

    actual = y_test_d[idx]
    predicted = y_pred_d[idx]

    print("\nMau", i + 1)
    print("Nhan thuc te:", actual)
    print("Nhan du doan:", predicted)

    # Hien thi hinh anh chu so
    plt.figure(figsize=(3, 3))

    plt.imshow(
        X_test_d[idx].reshape(8, 8),
        cmap="gray_r"
    )

    plt.title(
        f"Actual: {actual} - Predicted: {predicted}"
    )

    plt.axis("off")
    plt.show()

print("\n===== DA HOAN THANH LAB 4 =====")