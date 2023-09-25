import names
import random
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

def student_generator(x):
    students = []
    grades = []
    for i in range(x):
        new_student = {
        }
        new_student['name'] = names.get_full_name()
        new_student['age'] = random.randint(18,50)
        new_student['study habit'] = random.randint(1,10)
        students.append(new_student)

        if new_student['study habit'] >= 8:
            grades.append(random.randint(15,20))
        elif new_student['study habit'] >= 4 and new_student['study habit'] <= 8:
            grades.append(random.randint(5, 15))
        elif new_student['study habit'] >= 0 and new_student['study habit'] <= 4:
            grades.append(random.randint(0,5))
    return students, grades 


students, grades = student_generator(100)

data = pd.DataFrame(students)

X_train, X_test, y_train, y_test = train_test_split(data, grades, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train[['age', 'study habit']], y_train)

y_pred = model.predict(X_test[['age','study habit']])

mae = mean_absolute_error(y_test,y_pred)

print(f'Mean Absolut Error {mae}')

new_student_data = {
    'name': ['jonatan'],
    'age': [31],
    'study habit': [10]
}

new_student_pd = pd.DataFrame(new_student_data)
predicted_grade = model.predict(new_student_pd[['age','study habit']])

print(f"{new_student_data['name']} predicted grade is: {predicted_grade}")
