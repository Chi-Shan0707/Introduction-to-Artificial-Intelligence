class Student:
    def __init__(self, name, age, grades=None):
        self.name = name
        self.age = age
        self.grades = grades or {}

    def add_grade(self, subject, score):
        self.grades[subject] = score

    def average(self):
        if not self.grades:
            return 0.0
        return sum(self.grades.values()) / len(self.grades)

    def __repr__(self):
        return f"Student({self.name}, age={self.age}, avg={self.average():.1f})"

if __name__ == "__main__":
    s = Student("Li Wei", 20)
    s.add_grade("Math", 92)
    s.add_grade("English", 85)
    s.add_grade("Physics", 78)
    print(s)
    print(f"Grades: {s.grades}")
