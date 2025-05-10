import uuid
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Task:
    def __init__(self, description, priority):
        self.id = str(uuid.uuid4())[:8]
        self.description = description
        self.priority = priority

    def __str__(self):
        return f"[{self.id}] {self.description} (Priority: {self.priority})"

class TaskManager:
    def __init__(self):
        self.tasks = []

    def add_task(self, description, priority):
        task = Task(description, priority)
        self.tasks.append(task)
        print("Task added.")

    def remove_task(self, task_id):
        self.tasks = [task for task in self.tasks if task.id != task_id]
        print("Task removed (if it existed).")

    def list_tasks(self):
        if not self.tasks:
            print("No tasks available.")
        else:
            sorted_tasks = sorted(self.tasks, key=lambda t: t.priority)
            for task in sorted_tasks:
                print(task)

    def recommend_tasks(self, input_description):
        if not self.tasks:
            print("No tasks to recommend.")
            return

        descriptions = [task.description for task in self.tasks]
        descriptions.append(input_description)

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(descriptions)
        similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]

        ranked_tasks = sorted(
            zip(self.tasks, similarity_scores),
            key=lambda x: x[1],
            reverse=True
        )

        print("Top recommended tasks:")
        for task, score in ranked_tasks[:3]:
            print(f"{task} (Score: {score:.2f})")

def main():
    manager = TaskManager()

    while True:
        print("\n1. Add Task\n2. Remove Task\n3. List Tasks\n4. Recommend Tasks\n5. Exit")
        choice = input("Enter choice: ")

        if choice == '1':
            desc = input("Enter task description: ")
            prio = int(input("Enter priority (1=High, 5=Low): "))
            manager.add_task(desc, prio)

        elif choice == '2':
            tid = input("Enter Task ID to remove: ")
            manager.remove_task(tid)

        elif choice == '3':
            manager.list_tasks()

        elif choice == '4':
            desc = input("Enter a task description to get recommendations: ")
            manager.recommend_tasks(desc)

        elif choice == '5':
            break

        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()
