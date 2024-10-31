from locust import HttpUser, TaskSet, task

class ModelLoadTest(TaskSet):
    @task
    def predict(self):
        with open("test_image.jpg", "rb") as image_file:
            self.client.post("/predict", files={"image": image_file})

class LoadTestUser(HttpUser):
    tasks = [ModelLoadTest]
    min_wait = 1000
    max_wait = 2000
