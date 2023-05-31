# to run, type locust -f performance_test.py in terminal
# result @ http://localhost:8089/
from locust import HttpUser, task, between


class ApiUser(HttpUser):
    wait_time = between(1, 3)  # Wait time between consecutive tasks
    weight = 5

    @task
    def my_task(self):
        # Define the payload for the POST request
        payload = {
            'input_data': 'many generic sentences'
        }

        # Send a POST request to the endpoint
        self.client.post('predict', json=payload)


class AdminUser(HttpUser):
    wait_time = between(1, 3)
    weight = 1
    @task
    def my_task(self):
        payload = {'input_data': 'a possibly harder sentence to predict'}
        self.client.post('predict', json=payload)


class GradioUser(HttpUser):
    wait_time = between(1, 3)
    weight = 3
    @task
    def my_task(self):
        payload = {'input_data': 'Its sunny out'}
        self.client.post('predict', json=payload)
