from gradio_client import Client

user_input = input("Enter text to send to the endpoint: ")
client = Client("https://shubhamg79-emotionclassification.hf.space/")
result = client.predict(
    user_input,  # str representing input in 'text' Textbox component
    api_name="/predict"
)
print(result)
