# Import the Quix Streams modules for interacting with Kafka:
from quixstreams import Application
# (see https://quix.io/docs/quix-streams/v2-0-latest/api-reference/quixstreams.html for more details)

# Import additional modules as needed
import pandas as pd
import json
import random
from time import sleep
import os
from river import datasets

# import the dotenv module to load environment variables from a file
from dotenv import load_dotenv

load_dotenv(override=False)

# Create an Application.
app = Application()

# Define the topic using the "output" environment variable
topic_name = os.getenv("output", "")
if topic_name == "":
    raise ValueError("The 'output' environment variable is required. This is the output topic that data will be published to.")

topic = app.topic(topic_name)



def main():
    """
    Read data from the CSV file and publish it to Kafka
    """

    producer = app.get_producer()
    with producer:
        dataset = datasets.CreditCard()   
        while True:
            for x, y in dataset:
                print(f"Sending: {x, y}")
                data = {"x": x, "y": y}
                json_data = json.dumps(data)
                # producer.send("ml_training_data", value=data)
                producer.produce(
                    topic=topic.name,
                    # key=message_key,
                    value=json_data
                )
                sleep(0.5)
        

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Exiting.")