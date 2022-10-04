# https://betterdatascience.com/apache-kafka-in-python-how-to-stream-data-with-producers-and-consumers/

import time, json, random
from datetime import datetime
from data_generator import generate_message
from kafka import KafkaProducer


def serializer(message):
    return json.dumps(message).encode('utf-8')


# Kafka Producer
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=serializer
)

if __name__=='__main__':
    while True:

        dummy_message = generate_message()
        print(f'Producing Message @ {datetime.now()} | Message ={str(dummy_message)}')
        producer.send('messages',dummy_message)

        time_to_sleep = random.randint(1,11)
        time.sleep(time_to_sleep)

