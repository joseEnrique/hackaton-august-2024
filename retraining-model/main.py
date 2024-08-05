# Import the Quix Streams modules for interacting with Kafka:
from quixstreams import Application
# (see https://quix.io/docs/quix-streams/v2-0-latest/api-reference/quixstreams.html for more details)

# Import additional modules as needed
import os
from river import linear_model
from river import compose
from river import preprocessing
from river import metrics
from deep_river import classification
from torch import nn
from torch import optim
from torch import manual_seed
from dotenv import load_dotenv
import json


# import the dotenv module to load environment variables from a file
_ = manual_seed(42)

class MyModule(nn.Module):
     def __init__(self, n_features):
         super(MyModule, self).__init__()
         self.dense0 = nn.Linear(n_features, 5)
         self.nonlin = nn.ReLU()
         self.dense1 = nn.Linear(5, 2)
         self.softmax = nn.Softmax(dim=-1)

     def forward(self, X, **kwargs):
         X = self.nonlin(self.dense0(X))
         X = self.nonlin(self.dense1(X))
         X = self.softmax(X)
         return X


# use rocauc as the metric for evaluation
metric = metrics.Accuracy()
deep_metric = metrics.Accuracy()

# create a simple LR model with a scaler
model = compose.Pipeline(
    preprocessing.StandardScaler(), linear_model.LogisticRegression()
)
deep_model = compose.Pipeline(
     preprocessing.StandardScaler(),
     classification.Classifier(module=MyModule, loss_fn='binary_cross_entropy', optimizer_fn='adam'))


def predict(x,y):
    y_pred = model.predict_one(x)
    model.learn_one(x, y)
    metric.update(y, y_pred)
    return metric.get()

    # print(metric)    
def deep_predict(x,y):
    y_pred = deep_model.predict_one(x)
    deep_model.learn_one(x, y)
    deep_metric.update(y, y_pred)
    return deep_metric.get()

def send_outtopic(quix_app,c,deepl,linear):
    outtopic_name = os.getenv("output", "")
    if outtopic_name == "":
        raise ValueError("The 'output' environment variable is required. This is the output topic that data will be published to.")
    topic = quix_app.topic(outtopic_name)

    data = {'deepl':deepl,'linear':linear}
   
    json_data = json.dumps(data)
    producer.produce(topic=topic.name,value=json_data)


if __name__ == '__main__':
    load_dotenv(override=False)

    # Create an Application and run it in the main thread.
    quix_app = Application()
    producer = quix_app.get_producer()
    input_topic = quix_app.topic(os.environ["input"])

    sdf = quix_app.dataframe(input_topic)

    sdf["linear"] = sdf.apply(lambda message: predict(message['x'],message['y']))
    sdf["deep"] = sdf.apply(lambda message: deep_predict(message['x'],message['y']))
    # sdf = sdf.update(lambda message: print(f"Linear: {message['linear']}"))
    # sdf = sdf.update(lambda message: print(f"Deep: {message['deep']}"))
    sdf = sdf.update(lambda message: send_outtopic(quix_app,producer,message['deep'],message['linear']) )
    quix_app.run(sdf)
