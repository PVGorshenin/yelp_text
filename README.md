## Hello everyone!


# The goal of this repo

I chose textual information, because I have "ready to go" pipelines.
It was rusty, but I wanted to make it work. Also I solved this kind of task not long ago. Which gave me opportunity
to concentrate mostly on engineering part.

Final goal was to make a working docker.

# Definition of task

We have user reviews on some kind of goods. Those consist of textual review, star rating, etc. I chose to use field
named `useful` for the target. Business idea behind it is to be able to predict reviews with high popularity in order to
place them higher. We could have popular review at the moment, estimated one. But they are not new.
If we can predict useful review, we can lift new review with no user's feedback at the moment.


# Process of training

I used just first 10K of observations, because of computational cost. 

You can see training process in the branches run_lr and run_bert inside the repo. I used val data as the test data.
It is possible,because I don't use it as any tuning criteria. I don't tune at all. 
Ridge regression is used just as baseline. Bert is trained on MSE loss.
BertForSequenceClassification uses it for regression task under the hood. And metrics on validation set, 
used for after training estimation, is MAE. Just because it is easy to understand.

Bert show MAE ~= 0.99 compare to 1.14 of naive algorithm. I believe there is still a lot of opportunities to
increase metric. But it wasn't my goal. 

Further process of estimation would involve using ranking-specific metrics. I would also take special look on model
performance on high-value labels. 

# Inference

Inference is constructed with possibility to be generalized on N models. Predictions could be less than zero. We don't 
use this in real life. But for ranging task it's even better to leave negative numbers as it is. 




