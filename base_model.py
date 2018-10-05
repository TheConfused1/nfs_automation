import keras
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import Dense,Flatten
from keras.models  import Model
import numpy as np


shape=(299,299,1)
model_path=""                                                  #


base=InceptionResNetV2(include_top=False,input_shape=shape)
#base.load_weights("path")                                 #path to be given

X=base.output

X=Flatten()(X)

X=Dense(512,activation="relu" )(X)
X=Dense(64,activation="relu")(X)
X=Dense(256,activation="relu")(X)

pred=Dense(9,activation="softmax")(X)

model=Model(inputs=base.input,outputs=pred)

for layer in base.layers[:-20]:
    layer.trainable=False


#model.summary()
model.compile(optimizer="Adam",loss="catagorical_crossentropy",metrics=["accuracy"])
#model.save_model("my_model.h5")

i=0

while True:

    training_file="training_data-"+i+".npy"                                 #give training file model_path

    if os.path.isfile(training_file):

        print("loading file "+training_file+" ............")
        train=np.load(training_file)

        X=np.array([ j[0] for j in train ]).reshape(-1,299,299,1)
        y=[ j[1] for j in train ]

        model.fit(X,y,batch_size=256,epochs=1,validation_split=0.05,shuffle=True)


        if i%20==0:
            print("saving model........")
            model.save_model("my_model.h5")

    else:
        print("Training complete.")
        print("Ab chill karo!!!!!")

    i=i+1                          #updating file counter
