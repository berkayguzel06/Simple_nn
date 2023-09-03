import neural_network as nn
import data_types as dt
import matplotlib.pyplot as plt

def main():
    X,y = dt.generate_vertical_data(100,3)
    y_reshaped = y.reshape(-1, 1)
    z = nn.Network()
    nn.Layer(2,3,activation='relu')
    nn.Layer(3,4,activation='softmax')
    nn.Layer(4,3,activation='softmax')
    z.fit(X,y_reshaped,epochs=300,learning_rate=0.05)
    output = nn.layers[-1].output
    loss = nn.layers[-1].loss
    print(output[:5])
    print(loss)
    #plt.scatter(X[:,0],X[:,1],c=y,s=40,cmap='brg')
    #plt.show()

if __name__ == '__main__':
    main()