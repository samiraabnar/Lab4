import numpy as np

class RNN:

    # Initializing the RNN object
    def __init__(self, input_dim,output_dim,hidden_dim=100):
        # Assign instance variables
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.__init_parameters()


    # Randomly initialize the network parameters
    def __init_parameters(self):
        self.U = np.random.uniform(-np.sqrt(1. / self.input_dim), np.sqrt(1. / self.input_dim),
                                   (self.hidden_dim, self.input_dim))
        self.V = np.random.uniform(-np.sqrt(1. / self.hidden_dim), np.sqrt(1. / self.hidden_dim),
                                   (self.input_dim, self.hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1. / self.hidden_dim), np.sqrt(1. / self.hidden_dim),
                                   (self.hidden_dim, self.hidden_dim))


    def forward_pass(self):
        # The total number of time steps
        T = len(x)
        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden, which we set to 0
        s = np.zeros((T + 1, self.hidden_dim))
        s[-1] = np.zeros(self.hidden_dim)
        # The outputs at each time step. Again, we save them for later.
        o = np.zeros((T, self.word_dim))
        # For each time step...
        for t in np.arange(T):
            # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
            s[t] = np.tanh(self.U[:, x[t]] + self.W.dot(s[t - 1]))
            o[t] = softmax(self.V.dot(s[t]))
        return [o, s]

    def predict(self, x):
        # Perform forward propagation and return index of the highest score
        o, s = self.forward_propagation(x)
        return np.argmax(o, axis=1)

    def back_prppagation_through_time(self):
        NotImplementedError



if __name__ == '__main__':
    # example task #1: teach the RNN to Count
    # Teach the RNN english alphabets

    #Teach the RNN to generate english words
    #plot character embedings

    #Teach the RNN to generate english sentences

    #Teach RNN to parse --> PEN TREEBank




