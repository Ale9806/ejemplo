# Example 2

More code and stuff

Since this is markdown we can add code snippets: 
``` Python
### ACTIVATION FUNCTIONS 

def Sigmoid(Z):
    s = 1/(1+np.exp(-Z))
    return s

def ReLU(Z):
    return Z * (Z > 0)

# Derivatives of Activation Functions:

def dSigmoid(A):
    return A*(1-A)

def dReLU(Z):
    return 1 * (Z > 0)


### PARAMETER INITALIZATION
def initialize_parameters_xavier(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    Xavier = 1/sqrt(l-1)
    """
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims) # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters

def initialize_parameters_he(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    Xavier = 1/sqrt(l-1)
    """
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims) # number of layers in the network

    for l in range(1, L):                                                          
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])  *  np.sqrt(2 / layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters

### FORWARD PASS AND BACKPROPAGATION
def forward_propagation_with_dropout(X, parameters, keep_prob ):
    """
    Implements the forward propagation
    """
    np.random.seed(1)
    dropout ={}
    forward = {}
    forward["A" + str(0)] = X # set A0 to X this will be very helpful for for loopimplementation
    
    Num_of_parameters = len(parameters) // 2  # since we have W and b's  (but W and b share the same indexing for 1 layer)

    for i in range(1,Num_of_parameters+1):
        # LINEAR
        forward["Z"+str(i)] = np.dot(parameters["W"+str(i)], forward["A" + str(i-1)]) + parameters["b"+str(i)]
        
        # DROPOUT IS ONLY APPLIED TO HIDEN LAYERS 
        if i != Num_of_parameters: # last layer has to have a SIGMOID activation 
            forward["A"+str(i)] = ReLU(forward["Z"+str(i)])
            dropout["D"+str(i)]  = np.random.rand(*forward["A"+str(i)].shape)
            dropout["D"+str(i)]  = ( dropout["D"+str(i)]  < keep_prob ).astype(int) 
            forward["A"+str(i)] = forward["A"+str(i)]*dropout["D"+str(i)]
            forward["A"+str(i)] = forward["A"+str(i)]  / keep_prob  
        else:
            forward["A"+str(i)] = Sigmoid(forward["Z"+str(i)])
            
    AL =  forward["A"+str(i)]
    
    return AL, forward, dropout


def backward_propagation_with_dropout(X, Y,forward,parameters, keep_prob,dropout):
    """
    Implement the backward propagation 
    """
    m = X.shape[1]
    gradients = {}
    Num_of_parameters = len(parameters) // 2
    
    AL = list(forward)[-1] #calls last element of dicitonari in this cas AL
    gradients["dZ"+str(Num_of_parameters)] =  forward[AL] - Y #Derivative respect to cost multiply by derivative of sig
    
    for i in range(Num_of_parameters,0,-1):      
        
        if i != Num_of_parameters: # we have already calcualted dZ for AL
            
            gradients["dA"+str(i)]  = np.dot(parameters["W"+str(i+1)].T, gradients["dZ"+str(i+1)])
            gradients["dA"+str(i)]  = gradients["dA"+str(i)]*dropout["D"+str(i)]
            gradients["dA"+str(i)]  =  gradients["dA"+str(i)] /  keep_prob
            gradients["dZ"+str(i)]  = np.multiply(gradients["dA"+str(i)],dReLU(forward["A"+str(i)]))
        
        gradients["dW"+str(i)] = 1./m * np.dot(gradients["dZ"+str(i)],forward["A"+str(i-1)].T)
        gradients["db"+str(i)] = 1./m * np.sum(gradients["dZ"+str(i)], axis=1, keepdims = True)
    
    return gradients


    ###COST
    def compute_cost(AL, Y):
    """
    Implement the cost function
    """
    
    logprobs = np.multiply(-np.log(AL),Y) + np.multiply(-np.log(1 - AL), 1 - Y)
    cost = np.sum(logprobs)
    
    return cost

    ###  ADAM OPTIMIZATION (MOMENTUM + )
    def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    np.random.seed(seed)            # To make your "random" minibatches the same as ours
    m = X.shape[1]                  # number of training examples
    mini_batches = []
        
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1,m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[:, (k)*mini_batch_size : (k+1)* mini_batch_size]
        mini_batch_Y = shuffled_Y[:, (k)*mini_batch_size : (k+1)* mini_batch_size]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[:,num_complete_minibatches * mini_batch_size:]
        mini_batch_Y = shuffled_Y[:,num_complete_minibatches * mini_batch_size:]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def initialize_adam(parameters) :
    """
    Initializes v and s as two python dictionaries 
    """
    
    L = len(parameters) // 2 # number of layers in the neural networks
    v = {}
    s = {}
    
    # Initialize v, s. Input: "parameters". Outputs: "v, s".
    for l in range(L):
 
        v["dW" + str(l+1)] = np.zeros_like(parameters["W" + str(l+1)])
        v["db" + str(l+1)] = np.zeros_like(parameters["b" + str(l+1)])
        s["dW" + str(l+1)] = np.zeros_like(parameters["W" + str(l+1)])
        s["db" + str(l+1)] = np.zeros_like(parameters["b" + str(l+1)])
    
    return v, s

def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate = 0.01,
                                beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
    """
    Update parameters using Adam
    """
    
    L = len(parameters) // 2                 # number of layers in the neural networks
    v_corrected = {}                         # Initializing first moment estimate, python dictionary
    s_corrected = {}                         # Initializing second moment estimate, python dictionary
    
    # Perform Adam update on all parameters
    for l in range(L):
        # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".

        v["dW" + str(l + 1)] = beta1 * v["dW" + str(l + 1)] + (1 - beta1) * grads['dW' + str(l + 1)]
        v["db" + str(l + 1)] = beta1 * v["db" + str(l + 1)] + (1 - beta1) * grads['db' + str(l + 1)]

        # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
        v_corrected["dW" + str(l + 1)] = v["dW" + str(l + 1)] / (1 - beta1**t)
        v_corrected["db" + str(l + 1)] = v["db" + str(l + 1)] / (1 - beta1**t)

        # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
        s["dW" + str(l + 1)] = beta2 * s["dW" + str(l + 1)] + (1 - beta2) * grads['dW' + str(l + 1)]**2
        s["db" + str(l + 1)] = beta2 * s["db" + str(l + 1)] + (1 - beta2) * grads['db' + str(l + 1)]**2

        # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
        s_corrected["dW" + str(l + 1)] = s["dW" + str(l + 1)] / (1 - beta2**t)
        s_corrected["db" + str(l + 1)] = s["db" + str(l + 1)] / (1 - beta2**t)

        # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * v_corrected["dW" + str(l + 1)] / (np.sqrt(s_corrected["dW" + str(l + 1)]) + epsilon)
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * v_corrected["db" + str(l + 1)] / (np.sqrt(s_corrected["db" + str(l + 1)]) + epsilon)

    return parameters, v, s

```