import LSTM as i1
import LSTM_LL1 as i2
import LSTM_LL2 as i3



if __name__ == "__main__":

       ''' Model parameters to iterate through '''
    BATCH_SIZE = 15 #[10, 25, 50]     # mini_batch size
    MAX_EPOCH = 50      # maximum epoch to train
    hidden_size = 100 #[25,100,500,1000]    # size of hidden layer
    n_layers = 1        # number of lstm layers



    """ defining bounding box """
    # small bounding box
    xmin, xmax = -74.2754, -73.9374
    ymin, ymax = 40.4041, 40.6097
    box_size = 'S'

    i1.main(BATCH_SIZE, MAX_EPOCH, hidden_size, n_layers,
        box_size, xmin, xmax, ymin, ymax)
    i2.main(BATCH_SIZE, MAX_EPOCH, hidden_size, n_layers,
        box_size, xmin, xmax, ymin, ymax)
    i3.main(BATCH_SIZE, MAX_EPOCH, hidden_size, n_layers,
        box_size, xmin, xmax, ymin, ymax)




    # medium bounding box
    xmin, xmax = -74.6764, -69.5103
    ymin, ymax = 39.9218, 41.8667
    box_size = 'M'

    i1.main(BATCH_SIZE, MAX_EPOCH, hidden_size, n_layers,
        box_size, xmin, xmax, ymin, ymax)
    i2.main(BATCH_SIZE, MAX_EPOCH, hidden_size, n_layers,
        box_size, xmin, xmax, ymin, ymax)
    i3.main(BATCH_SIZE, MAX_EPOCH, hidden_size, n_layers,
        box_size, xmin, xmax, ymin, ymax)



    # large bounding box 
    xmin, xmax = -77.9897, -66.2786
    ymin, ymax = 35.7051, 45.5341
    box_size = 'L'

    i1.main(BATCH_SIZE, MAX_EPOCH, hidden_size, n_layers,
        box_size, xmin, xmax, ymin, ymax)
    i2.main(BATCH_SIZE, MAX_EPOCH, hidden_size, n_layers,
        box_size, xmin, xmax, ymin, ymax)
    i3.main(BATCH_SIZE, MAX_EPOCH, hidden_size, n_layers,
        box_size, xmin, xmax, ymin, ymax)