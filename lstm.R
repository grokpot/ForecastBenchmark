# http://datasideoflife.com/?p=1171
# https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
# https://github.com/MohammadFneish7/Keras_LSTM_Diagram
# https://medium.com/deep-learning-with-keras/lstm-understanding-output-types-e93d2fb57c77
# https://www.altumintelligence.com/articles/a/Time-Series-Prediction-Using-LSTM-Deep-Neural-Networks
# https://keras.rstudio.com/articles/faq.html
# https://stackoverflow.com/questions/38714959/understanding-keras-lstms
# https://towardsdatascience.com/time-series-analysis-visualization-forecasting-with-lstm-77a905180eba
# https://keras.rstudio.com/articles/examples/index.html
# https://colah.github.io/posts/2015-08-Understanding-LSTMs/
# https://www.kaggle.com/rtatman/beginner-s-intro-to-rnn-s-in-r/log
# https://cran.rstudio.com/web/packages/keras/vignettes/sequential_model.html
# https://datascience.stackexchange.com/questions/42499/does-this-encoder-decoder-lstm-make-sense-for-time-series-sequence-to-sequence
# https://towardsdatascience.com/time-series-forecasting-with-deep-learning-and-attention-mechanism-2d001fc871fc
# https://github.com/lkulowski/LSTM_encoder_decoder
# https://machinelearningmastery.com/encoder-decoder-attention-sequence-to-sequence-prediction-keras/
# https://www.altumintelligence.com/articles/a/Time-Series-Prediction-Using-LSTM-Deep-Neural-Networks


setwd("/Users/prater/Dev/ForecastBenchmark")
source("R/benchmark.R")
source("R/forecasting.R")
source("R/measures.R")
load(file = "R/sysdata.rda")
library(forecast)
library(keras)
library(caret)
library(xts)

DEBUG = FALSE

p <- function(msg, val) {
    if (DEBUG) {
        print(paste0(msg, val))
    }
}


SCALER = "minmax"
get_scale_factors <- function(x) {
    if (SCALER == "minmax") {
        return (c(min(x), max(x)))
    } else {
        return (c(mean(tsd), sd(tsd)))
    }
}
scale <- function(x, scale_factors) {
    if (SCALER == "minmax") {
        return ((x - scale_factors[1]) / (scale_factors[2] - scale_factors[1]))
    } else {
        return ((x - scale_factors[1]) / scale_factors[2])
    }
    
}
unscale <- function(x, scale_factors) {
    if (SCALER == "minmax") {
        return (x*(scale_factors[2]-scale_factors[1]) + scale_factors[1])
    } else {
        return (x * scale_factors[2] + scale_factors[1])
    }
    
}


lstm <- function(tsd, h) {
    ## TODO ##
    # - check if model input is in the correct form (RxC vs CxR)
    # - try a different use case
    
    
    p("TSD Length: ", length(tsd))
    p("TSD Start: ", start(tsd))
    p("TSD End: ", end(tsd))
    p("TSD Frequency: ", frequency(tsd))
    p("H: ", h)

    # In case of predictors that translates to an array of dimensions: (nrow(data) – lag – h + 1, h, 1), where lag = h.
    # lag <- h
    
    # lookback = .01 * length(tsd)
    lookback = h
    horizon = h
    num_samples = 200
    tt_split = .7
    
    ## TRAIN
    split_idx = round(tt_split * length(tsd))
    
    # Compute scale factors only for train
    train_arr = tsd[1:split_idx]
    scale_factors <- get_scale_factors(train_arr)
    tsd_scaled <- sapply(tsd, scale, scale_factors=scale_factors)
    
    start_indexes <- seq(1, split_idx, length.out=num_samples)
    train_mtx<- matrix(nrow = length(start_indexes), ncol = (lookback + horizon))
    for (i in 1:length(start_indexes)){
        train_mtx[i,] <- tsd_scaled[start_indexes[i]:(start_indexes[i] + (lookback + horizon) - 1)]
    }
    # make sure it's numeric
    train_mtx <- train_mtx * 1
    # remove na's if you have them
    if(anyNA(train_mtx)){
        train_mtx <- na.omit(train_mtx)
    }
    # Split data into input (X) and output (y)
    X_train = train_mtx[,1:lookback]
    dim(X_train) = c(nrow(X_train), ncol(X_train), 1)
    y_train = train_mtx[,(lookback+1):ncol(train_mtx)]
    dim(y_train) = c(nrow(y_train), ncol(y_train), 1)
    
    
    ## TEST
    # Do NOT compute scale factors here
    
    start_indexes <- seq(1, split_idx, length.out=(tt_split * num_samples))
    test_mtx<- matrix(nrow = length(start_indexes), ncol = (lookback + horizon))
    for (i in 1:length(start_indexes)){
        test_mtx[i,] <- tsd_scaled[start_indexes[i]:(start_indexes[i] + (lookback + horizon) - 1)]
    }
    # make sure it's numeric
    test_mtx <- test_mtx * 1
    # remove na's if you have them
    if(anyNA(test_mtx)){
        test_mtx <- na.omit(test_mtx)
    }
    # Split data into input (X) and output (y)
    X_test <- test_mtx[,1:lookback]
    y_test <- test_mtx[,(lookback+1):ncol(test_mtx)]
    
    
    ts_scaled = ts(tsd_scaled, start=start(tsd), end=end(tsd), frequency=frequency(tsd))
    ts_train = subset(ts_scaled, start=(1), end=(split_idx))
    ts_test = subset(ts_scaled, start=(split_idx), end=(length(tsd)))
    autoplot(ts_scaled) + autolayer(ts_train) + autolayer(ts_test)
    
    
    
    
    
    
    
    
    
    # scale_factors <- get_scale_factors(tsd)
    # scaled <- sapply(tsd, scale, scale_factors=scale_factors)
    # # scaled = tsd
    # p("Scale Factors: ", scale_factors)
    # 
    # # https://www.kaggle.com/rtatman/beginner-s-intro-to-rnn-s-in-r
    # # https://stats.stackexchange.com/questions/7757/data-normalization-and-standardization-in-neural-networks
    # # get a list of start indexes for our (overlapping) chunks
    # # start_indexes <- seq(1, length(scaled) - lookback - horizon, by=(.001*length(scaled)))
    # start_indexes <- seq(1, length(scaled) - lookback - horizon, length.out=200)
    # # create an empty matrix to store our data in
    # data_matrix<- matrix(nrow = length(start_indexes), ncol = (lookback + horizon))
    # # fill our matrix with the overlapping slices of our dataset
    # for (i in 1:length(start_indexes)){
    #     data_matrix[i,] <- scaled[start_indexes[i]:(start_indexes[i] + (lookback + horizon) - 1)]
    # }
    # 
    # # make sure it's numeric
    # data_matrix <- data_matrix * 1
    # # remove na's if you have them
    # if(anyNA(data_matrix)){
    #     data_matrix <- na.omit(data_matrix)
    # }
    # 
    # # Split data into input (X) and output (y)
    # X <- matrix(data_matrix[,1:lookback])
    # y <- matrix(data_matrix[,(lookback+1):ncol(data_matrix)])
    # # create an index to split our data into testing & training sets
    # 
    # # training data
    # split_idx = round(.9 * nrow(data_matrix))
    # X_train = array(X[1:split_idx,], dim = c(split_idx, lookback, 1))
    # y_train = array(y[1:split_idx,], dim = c(split_idx, horizon, 1))
    # # testing data
    # X_test = array(X[(split_idx + 1):nrow(data_matrix),], dim = c((nrow(data_matrix) - split_idx), lookback, 1))
    # y_test = array(y[(split_idx + 1):nrow(data_matrix),], dim = c((nrow(data_matrix) - split_idx), horizon, 1))
    # 
    
    
    
    ### Model Definition
    # https://stackoverflow.com/questions/38714959/understanding-keras-lstms
    # lstm_model <- keras_model_sequential()
    # lstm_model %>%
    #     layer_lstm(
    #         units = 64, # size of the layer
    #         batch_input_shape = c(batch_size, lookback, 1), # batch size, timesteps, features
    #         return_sequences = FALSE,
    #         # stateful = FALSE,
    #         activation='softmax'
    #     ) %>%
    #     # https://keras.io/api/layers/activations/
    #     # layer_dense(units=horizon, activation='softmax') %>%
    #     # layer_dropout(rate = 0.5) %>%
    #     layer_repeat_vector(horizon) %>%
    #     # fraction of the units to drop for the linear transformation of the inputs
    #     layer_lstm(
    #         units = 64,
    #         return_sequences = TRUE,
    #         # stateful = FALSE
    #     ) %>%
    #     # layer_dropout(rate = 0.5) %>%
    #     # https://stackoverflow.com/questions/53663407/keras-lstm-different-input-output-shape
    #     time_distributed(keras::layer_dense(units = 1))
    
    lstm_model <- keras_model_sequential()
    lstm_model %>%
        layer_lstm(
            units = 64, # size of the layer
            input_shape = c(lookback, 1),
            # batch_input_shape = c(batch_size, lookback, 1), # batch size, timesteps, features
            # batch_size=1,
            # input_shape=c(NULL, 1),
            return_sequences = TRUE,
            # activation='softmax'
        ) %>%
        time_distributed(keras::layer_dense(units = 1))
    
    
    lstm_model %>%
        compile(loss = "mape", optimizer = "adam",)
    summary(lstm_model)
    
    ### Training
    model_to_plot = lstm_model %>% fit(
        x = X_train,
        y = y_train,
        # batch_size = batch_size,
        epochs = 200,
        verbose = 0,
        shuffle = FALSE,
        validation_split=.1
    )
    
    ts_scaled = ts(scaled, start=start(tsd), end=end(tsd), frequency=frequency(tsd))
    
    ### Test Prediction
    ts_x_pred = subset(ts_scaled, start=(length(ts_scaled) - lookback - horizon + 1), end=(length(scaled) - horizon))
    # ts_x_pred = subset(ts_scaled, start=20, end=(20+lookback))
    ts_y_actual = subset(ts_scaled, start=(length(scaled) - horizon), end=(length(scaled)))
    # ts_y_actual = subset(ts_scaled, start=(20+lookback), end=(20+lookback+horizon))
    x_pred = array(data = ts_x_pred, dim = c(1, lookback, 1))
    y_pred = predict(lstm_model, x_pred, batch_size = 1) %>% .[, , 1]
    ts_y_pred = ts(y_pred, start=start(ts_y_actual), end=end(ts_y_actual), frequency=frequency(ts_y_actual))
    autoplot(ts_scaled) + autolayer(ts_x_pred) + autolayer(ts_y_actual) + autolayer(ts_y_pred)
    
    ### Actual Prediction:
    ts_x_pred = subset(ts_scaled, start=(length(ts_scaled) - lookback + 1), end=(length(scaled)))
    x_pred = array(data = ts_x_pred, dim = c(1, lookback, 1))
    y_pred <- predict(lstm_model, x_pred, batch_size = 1) %>% .[, , 1]
    # Unscale
    y_pred_unscaled <- sapply(y_pred, unscale, scale_factors=scale_factors)

    return(y_pred_unscaled)
}

benchmark(lstm, usecase = "human", type = "multi", output = "/Users/prater/Dev/ForecastBenchmark/results/output.csv")
