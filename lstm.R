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

DEBUG = FALSE
IS_COLAB = TRUE

if (IS_COLAB) {
    setwd(".")
    
} else {
    setwd("/Users/prater/Dev/ForecastBenchmark")
}
source("R/benchmark.R")
source("R/forecasting.R")
source("R/measures.R")
load(file = "R/sysdata.rda")
library(forecast)
library(keras)
library(caret)
library(xts)

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
        return (c(mean(x), sd(x)))
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
    # TO TEST #
    # - num units as a fixed number or as a factor of input
    
    ## TODO ##
    # - check if model input is in the correct form (RxC vs CxR)
    # - try a different use case
    # - remove internal testing, make sequence 100% training and use validation split
    
    
    p("TSD Length: ", length(tsd))
    p("TSD Start: ", start(tsd))
    p("TSD End: ", end(tsd))
    p("TSD Frequency: ", frequency(tsd))
    p("H: ", h)
    
    # In case of predictors that translates to an array of dimensions: (nrow(data) – lag – h + 1, h, 1), where lag = h.
    # lag <- h
    
    # lookback = .01 * length(tsd)
    IS_PROD = TRUE
    lookback = h
    horizon = h
    num_samples = 500
    tt_split = .7
    
    if (IS_PROD) {
        tt_split = 1
    } else {
        tt_split = tt_split
    }
    ## TRAIN
    split_idx = round(tt_split * length(tsd))
    
    # Compute scale factors only for train
    train_arr = tsd[1:split_idx]
    scale_factors <- get_scale_factors(train_arr)
    tsd_scaled <- sapply(tsd, scale, scale_factors=scale_factors)
    
    train_seq_indexes <- seq(1, split_idx, length.out=num_samples)
    train_mtx<- matrix(nrow = length(train_seq_indexes), ncol = (lookback + horizon))
    for (i in 1:length(train_seq_indexes)){
        train_mtx[i,] <- tsd_scaled[train_seq_indexes[i]:(train_seq_indexes[i] + (lookback + horizon) - 1)]
    }
    # make sure it's numeric
    train_mtx <- train_mtx * 1
    # remove na's if you have them
    if(anyNA(train_mtx)){
        train_mtx <- na.omit(train_mtx)
    }
    # Split data into input (X) and output (y)
    X_train = train_mtx[,1:lookback]
    X_train = array(X_train, dim=c(nrow(X_train), ncol(X_train), 1))
    y_train = train_mtx[,(lookback+1):ncol(train_mtx)]
    y_train = array(y_train, dim=c(nrow(y_train), ncol(y_train), 1))
    
    ## TEST - DONT IMMEDIATELY DELETE
    # # Do NOT compute scale factors here
    # test_seq_indexes <- seq(split_idx + 1, (length(tsd_scaled) - lookback - horizon), length.out=(tt_split * num_samples))
    # test_mtx<- matrix(nrow = length(test_seq_indexes), ncol = (lookback + horizon))
    # for (i in 1:length(test_seq_indexes)){
    #     test_mtx[i,] <- tsd_scaled[test_seq_indexes[i]:(test_seq_indexes[i] + (lookback + horizon) - 1)]
    # }
    # # make sure it's numeric
    # test_mtx <- test_mtx * 1
    # # remove na's if you have them
    # if(anyNA(test_mtx)){
    #     test_mtx <- na.omit(test_mtx)
    # }
    # # Split data into input (X) and output (y)
    # X_test <- test_mtx[,1:lookback]
    # y_test <- test_mtx[,(lookback+1):ncol(test_mtx)]
    # 
    
    ts_scaled = ts(tsd_scaled, start=start(tsd), end=end(tsd), frequency=frequency(tsd))
    train_ts = subset(ts_scaled, start=(1), end=(split_idx))
    test_ts = subset(ts_scaled, start=(split_idx + 1), end=(length(tsd)))
    autoplot(ts_scaled) + autolayer(train_ts) + autolayer(test_ts)
    
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
    
    num_units = 128
    lstm_model <- keras_model_sequential()
    lstm_model %>%
        layer_lstm(
            units = num_units, # size of the layer
            input_shape = c(lookback, 1),
            return_sequences = TRUE,
            # selu
            activation='selu'
        ) %>%
        # layer_dense(units=30, activation = 'selu') %>%
        layer_dropout(rate = 0.5) %>%
        layer_dense(units=(num_units / 2)) %>%
        time_distributed(keras::layer_dense(units = 1))
    
    
    lstm_model %>%
        compile(loss = "mse", optimizer = "adam",)
    summary(lstm_model)
    
    ### Training
    model_to_plot = lstm_model %>% fit(
        x = X_train,
        y = y_train,
        # batch_size = batch_size,
        epochs = 100,
        verbose = 0,
        shuffle = FALSE,
        validation_split=.1
    )
    
    # ### Test Prediction 1
    # offset = 100
    # ts_x_pred = subset(ts_scaled, start=(length(ts_scaled) - lookback - horizon + 1 - offset), end=(length(ts_scaled) - horizon - offset))
    # # ts_x_pred = subset(ts_scaled, start=20, end=(20+lookback))
    # ts_a_pred = subset(ts_scaled, start=(length(ts_scaled) - horizon + 1 - offset), end=(length(ts_scaled) - offset))
    # # ts_y_actual = subset(ts_scaled, start=(20+lookback), end=(20+lookback+horizon))
    # x_pred = array(data = ts_x_pred, dim = c(1, lookback, 1))
    # y_pred = predict(lstm_model, x_pred, batch_size = 1) %>% .[, , 1]
    # ts_y_pred = ts(y_pred, start=start(ts_a_pred), end=end(ts_a_pred), frequency=frequency(ts_a_pred))
    # autoplot(ts_scaled) + autolayer(ts_x_pred) + autolayer(ts_a_pred) + autolayer(ts_y_pred)
    # 
    # ### Test Prediction 2
    # test_x = array(data = test_ts[1:lookback], dim = c(1, lookback, 1))
    # predict(lstm_model, test_x, batch_size = 1) %>% .[, , 1]
    # # ts_y_pred = ts(y_pred, start=1, end=lookback, frequency=frequency(tsd_scaled))
    
    ### Actual Prediction:
    x_pred = array(data = tsd_scaled[(length(tsd_scaled) - lookback + 1): length(tsd_scaled)], dim = c(1, lookback, 1))
    y_pred <- predict(lstm_model, x_pred, batch_size = 1) %>% .[, , 1]
    # Unscale
    y_pred_unscaled <- sapply(y_pred, unscale, scale_factors=scale_factors)
    
    return(y_pred_unscaled)
}

benchmark(lstm, usecase = "nature", type = "one", output = "./results/output.csv")
