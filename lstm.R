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

DEBUG = TRUE
IS_COLAB = FALSE
IS_PROD = FALSE

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
        if (IS_COLAB) {
            message(msg)
            message(val)
        } else {
            print(paste0(msg, val))
        }
    }
}


SCALER = "second"
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

library(ggplot2)

lstm <- function(ts_input, h) {
    ### NOTES ###
    # impact of params
    # - num units: small
    # - num samples: big
    # - epochs: ?
    # - dense layers: ?
    # - activation: `selu` big
    
    # TO TEST #
    # - num units as a fixed number or as a factor of input
    
    ## TODO ##
    # - check if model input is in the correct form (RxC vs CxR)
    # - try a different use case
    # - remove internal testing, make sequence 100% training and use validation split
    
    # Reset graphics (plotting)
    # dev.off()
    
    p("TSD Length: ", length(ts_input))
    p("TSD Start: ", start(ts_input))
    p("TSD End: ", end(ts_input))
    p("TSD Frequency: ", frequency(ts_input))
    p("H: ", h)
    
    horizon = h
    lookback = horizon
    # # lookback = as.integer(.3 * length(ts_input))
    # num_samples = length(ts_input) / 2
    if (IS_PROD) {
        tt_split = 1
        build_test = FALSE
    } else {
        tt_split = .9
        build_test = TRUE
    }
    
    # Uncomment to test with different dataset
    ts_input <- ts(ggplot2::economics$unemploy,
                   start = c(1967, 7),
                   end = c(2015, 4),
                   frequency = 12)
    
    input = as.array(ts_input)
    
    #### BUILD TRAIN
    # Get scale factors
    split_idx = as.integer(tt_split * length(input))
    scale_factors <- get_scale_factors(input[1:length(input)])

    # Create scaled versions of input data    
    ts_input_scaled = ts(sapply(input, scale, scale_factors=scale_factors), start=start(ts_input), end=end(ts_input), frequency=frequency(ts_input))
    input_scaled = as.array(ts_input_scaled)
    
    # Create scaled versions of training data
    ts_train = subset(ts_input, start=1, end=split_idx)
    train = as.array(ts_train)
    ts_train_scaled = subset(ts_input_scaled, start=1, end=split_idx)
    train_scaled = as.array(ts_train_scaled)

    train_idxs <- seq(1, split_idx, by=(lookback))
    train_mtx<- matrix(nrow = length(train_idxs), ncol = (lookback + horizon))
    for (i in 1:length(train_idxs)){
        train_mtx[i,] <- input_scaled[train_idxs[i]:(train_idxs[i] + (lookback + horizon) - 1)]
    }
    # make sure it's numeric
    train_mtx <- train_mtx * 1
    # remove na's if you have them
    if(anyNA(train_mtx)){
        train_mtx <- na.omit(train_mtx)
    }
    # Split data into input (X) and output (y)
    X_train = train_mtx[,1:lookback]
    stopifnot(ncol(X_train) == lookback)
    X_train = array(X_train, dim=c(nrow(X_train), ncol(X_train), 1))
    y_train = train_mtx[,(lookback+1):ncol(train_mtx)]
    stopifnot(ncol(y_train) == horizon)
    y_train = array(y_train, dim=c(nrow(y_train), ncol(y_train), 1))
    p("Num samples: ", nrow(X_train))
    
    #### BUILD TEST
    if (build_test) {
        ts_test = subset(ts_input, start=(split_idx + 1), end=(length(ts_input)))
        test = as.array(ts_test)
        ts_test_scaled = subset(ts_input_scaled, start=(split_idx + 1), end=(length(ts_input)))
        test_scaled = as.array(ts_test_scaled)
        
        offset = 20
        y_test_end_idx = length(test_scaled) - offset
        y_test_start_idx = y_test_end_idx - horizon
        x_test_end_idx = y_test_start_idx - 1
        x_test_start_idx = x_test_end_idx - lookback
        
        ts_x_test = subset(ts_test, start=x_test_start_idx, end=x_test_end_idx)
        x_test = as.array(ts_x_test)
        ts_x_test_scaled = subset(ts_test_scaled, start=x_test_start_idx, end=x_test_end_idx)
        x_test_scaled = as.array(ts_x_test_scaled)
        # Check that lengths are all equal
        stopifnot(length(unique(c(length(ts_x_test), length(x_test), length(ts_x_test_scaled), length(x_test_scaled))))==1)
        
        ts_y_test = subset(ts_test, start=y_test_start_idx, end=y_test_end_idx)
        y_test = as.array(ts_y_test)
        ts_y_test_scaled = subset(ts_test_scaled, start=y_test_start_idx, end=y_test_end_idx)
        y_test_scaled = as.array(ts_y_test_scaled)
        # Check that lengths are all equal
        stopifnot(length(unique(c(length(ts_y_test), length(y_test), length(ts_y_test_scaled), length(y_test_scaled))))==1)
        
        X_test <- array(
            data = x_test,
            dim = c(
                1,
                lookback,
                1
            )
        )
    }
    
    # Plot train/test split
    autoplot(ts_input) + autolayer(ts_train) + autolayer(ts_test)
    autoplot(ts_input_scaled) + autolayer(ts_train_scaled) + autolayer(ts_test_scaled)
    
    
 
    #### Model Definition
    # https://stackoverflow.com/questions/38714959/understanding-keras-lstms
    # https://keras.io/api/layers/activations/
    # https://stackoverflow.com/questions/53663407/keras-lstm-different-input-output-shape
    num_epochs = 150
    # batch_size = nrow(X_train)
    lstm_model <- keras_model_sequential()
    
    #### ENCODER / DECODER
    lstm_model %>%
        layer_lstm(
            units = lookback,
            input_shape = c(lookback, 1),
            # batch_input_shape = c(1, lookback, 1),
            return_sequences = FALSE,
            # activation="selu",
            # activation="tanh",
            # recurrent_activation="sigmoid",
            # recurrent_dropout=0.5,
            # unroll=FALSE,
            # use_bias=TRUE,
        ) %>%
        # layer_lstm(
        #     units = lookback * 2,
        #     input_shape = c(lookback, 1),
        #     return_sequences = FALSE,
        #     activation="selu",
        #     # activation="tanh",
        #     # recurrent_activation="sigmoid",
        #     # recurrent_dropout=0.5,
        #     # unroll=FALSE,
        #     # use_bias=TRUE,
        # ) %>%
        # layer_dense(units=horizon) %>%
        # Reshape to output of size `horizon`
        layer_repeat_vector(horizon) %>%
        layer_lstm(
            units = horizon,
            return_sequences = TRUE,
            # activation="selu",
            # activation="tanh",
            # recurrent_activation="sigmoid",
            # recurrent_dropout=0.5,
            # unroll=FALSE,
            # use_bias=TRUE,
        ) %>%
        # layer_dense(units=(horizon/2)) %>%
    
    time_distributed(keras::layer_dense(units = 1))
    lstm_model %>%
        compile(loss = 'mape', optimizer = 'adam', metrics = c('mae', 'mape'))
    # compile(loss = "mse", optimizer = optimizer_adam(lr = 0.0001))
    summary(lstm_model)
    
    #### MINE
    early_stop = callback_early_stopping(monitor = "loss", patience=(0.2 * num_epochs))

    ### Training
    p("Starting training", NULL)
    train_result = lstm_model %>% fit(
        x = X_train,
        y = y_train,
        epochs = num_epochs,
        # try varying this
        batch_size = 1,
        verbose = 0,
        shuffle = FALSE,
        validation_split=.2,
        callbacks = c(early_stop)
    )
    plot(train_result)
    # p("Training complete", NULL)
    

    #### PREDICT
    if (build_test) {
        # Make prediction
        pred_scaled <- lstm_model %>% predict(X_test, batch_size = 1) %>% .[, , 1]
        # TODO: unscale with method
        pred <- pred_scaled * scale_factors[2] + scale_factors[1]
        
        # Build ts versions
        ts_p_test_scaled = ts(pred_scaled, start=start(ts_y_test_scaled), end=end(ts_y_test_scaled), frequency=frequency(ts_y_test_scaled))
        ts_p_test = ts(pred, start=start(ts_y_test), end=end(ts_y_test), frequency=frequency(ts_y_test))
        
        autoplot(ts_input) + autolayer(ts_train) + autolayer(ts_x_test) + autolayer(ts_y_test) + autolayer(ts_p_test)
        autoplot(ts_input_scaled) + autolayer(ts_train_scaled) + autolayer(ts_x_test_scaled) + autolayer(ts_y_test_scaled) + autolayer(ts_p_test_scaled)
    } else {
        # pred_scaled <- lstm_model %>% predict(X_test, batch_size = 1) %>% .[, , 1]
        
    }

    return(lstm_forecast)
}

benchmark(lstm, usecase = "finance", type = "multi", output = "./results/output.csv")
