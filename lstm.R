# http://datasideoflife.com/?p=1171

library(ForecastBenchmark)
library(forecast)
library(keras)

lstm <- function(tsd, h) {
    print(paste0("TSD Length: ", length(tsd)))
    print(paste0("TSD Start: ", start(tsd)))
    print(paste0("TSD End: ", end(tsd)))
    print(paste0("TSD Frequency: ", frequency(tsd)))
    print(paste0("H: ", h))

    # In case of predictors that translates to an array of dimensions: (nrow(data) – lag – h + 1, h, 1), where lag = h.
    prediction <- h
    lag <- h

    scale_factors <- c(mean(tsd), sd(tsd))
    print(paste0("Scale Factors: ", scale_factors))

    scaled_train <- sapply(tsd, function(x) (x - scale_factors[1]) / scale_factors[2])
    
    scaled_train <- as.matrix(scaled_train)
    # we lag the data h times and arrange that into columns
    x_train_data <- t(sapply(
        1:(length(scaled_train) - lag - h + 1),
        function(x) scaled_train[x:(x + lag - 1), 1]
    ))
    # now we transform it into 3D form
    x_train_arr <- array(
        data = as.numeric(unlist(x_train_data)),
        dim = c(
            nrow(x_train_data),
            lag,
            1
        )
    )
    
    # Now we apply similar transformation for the Y values.
    y_train_data <- t(sapply(
        (1 + lag):(length(scaled_train) - h + 1),
        function(x) scaled_train[x:(x + h - 1)]
    ))
    y_train_arr <- array(
        data = as.numeric(unlist(y_train_data)),
        dim = c(
            nrow(y_train_data),
            h,
            1
        )
    )
    
    # In the same manner we need to prepare input data for the prediction, which are in fact last 12 observations from our training set.
    x_test <- tsd[(nrow(scaled_train) - h + 1):nrow(scaled_train)]
    
    # scale the data with same scaling factors as for training
    x_test_scaled <- (x_test - scale_factors[1]) / scale_factors[2]
    # this time our array just has one sample, as we intend to perform one 12-months prediction
    x_pred_arr <- array(
        data = x_test_scaled,
        dim = c(
            1,
            lag,
            1
        )
    )
    
    ### LSTM Setup
    lstm_model <- keras_model_sequential()
    lstm_model %>%
        layer_lstm(
            units = 100, # size of the layer
            batch_input_shape = c(1, h, 1), # batch size, timesteps, features
            return_sequences = TRUE,
            stateful = TRUE
        ) %>%
        # fraction of the units to drop for the linear transformation of the inputs
        layer_dropout(rate = 0.5) %>%
        layer_lstm(
            units = 50,
            return_sequences = TRUE,
            stateful = TRUE
        ) %>%
        layer_dropout(rate = 0.5) %>%
        time_distributed(keras::layer_dense(units = 1))
    lstm_model %>%
        compile(loss = "mae", optimizer = "adam", metrics = "accuracy")
    # summary(lstm_model)
    
    ### LSTM Training
    lstm_model %>% fit(
        x = x_train_arr,
        y = y_train_arr,
        batch_size = 1,
        epochs = 100,
        verbose = 0,
        shuffle = FALSE
    )
    
    ### LSTM Prediction
    lstm_forecast <- lstm_model %>%
        predict(x_pred_arr, batch_size = 1) %>%
        .[, , 1]
    # Unscale
    lstm_forecast <- lstm_forecast * scale_factors[2] + scale_factors[1]
    
    ### Forecast
    fitted <- predict(lstm_model, x_train_arr, batch_size = 1) %>% .[, , 1]
    
    if (dim(fitted)[2] > 1) {
        fit <- c(fitted[, 1], fitted[dim(fitted)[1], 2:dim(fitted)[2]])
    } else {
        fit <- fitted[, 1]
    }
    
    # additionally we need to rescale the data
    fitted <- fit * scale_factors[2] + scale_factors[1]
    
    # Specify first forecast values as not available
    fitted <- c(rep(NA, lag), fitted)
    
    input_ts <- head(tsd, length(tsd) - h)
    lstm_forecast <- timetk::tk_ts(lstm_forecast,
                                   start = end(input_ts),
                                   end = end(tsd),
                                   frequency = frequency(tsd)
    )
    
    # Finally we can define the forecast object
    forecast_list <- list(
        model = NULL,
        method = "LSTM",
        mean = lstm_forecast,
        x = input_ts,
        fitted = fitted,
        residuals = as.numeric(input_ts) - as.numeric(fitted)
    )
    class(forecast_list) <- "forecast"
    
    autoplot(tsd, color="green") + autolayer(input_ts, color="black") + autolayer(lstm_forecast, color="red")
    autoplot(forecast_list)
    
    stop()
    return(forecast_list)
}

ets <- function(tsd, h) {
    # return(forecast(ets(tsd), h = h)$mean)
    values <- ets(tsd)
    print(values)
    values <- forecast(values, h = h)$mean
    return(values)
}

theta <- function(tsd, h) {
    values <- thetaf(tsd, h = h)$mean
    print(values)
    plot(tsd)
    stop()
    return(values)
}

benchmark(lstm, usecase = "human", type = "rolling", output = "/Users/prater/Dev/fhnw-aci-libra/results/nature_rolling.csv")